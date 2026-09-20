
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hid_dim] * (n_layers - 1) + [in_dim]
        for i in range(n_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, edge_index):
        x = H
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class ClusterHead(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, K: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, K)
        self.dropout = nn.Dropout(dropout)

    def set_centroids(self, centroids):
        with torch.no_grad():
            c = centroids.to(self.proj.weight.device)
            b = -0.5 * (c * c).sum(dim=1)
            self.proj.weight.copy_(c)
            self.proj.bias.copy_(b)

    def forward(self, Z, tau: float):
        return F.softmax(self.dropout(self.proj(Z)) / tau, dim=-1)


class AdaptiveCommunityLearner(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, K: int, n_layers: int = 2,
                 dropout: float = 0.1, dual_view: bool = True):
        super().__init__()
        self.dual_view = dual_view
        self.enc = GraphEncoder(in_dim, hid_dim, n_layers=n_layers, dropout=dropout)
        self.head = ClusterHead(in_dim, hid_dim, K, dropout=dropout)
        self.head_attr = ClusterHead(in_dim, hid_dim, K, dropout=dropout)

    def forward(self, H, edge_index, tau: float):
        Z = self.enc(H, edge_index)
        S_struct = self.head(Z, tau)
        S_attr = self.head_attr(H, tau) if self.dual_view else S_struct
        comm_mean = (S_struct.t() @ Z) / (S_struct.t().sum(dim=1, keepdim=True) + 1e-8)
        comm_mean_emb = S_struct @ comm_mean
        return Z, S_struct, S_attr, comm_mean, comm_mean_emb

    @torch.no_grad()
    def kmeans_init(self, H, edge_index, K, seed: int = 0):
        Z = self.enc(H, edge_index).cpu().numpy().astype(np.float64)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(Z)
            centroids = torch.from_numpy(km.cluster_centers_).float()
        except Exception:
            idx = np.random.RandomState(seed).choice(Z.shape[0], K, replace=False)
            centroids = torch.from_numpy(Z[idx]).float()
        b = -0.5 * (centroids * centroids).sum(dim=1)
        logits = Z @ centroids.numpy().T + b.numpy()
        scale = float(np.std(logits)) + 1e-6
        centroids_scaled = centroids / scale
        b_scaled = b / scale
        with torch.no_grad():
            last = self.head.proj
            last.weight.copy_(centroids_scaled.to(last.weight.device))
            last.bias.copy_(b_scaled.to(last.bias.device))

        if not self.dual_view:
            return
        H_np = H.cpu().numpy().astype(np.float64)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, n_init=4, random_state=seed + 1).fit(H_np)
            centroids_attr = torch.from_numpy(km.cluster_centers_).float()
        except Exception:
            idx = np.random.RandomState(seed + 1).choice(H_np.shape[0], K, replace=False)
            centroids_attr = torch.from_numpy(H_np[idx]).float()
        b_attr = -0.5 * (centroids_attr * centroids_attr).sum(dim=1)
        logits_attr = H_np @ centroids_attr.numpy().T + b_attr.numpy()
        scale_attr = float(np.std(logits_attr)) + 1e-6
        centroids_attr_scaled = centroids_attr / scale_attr
        b_attr_scaled = b_attr / scale_attr
        with torch.no_grad():
            last_attr = self.head_attr.proj
            last_attr.weight.copy_(centroids_attr_scaled.to(last_attr.weight.device))
            last_attr.bias.copy_(b_attr_scaled.to(last_attr.weight.device))


class ContextEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATConv(in_dim, hid_dim // heads, heads=heads,
                           add_self_loops=False, dropout=dropout)
        self.glob = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, edge_index, comm_mean_emb):
        c_l = self.dropout(self.gat(H, edge_index))
        c_g = self.dropout(self.glob(comm_mean_emb))
        return c_l, c_g


class ContextReasoner(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, max_neigh: int,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 use_local: bool = True, use_global: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.max_neigh = max_neigh
        self.use_local = use_local
        self.use_global = use_global
        self.neigh_proj = nn.Linear(in_dim, hid_dim)
        self.comm_tok = nn.Linear(hid_dim, hid_dim)
        self.local_tok = nn.Linear(hid_dim, hid_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hid_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, nhead=nhead, dim_feedforward=2 * hid_dim,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hid_dim)
        self.out = nn.Linear(hid_dim, in_dim)

    def forward(self, H, nbr_idx, nbr_mask, c_l, c_g):
        N = H.size(0)
        K = self.max_neigh
        pad = nbr_idx < 0
        safe_idx = nbr_idx.clamp(min=0)
        neigh_h = self.neigh_proj(H[safe_idx])
        neigh_h = neigh_h.masked_fill(pad.unsqueeze(-1), 0.0)
        comm_t = self.comm_tok(c_g).unsqueeze(1)
        local_t = self.local_tok(c_l).unsqueeze(1)
        q = self.query.expand(N, 1, -1)
        tokens = torch.cat([neigh_h, comm_t, local_t, q], dim=1)

        special_keep = torch.ones(N, 3, dtype=torch.bool, device=H.device)
        if not self.use_global:
            special_keep[:, 0] = False
        if not self.use_local:
            special_keep[:, 1] = False
        keep = torch.cat([nbr_mask, special_keep], dim=1)
        kpm = ~keep
        kpm[:, -1] = False
        out = self.encoder(tokens, src_key_padding_mask=kpm)
        h_hat = self.out(self.norm(out[:, -1, :]))
        h_hat_norm = F.normalize(h_hat, dim=-1)
        return h_hat, h_hat_norm


class CounterfactualGenerator(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, K: int, dropout: float = 0.1):
        super().__init__()
        self.comm_emb = nn.Embedding(K, hid_dim)
        nn.init.normal_(self.comm_emb.weight, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + hid_dim, hid_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid_dim, in_dim))

    def forward(self, H, S_cf):
        e_cf = S_cf @ self.comm_emb.weight
        return self.mlp(torch.cat([H, e_cf], dim=-1))


class SeverityCalibrator(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hid_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid_dim, 1))

    def forward(self, h_pred, h_cf):
        x = torch.cat([h_pred, h_cf], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


class CARN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, max_neigh: int, K: int,
                 heads: int = 4, nhead: int = 4, num_layers: int = 2,
                 gnn_layers: int = 2, dropout: float = 0.1,
                 cf_mode: str = "intervene", ablate: str = "none",
                 dual_view: bool = True, use_local: bool = True, use_global: bool = True):
        super().__init__()
        self.comm_learner = AdaptiveCommunityLearner(
            in_dim, hid_dim, K, n_layers=gnn_layers, dropout=dropout, dual_view=dual_view)
        self.ctx_enc = ContextEncoder(in_dim, hid_dim, heads=heads, dropout=dropout)
        self.reasoner = ContextReasoner(in_dim, hid_dim, max_neigh, nhead=nhead,
                                        num_layers=num_layers, dropout=dropout,
                                        use_local=use_local, use_global=use_global)
        self.cf_mode = cf_mode
        self.ablate = ablate
        if cf_mode == "generator":
            self.cf_generator = CounterfactualGenerator(in_dim, hid_dim, K, dropout=dropout)
        self.K = K

    def forward(self, H, edge_index, nbr_idx, nbr_mask, tau: float):
        Z, S_struct, S_attr, comm_mean, comm_mean_emb = self.comm_learner(H, edge_index, tau)

        c_l = self.ctx_enc.dropout(self.ctx_enc.gat(H, edge_index))

        c_g_fact = self.ctx_enc.dropout(self.ctx_enc.glob(comm_mean_emb))
        h_pred, h_pred_norm = self.reasoner(H, nbr_idx, nbr_mask, c_l, c_g_fact)

        if self.ablate in ("no_cf", "no_intervention"):
            return h_pred, h_pred_norm, None, None, S_struct, S_attr, comm_mean, Z

        if self.cf_mode == "generator":
            h_cf = self.cf_generator(H, S_attr)
            h_cf_norm = F.normalize(h_cf, dim=-1)
        else:
            comm_mean_attr = (S_attr.t() @ Z) / (S_attr.t().sum(dim=1, keepdim=True) + 1e-8)
            comm_mean_emb_cf = S_attr @ comm_mean_attr
            c_g_cf = self.ctx_enc.dropout(self.ctx_enc.glob(comm_mean_emb_cf))
            h_cf, h_cf_norm = self.reasoner(H, nbr_idx, nbr_mask, c_l, c_g_cf)

        return h_pred, h_pred_norm, h_cf, h_cf_norm, S_struct, S_attr, comm_mean, Z