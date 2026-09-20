
import sys

from .train import parse_args, run_args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = parse_args(argv)
    run_args(opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())