from __future__ import annotations

import sys

MESSAGE = "This CLI moved to: python -m building.cli.geometry_smoke_fixed_allocation"


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
