#!/usr/bin/env python3

"""V12 NEW-only training entry point.

This keeps the established training implementation while making it impossible
to accidentally use the OLD/HaGRID data path.
"""

from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    if "--new-only" not in sys.argv:
        sys.argv.append("--new-only")
    main()
