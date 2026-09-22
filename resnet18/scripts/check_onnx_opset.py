#!/usr/bin/env python3
"""Inspect an ONNX file and optionally enforce its default opset version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Load, validate, and report the ONNX model metadata."""
    parser = argparse.ArgumentParser(description="Check ONNX model opset version(s).")
    parser.add_argument("onnx_file", type=Path, help="Path to .onnx file")
    parser.add_argument("--expect", type=int, default=None, help="Expected ai.onnx opset version, e.g. 13")
    parser.add_argument("--no-checker", action="store_true", help="Skip onnx.checker.check_model")
    parser.add_argument("--verbose", action="store_true", help="Print producer info and node/op counts")
    args = parser.parse_args()

    if not args.onnx_file.exists():
        print(f"ERROR: file not found: {args.onnx_file}", file=sys.stderr)
        return 2

    try:
        import onnx
    except ImportError:
        print("ERROR: Python package 'onnx' is not installed.", file=sys.stderr)
        print("Install with: python3 -m pip install onnx", file=sys.stderr)
        return 2

    try:
        model = onnx.load(str(args.onnx_file))
    except Exception as exc:
        print(f"ERROR: could not load ONNX file: {exc}", file=sys.stderr)
        return 2

    print(f"File: {args.onnx_file}")
    print(f"IR version: {model.ir_version}")

    if args.verbose:
        print(f"Producer: {model.producer_name} {model.producer_version}".strip())
        if model.domain:
            print(f"Model domain: {model.domain}")
        if model.model_version:
            print(f"Model version: {model.model_version}")

    opsets: dict[str, int] = {}
    print("Opset imports:")
    # ONNX can declare several domains; the default ai.onnx domain is the one we verify.
    for item in model.opset_import:
        domain = item.domain or "ai.onnx"
        opsets[domain] = item.version
        print(f"  {domain}: {item.version}")

    main_opset = opsets.get("ai.onnx")
    if main_opset is None:
        print("ERROR: no default ai.onnx opset import found.", file=sys.stderr)
        return 1

    if not args.no_checker:
        # Metadata alone is not enough; the checker catches malformed graph structure.
        try:
            onnx.checker.check_model(model)
            print("ONNX checker: OK")
        except Exception as exc:
            print(f"ONNX checker: FAILED: {exc}", file=sys.stderr)
            return 1

    if args.verbose:
        op_counts: dict[str, int] = {}
        for node in model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        print(f"Graph nodes: {len(model.graph.node)}")
        print("Operator counts:")
        for op, count in sorted(op_counts.items()):
            print(f"  {op}: {count}")

    if args.expect is not None:
        if main_opset == args.expect:
            print(f"Expected opset {args.expect}: OK")
            return 0
        print(f"Expected opset {args.expect}: FAILED, got {main_opset}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
