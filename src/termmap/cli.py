"""Command-line entry point: `termmap audit | build-index | train | predict`."""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="termmap")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="data audit (aggregate stats) -> reports/data_audit.json")
    sub.add_parser("build-index", help="build retrieval indexes from the dictionaries")
    sub.add_parser("train", help="all stages: candidates, cross-encoders, LambdaRank, calibration, final artifacts")
    sub.add_parser("retrieval-cv", help="M1 retrieval study: 5-fold CV with channel/variant ablations")
    pr = sub.add_parser("predict", help="predict both codes for every input row")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    args = p.parse_args(argv)

    import logging

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from termmap.config import load_config, set_seeds
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])

    if args.cmd == "build-index":
        from termmap.retrieval.engine import build_index
        build_index(cfg)
    elif args.cmd == "train":
        from termmap.train import train
        train(cfg)
    elif args.cmd == "retrieval-cv":
        from termmap.train import retrieval_cv
        retrieval_cv(cfg)
    elif args.cmd == "predict":
        from termmap.pipeline import predict_file
        print(json.dumps(predict_file(cfg, args.input, args.output)))
    elif args.cmd == "audit":
        from termmap.audit import run
        print(json.dumps(run(args.config), indent=2, default=str))
    else:
        raise SystemExit(f"'{args.cmd}' is implemented in a later milestone")


if __name__ == "__main__":
    main()
