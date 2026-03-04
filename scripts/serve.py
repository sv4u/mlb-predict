"""Start the MLB Win Probability web dashboard.

Usage
-----
    python scripts/serve.py                   # default: localhost:8087
    python scripts/serve.py --port 9000       # custom port
    python scripts/serve.py --model xgboost   # use a different model type
    python scripts/serve.py --model lightgbm  # LightGBM model
    python scripts/serve.py --reload          # auto-reload on code changes (dev)
    python scripts/serve.py --verbose         # DEBUG-level logging

Background / daemon
-------------------
    # Start in background and save PID
    nohup python scripts/serve.py --model xgboost >> logs/server.log 2>&1 &
    echo $! > server.pid

    # Kill the server
    kill $(cat server.pid)              # graceful
    kill -9 $(cat server.pid)           # force
    kill $(lsof -ti:8087)              # by port
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))


def main() -> None:
    """Parse arguments, configure logging, and start the Uvicorn server."""
    ap = argparse.ArgumentParser(description="Start the MLB Win Probability dashboard.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8087)
    ap.add_argument(
        "--model",
        default="stacked",
        choices=["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"],
    )
    ap.add_argument("--reload", action="store_true")
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging for all modules.",
    )
    ap.add_argument(
        "--log-format",
        default="auto",
        choices=["human", "json", "auto"],
        help="Log output format (default: auto — human locally, json in production).",
    )
    args = ap.parse_args()

    os.environ.setdefault("WINPROB_MODEL_TYPE", args.model)

    from winprob.logging_config import setup_logging

    setup_logging(verbose=args.verbose or None, log_format=args.log_format)

    import uvicorn  # type: ignore

    log_level = "debug" if args.verbose else "info"

    uvicorn.run(
        "winprob.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
