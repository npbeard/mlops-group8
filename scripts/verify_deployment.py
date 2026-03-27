"""
Smoke-test the deployed FastAPI service and confirm it serves the
promoted model.

Usage:
python scripts/verify_deployment.py \
    --base-url https://mlops-group8-1.onrender.com
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment_verifier import DEFAULT_PREDICT_PAYLOAD, verify_deployment


def _load_payload(path: str | None) -> dict:
    if path is None:
        return DEFAULT_PREDICT_PAYLOAD

    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="https://mlops-group8-1.onrender.com",
        help=(
            "Base deployment URL, for example "
            "https://mlops-group8-1.onrender.com"
        ),
    )
    parser.add_argument(
        "--payload-file",
        help="Optional JSON file to send to /predict",
    )
    parser.add_argument(
        "--expect-source",
        default="wandb",
        help="Expected model source reported by the deployment",
    )
    parser.add_argument(
        "--expect-alias",
        default="prod",
        help="Expected promoted alias reported by /predict",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()

    result = verify_deployment(
        args.base_url,
        payload=_load_payload(args.payload_file),
        expected_source=args.expect_source,
        expected_alias=args.expect_alias,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
