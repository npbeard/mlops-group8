"""
Promote a W&B model artifact to the configured production alias.

Usage:
python scripts/promote_model.py --source latest --target prod
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import load_config
from src.model_registry import promote_model_artifact

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at runtime
    load_dotenv = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project config file",
    )
    parser.add_argument(
        "--source",
        default="latest",
        help="Candidate alias or version to promote, "
             "for example latest or v12",
    )
    parser.add_argument(
        "--target",
        default="prod",
        help="Alias to attach to the promoted artifact",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if load_dotenv is not None:
        load_dotenv(
            dotenv_path=config_path.resolve().parent / ".env",
            override=False,
        )

    config = load_config(config_path)
    result = promote_model_artifact(
        config,
        source_alias=args.source,
        target_alias=args.target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
