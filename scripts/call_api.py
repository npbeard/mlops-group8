"""
Minimal client for sending JSON requests to the deployed FastAPI service.

Usage:
python scripts/call_api.py --url \
    https://your-render-service.onrender.com/predict
"""

import argparse
import json
from urllib import request


DEFAULT_PAYLOAD = {
    "instances": [
        {
            "acousticness": 0.12,
            "danceability": 0.65,
            "duration_ms": 200000,
            "energy": 0.71,
            "instrumentalness": 0.0,
            "key": 5,
            "liveness": 0.11,
            "loudness": -5.1,
            "mode": 1,
            "speechiness": 0.05,
            "tempo": 120.5,
            "valence": 0.44,
        }
    ]
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a prediction request."
    )
    parser.add_argument(
        "--url", required=True, help="Full /predict endpoint URL"
    )
    args = parser.parse_args()

    body = json.dumps(DEFAULT_PAYLOAD).encode("utf-8")
    req = request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
