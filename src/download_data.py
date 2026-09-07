"""Download the public IBM Telco Customer Churn dataset reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def download(destination: str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, path)
    rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if rows != 7_043:
        raise ValueError(f"Unexpected row count: {rows}")
    print(f"Downloaded {rows} rows to {path} (sha256={digest})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw.csv")
    download(parser.parse_args().output)
