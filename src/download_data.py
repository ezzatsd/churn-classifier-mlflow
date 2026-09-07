"""Download the public IBM Telco Customer Churn dataset reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import ssl
import tempfile
import urllib.request
from pathlib import Path

import certifi

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def download(destination: str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "churn-classifier/1.0"})
    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=30, context=tls_context) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Dataset checksum mismatch: expected {SHA256}, got {digest}")

    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)

    rows = payload.count(b"\n") - 1
    if rows != 7_043:
        path.unlink(missing_ok=True)
        raise ValueError(f"Unexpected row count: {rows}")
    print(f"Downloaded {rows} rows to {path} (sha256={digest})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw.csv")
    download(parser.parse_args().output)
