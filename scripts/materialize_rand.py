"""Rebuild the exact, real RAND HIE research CSV from statsmodels 0.14.6.

No synthetic observations are generated. Fail closed if source bytes differ.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import statsmodels
from statsmodels.datasets import randhie

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "rand_hie_statsmodels_sample.csv"
EXPECTED = "6c58ff02c7a568c267e93c3fffc517daba34239da2fed6287e94098b08471a5d"

if statsmodels.__version__ != "0.14.6":
    raise RuntimeError(
        "For verified data export, install statsmodels==0.14.6; "
        f"current version is {statsmodels.__version__}"
    )
df = randhie.load_pandas().data
if df.shape != (20190, 10):
    raise RuntimeError(f"Unexpected historic RAND sample shape: {df.shape}")
payload = df.to_csv(index=False, float_format="%.8g").encode("utf-8")
digest = hashlib.sha256(payload).hexdigest()
if digest != EXPECTED:
    raise RuntimeError(f"Archived sample differs from expected SHA-256: {digest}")
TARGET.parent.mkdir(exist_ok=True, parents=True)
TARGET.write_bytes(payload)
print(f"Verified real historical RAND dataset: {df.shape}; sha256={digest}")
