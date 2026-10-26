"""
Fetch the trained LoRA weights and tokenizer into models/pos-slm-llama3b/adapter/.

The weights (195 MB) and the tokenizer (17 MB) are too large to keep as normal git files,
so they live on the GitHub release v1.0-adapter. This script downloads them next to the
adapter config and checks each SHA-256 against adapter_files.sha256.

Usage:  python models/download_adapter.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "models" / "pos-slm-llama3b" / "adapter"
RELEASE = "https://github.com/bjjkim91-dev/pos-cleanscan/releases/download/v1.0-adapter/"
FILES = ["adapter_model.safetensors", "tokenizer.json"]
SHA_FILE = ADAPTER / "adapter_files.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_hashes() -> dict:
    if not SHA_FILE.exists():
        return {}
    out = {}
    for line in SHA_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def fetch(name: str, expected: str | None):
    target = ADAPTER / name
    if target.exists() and expected and sha256(target) == expected:
        print(f"already present and verified: {target.name}")
        return
    url = RELEASE + name
    print(f"downloading {url}")
    tmp = target.with_suffix(target.suffix + ".part")

    def progress(blocks, block_size, total):
        if total > 0:
            sys.stdout.write(f"\r  {blocks * block_size / 1e6:7.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    got = sha256(tmp)
    if expected and got != expected:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {name}: expected {expected}, got {got}")
    tmp.replace(target)
    print(f"saved {target.name} ({target.stat().st_size / 1e6:.1f} MB, sha256 {got[:12]}...)")


def main():
    ADAPTER.mkdir(parents=True, exist_ok=True)
    hashes = expected_hashes()
    for name in FILES:
        fetch(name, hashes.get(name))


if __name__ == "__main__":
    main()
