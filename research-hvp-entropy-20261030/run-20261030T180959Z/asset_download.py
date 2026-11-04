"""Download only the authorized official DINO sources and checkpoint; execute no source."""
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SOURCES = ROOT / "sources"
LIMIT = 3 * 1024**3
COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
BASE = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/"
CHECKPOINT = BASE + "dino_deitsmall16_pretrain_full_checkpoint.pth"
ALLOWED = {"raw.githubusercontent.com", "dl.fbaipublicfiles.com"}

def stamp():
    return datetime.now(timezone.utc).isoformat()

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def valid(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED:
        raise RuntimeError("Unapproved source/redirect: " + url)

class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        valid(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def size_now():
    return sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())

def download(url, target, expected=None):
    valid(url)
    opener = urllib.request.build_opener(CheckedRedirect())
    begin = time.monotonic()
    with opener.open(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
        valid(response.url)
        length = int(response.headers["Content-Length"])
        head = {"url": response.url, "status": response.status, "headers": dict(response.headers)}
    if length <= 0 or size_now() + length > LIMIT:
        raise RuntimeError("Disk budget or content-length gate failed")
    if expected is not None and length != expected:
        raise RuntimeError(f"HEAD size changed: {length} != {expected}")
    receipt = {"requested_url": url, "head": head, "expected_bytes": length, "started_utc": stamp()}
    save("current_download.json", receipt)
    digest = hashlib.sha256()
    count = 0
    partial = target.with_suffix(target.suffix + ".partial")
    with opener.open(url, timeout=45) as response, partial.open("xb") as out:
        valid(response.url)
        receipt["final_url"] = response.url
        receipt["final_domain"] = urllib.parse.urlparse(response.url).hostname
        receipt["response_headers"] = dict(response.headers)
        if int(response.headers["Content-Length"]) != length:
            raise RuntimeError("GET length differs from HEAD")
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > length or size_now() > LIMIT or time.time() > FIRST["epoch"] + 7200 - 120:
                raise RuntimeError("Size or wall budget exceeded during download")
            out.write(chunk)
            digest.update(chunk)
    if count != length:
        raise RuntimeError("Downloaded size differs from Content-Length")
    partial.rename(target)
    receipt.update(bytes=count, sha256_local=digest.hexdigest(), sha256_is_author_published=False,
                   finished_utc=stamp(), elapsed_seconds=time.monotonic() - begin,
                   relative_path=target.relative_to(ROOT).as_posix())
    return receipt

if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    SOURCES.mkdir(exist_ok=True)
    FIRST = {"utc": stamp(), "epoch": time.time(), "deadline_seconds": 7200}
    if (ROOT / "first_download.json").exists():
        raise SystemExit("Refusing to restart a run or overwrite the clock")
    save("first_download.json", FIRST)
    receipts = []
    try:
        for name in ("README.md", "LICENSE"):
            receipts.append(download(f"https://raw.githubusercontent.com/facebookresearch/dino/{COMMIT}/{name}", SOURCES / name))
            save("asset_download_receipts.json", receipts)
        readme = (SOURCES / "README.md").read_text()
        assert CHECKPOINT in readme and BASE + "args.txt" in readme
        receipts.append(download(BASE + "args.txt", SOURCES / "native_args.json"))
        save("asset_download_receipts.json", receipts)
        args = json.loads((SOURCES / "native_args.json").read_text())
        required = json.loads((ROOT / "PROTOCOL.json").read_text())["native_args_expected"]
        assert all(args[k] == v for k, v in required.items())
        receipts.append(download(CHECKPOINT, ASSETS / "dino_deitsmall16_pretrain_full_checkpoint.pth", 352247243))
        save("asset_download_receipts.json", receipts)
        print(json.dumps({"download_complete": True, "bytes": receipts[-1]["bytes"], "sha256_local": receipts[-1]["sha256_local"], "wall_seconds": time.time() - FIRST["epoch"]}), flush=True)
    except Exception as exc:
        save("download_failure.json", {"type": type(exc).__name__, "error": str(exc), "utc": stamp()})
        raise
