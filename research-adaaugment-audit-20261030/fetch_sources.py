"""Read-only retrieval of pinned public text; never imports downloaded code."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "d854af15467285ad8f46500d6fc2da5225fb5bd0"


def main():
    tree = json.loads((ROOT / "sources/tree.json").read_text())
    assert tree["sha"] == COMMIT and tree["truncated"] is False
    records = json.loads((ROOT / "source-receipts.json").read_text())
    selected = [e for e in tree["tree"] if e["type"] == "blob" and
                (e["path"].endswith((".py", ".yaml")) or e["path"] == "README.md")]
    for entry in selected + [{"path": "paper-2405.11467v3.html"}]:
        rel = entry["path"]
        url = (f"https://raw.githubusercontent.com/Jackbrocp/AdaAugment/{COMMIT}/{rel}"
               if "sha" in entry else "https://arxiv.org/html/2405.11467v3")
        request = urllib.request.Request(url, headers={"User-Agent": "Bounded-Source-Audit/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(4_000_001)
            status, final_url = response.status, response.url
        assert len(body) <= 4_000_000
        body.decode("utf-8")
        blob_sha1 = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if "sha" in entry:
            assert blob_sha1 == entry["sha"]
        target = ROOT / "sources" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        records.append({"path": str(target.relative_to(ROOT)), "url": url,
                        "final_url": final_url, "http_status": status,
                        "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                        "git_blob_sha1": blob_sha1 if "sha" in entry else None,
                        "verified_against_pinned_tree": "sha" in entry,
                        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        (ROOT / "source-receipts.json").write_text(json.dumps(records, indent=2) + "\n")
        print(rel, len(body), records[-1]["sha256"], flush=True)


if __name__ == "__main__":
    main()
