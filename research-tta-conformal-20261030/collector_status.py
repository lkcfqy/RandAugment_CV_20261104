import json
from pathlib import Path
r=Path('/content/tta_conformal_20261030')
events=r/'output/events.jsonl'
last=json.loads(events.read_text().splitlines()[-1]) if events.exists() and events.stat().st_size else None
bundle=None
log=r/'process.log'
if log.exists():
    for line in log.read_text().splitlines()[-3:]:
        try:
            item=json.loads(line)
            if 'BUNDLE_READY' in item:bundle=item
        except json.JSONDecodeError:pass
print(json.dumps({'last_event':last,'bundle':bundle}))
