import json
from pathlib import Path
r=Path('/content/tta_conformal_20261030')
p=r/'output/events.jsonl'
print('EVENTS:', '\n'.join(p.read_text().splitlines()[-4:]) if p.exists() else 'not yet created')
state=r/'output/run-state.json'
print('STATE:',state.read_text() if state.exists() else 'RUNNING')
log=r/'process.log'
print('LOG_TAIL:', '\n'.join(log.read_text().splitlines()[-3:]) if log.exists() else 'missing')
