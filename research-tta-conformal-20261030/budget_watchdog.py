"""Local release backstop; only this named candidate session, no new OAuth."""
import json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
start=json.loads((r/'cli-ledger/allocate-t4.json').read_text())['started_epoch']
while not (r/'released.json').exists():
    if time.time()-start>=115*60:
        subprocess.run(['/Users/lkc/.local/share/google-colab-cli/bin/python',str(r/'cli_record.py'),
                        'watchdog-stop','stop','--session','tta-conformal-20261030'],timeout=190)
        break
    time.sleep(15)
print('Local GPU budget watchdog finished',flush=True)
