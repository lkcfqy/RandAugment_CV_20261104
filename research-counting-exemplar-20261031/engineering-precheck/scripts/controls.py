"""Small fail-closed controls; no ML packages or model imports."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import tempfile


class GuardViolation(RuntimeError):
    pass


def save_engineering_json(path, data):
    forbidden = {'pred_logits', 'pred_boxes', 'count', 'scores', 'boxes'}
    def check(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise GuardViolation('prediction payload cannot be persisted')
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
    check(data)
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', prefix=path.name+'.', suffix='.tmp', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def claim_launch(path, data):
    try:
        with Path(path).open('x') as f:
            json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError as exc:
        raise GuardViolation('this precheck already launched; no automatic retry') from exc


class AttemptLedger:
    def __init__(self, path, maximum):
        self.path = Path(path)
        self.maximum = maximum
        if self.path.exists():
            raise GuardViolation('attempt ledger already exists; no automatic retry')
        self.rows = []

    def reserve(self, label):
        if self.rows and self.rows[-1]['status'] == 'started':
            raise GuardViolation('previous attempt still unfinished')
        if len(self.rows) >= self.maximum:
            raise GuardViolation('forward attempt budget exhausted')
        self.rows.append({'index': len(self.rows)+1, 'label': label, 'status': 'started', 'started_monotonic': time.monotonic()})
        save_engineering_json(self.path, self.rows)

    def finish(self, success, detail=None):
        if not self.rows or self.rows[-1]['status'] != 'started':
            raise GuardViolation('no unfinished attempt')
        row = self.rows[-1]
        row['status'] = 'completed' if success else 'failed'
        row['elapsed_seconds'] = time.monotonic()-row['started_monotonic']
        if detail is not None:
            row['detail'] = detail
        save_engineering_json(self.path, self.rows)


def supervise(command, directory, wall_seconds, rss_limit, mps_limit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reason = None
    peak_rss = 0
    peak_mps = 0
    with (directory/'worker.log').open('w') as log:
        child = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                elapsed = time.monotonic()-started
                # A short subprocess timeout cannot hold up the outer wall watchdog.
                try:
                    p = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)], capture_output=True, text=True, timeout=.05)
                    rss = int(p.stdout.strip() or 0)*1024
                    peak_rss = max(peak_rss, rss)
                except (subprocess.TimeoutExpired, ValueError):
                    pass
                memory_path = directory/'memory.json'
                if memory_path.exists():
                    memory = json.loads(memory_path.read_text())
                    peak_mps = max(peak_mps, memory.get('mps_driver_bytes', 0), memory.get('peak_mps_driver_bytes', 0), memory.get('mps_current_bytes', 0))
                    peak_rss = max(peak_rss, memory.get('peak_rss_bytes', 0))
                if elapsed >= wall_seconds:
                    reason = 'wall_limit'
                elif peak_rss > rss_limit:
                    reason = 'rss_limit'
                elif peak_mps > mps_limit:
                    reason = 'mps_limit'
                if reason:
                    break
                time.sleep(min(.05, max(0, wall_seconds-(time.monotonic()-started))))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    result = {'pid': child.pid, 'returncode': child.returncode, 'elapsed_seconds': time.monotonic()-started, 'termination_reason': reason, 'peak_rss_bytes_sampled': peak_rss, 'peak_mps_driver_bytes_sampled': peak_mps, 'wall_limit_seconds': wall_seconds, 'rss_limit_bytes': rss_limit, 'mps_limit_bytes': mps_limit, 'retries': 0}
    save_engineering_json(directory/'supervisor.json', result)
    return result
