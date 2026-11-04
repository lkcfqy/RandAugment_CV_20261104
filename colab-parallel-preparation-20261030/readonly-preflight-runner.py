"""Run installed official CLI read commands; forbid its interactive OAuth fallback.

No token contents are inspected or emitted. No runtime allocation command is run.
The guard only narrows behavior in this process; installed files are unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import contextlib
import io
import json
import re
import signal
import sys

import colab_cli.auth as auth
from colab_cli.cli import app
from google.oauth2.credentials import Credentials

CLI = '/Users/lkc/.local/bin/colab'
ROOT = Path('/Users/lkc/Documents/Codex/2026-10-30/task-2/colab-readonly-preflight-20261030')
ROOT.mkdir(exist_ok=False)


class ReauthorizationBlocked(RuntimeError):
    pass


def forbid_new_oauth(*args, **kwargs):
    raise ReauthorizationBlocked('Existing authorization unusable; new OAuth was not started')


auth._run_remote_flow = forbid_new_oauth
refresh_calls = 0
original_refresh = Credentials.refresh


def existing_refresh(self, *args, **kwargs):
    global refresh_calls
    refresh_calls += 1
    return original_refresh(self, *args, **kwargs)


Credentials.refresh = existing_refresh


def deadline(signum, frame):
    raise TimeoutError('Read-only command exceeded 45 seconds')


signal.signal(signal.SIGALRM, deadline)


def sanitized(text):
    # Defense in depth; successful commands below emit only help/session/usage.
    text = re.sub(r'ya29\.[A-Za-z0-9_.-]+', '[REDACTED]', text)
    text = re.sub(r'(?i)(access_token|refresh_token|id_token|client_secret|authorization_code)([\s\"\x27:=]+)[^\s,\"\x27]+',
                  r'\1\2[REDACTED]', text)
    text = re.sub(r'https://accounts\.google\.com/[^\s]+', '[OAUTH URL OMITTED]', text)
    return text


def run(args, name):
    stdout, stderr = io.StringIO(), io.StringIO()
    started = datetime.now(timezone.utc).isoformat()
    error_type = None
    exit_code = 0
    signal.alarm(45)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return_value = app(args=args, prog_name=CLI, standalone_mode=False)
            if isinstance(return_value, int):
                exit_code = return_value
    except ReauthorizationBlocked:
        exit_code, error_type = 1, 'ReauthorizationBlocked'
    except TimeoutError:
        exit_code, error_type = 124, 'ReadOnlyTimeout'
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
        error_type = None if exit_code == 0 else 'SystemExit'
    except Exception as error:
        exit_code = 1
        error_type = type(error).__name__
        # Only retain status codes / named grant failures from error text.
        safe = re.findall(r'\b(?:401|403|404|429|500|502|503|invalid_grant|invalid_client|invalid_scope|access_denied)\b', str(error))
        if safe:
            stderr.write('Safe error indicators: ' + ', '.join(sorted(set(safe))))
    finally:
        signal.alarm(0)
    output = sanitized(stdout.getvalue())
    err = sanitized(stderr.getvalue())
    if exit_code != 0:
        # Do not archive arbitrary rich tracebacks or request response bodies.
        output = ''
        indicators = re.findall(r'\b(?:401|403|404|429|500|502|503|invalid_grant|invalid_client|invalid_scope|access_denied)\b', err)
        err = ', '.join(sorted(set(indicators)))
    record = {'command': [CLI, *args], 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'exit_code': exit_code, 'error_type': error_type,
              'stdout': output, 'safe_error_indicators': err if exit_code else '',
              'interactive_oauth_enabled': False}
    (ROOT / (name + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    if exit_code == 0:
        (ROOT / (name + '.txt')).write_text(output)
    print(json.dumps({'step': name, 'exit_code': exit_code, 'error_type': error_type,
                      'output': output if name in ('version', 'sessions', 'usage') else 'help saved',
                      'safe_error_indicators': record['safe_error_indicators']}, ensure_ascii=False), flush=True)
    return record


if version('google-colab-cli') != '0.7.5':
    raise SystemExit('Installed CLI version changed; inspect before proceeding')

records = [run(['version'], 'version')]
sessions = run(['sessions'], 'sessions')
records.append(sessions)
if sessions['exit_code'] == 0:
    records.append(run(['usage'], 'usage'))
for command in ('run', 'new', 'stop', 'download', 'status'):
    records.append(run([command, '--help'], command + '-help'))

summary = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'cli_version': version('google-colab-cli'),
           'cli_entrypoint': CLI, 'implementation': 'Official installed Typer CLI app, with process-local interactive OAuth fallback disabled',
           'existing_authorization_refresh_calls': refresh_calls, 'new_oauth_started': False,
           'token_contents_printed_or_archived': False, 'runtime_allocation_requested': False,
           'training_or_remote_code_execution': False, 'Mac_GPU_used': False,
           'commands': records}
(ROOT / 'preflight.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'completed': True, 'output': str(ROOT), 'existing_authorization_refresh_calls': refresh_calls,
                  'new_oauth_started': False, 'runtime_allocation_requested': False}, ensure_ascii=False), flush=True)
