"""Call installed official CLI while forbidding new OAuth. No token inspection."""
import sys
import colab_cli.auth as auth
from colab_cli.cli import app
def forbid(*args, **kwargs):
    raise RuntimeError('BLOCKED: existing authorization unusable; new OAuth prohibited')
auth._run_remote_flow = forbid
if __name__ == '__main__':
    app(args=sys.argv[1:], prog_name='colab', standalone_mode=True)
