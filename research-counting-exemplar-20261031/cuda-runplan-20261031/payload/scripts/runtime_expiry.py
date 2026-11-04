"""Optional official Colab in-VM unassign fallback; bounded and no credentials."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args();status={'status':'CHECKING_NATIVE_API','credentials_uploaded':False}
    try:
        spec=importlib.util.find_spec('google.colab.runtime')
        if spec is None or not os.environ.get('TBE_RUNTIME_ADDR'):raise RuntimeError('native runtime management service unavailable')
        status.update(status='ARMED',runtime_source_sha256=hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),deadline_monotonic=args.deadline)
        args.state.write_text(json.dumps(status))
        while time.monotonic()<args.deadline-12:time.sleep(min(1,max(0,args.deadline-12-time.monotonic())))
        # Official API POSTs the internal management endpoint, then disconnects UI.
        result=subprocess.run([sys.executable,'-c','from google.colab import runtime; runtime.unassign()'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(.1,min(10,args.deadline-time.monotonic())))
        status.update(status='UNASSIGN_REQUEST_RETURNED',returncode=result.returncode)
    except BaseException as error:status.update(status='UNASSIGN_UNCONFIRMED',error_type=type(error).__name__)
    args.state.write_text(json.dumps(status));return 0

if __name__=='__main__':raise SystemExit(main())
