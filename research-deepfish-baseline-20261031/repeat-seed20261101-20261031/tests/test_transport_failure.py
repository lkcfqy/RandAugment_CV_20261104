"""A reported CLI failure must terminate its live child, not burn the GPU budget."""
import importlib.util, os, pathlib, sys, time, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transport_under_test',ROOT/'gpu-run/local_transport.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
class FailureTermination(unittest.TestCase):
 def test_failure_callback_terminates_live_cli_process_immediately(self):
  owned=[];started=time.monotonic()
  def on_line(line):
   if 'CLI_FAILED' in line:raise RuntimeError('reported CLI failure')
  with self.assertRaisesRegex(RuntimeError,'reported CLI failure'):
   transport.bounded_stream([sys.executable,'-c','import time;print("CLI_FAILED",flush=True);time.sleep(30)'],on_line,lambda:False,on_started=owned.append)
  self.assertLess(time.monotonic()-started,3)
  self.assertEqual(len(owned),1)
  with self.assertRaises(ProcessLookupError):os.kill(owned[0],0)
if __name__=='__main__':unittest.main()
