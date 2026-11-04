import json,os,sys,tempfile,time,unittest
from pathlib import Path
from contracts import make_packet

class TransportTests(unittest.TestCase):
    def test_streamed_result_is_durable_and_conflict_rejected(self):
        from transport import Collector
        with tempfile.TemporaryDirectory() as tmp:
            c=Collector(Path(tmp));c.line(make_packet('results/0000.json',b'{"count":1}'))
            self.assertEqual((Path(tmp)/'artifacts/results/0000.json').read_bytes(),b'{"count":1}')
            c.line(make_packet('results/0000.json',b'{"count":1}'))
            with self.assertRaises(ValueError):c.line(make_packet('results/0000.json',b'{"count":2}'))
            c.line(make_packet('engineering/status.json',b'1'));c.line(make_packet('engineering/status.json',b'2'))
            self.assertEqual((Path(tmp)/'artifacts/engineering/status.json').read_bytes(),b'2')

    def test_subprocess_watchdog_preserves_output_before_timeout(self):
        from transport import bounded_stream
        seen=[];start=time.monotonic()
        result=bounded_stream([sys.executable,'-u','-c','import time;print("READY",flush=True);time.sleep(10)'],seen.append,lambda:time.monotonic()-start>.3,poll_seconds=.02)
        self.assertTrue(result['terminated_by_watchdog']);self.assertIn('READY\n',seen)
        self.assertLess(time.monotonic()-start,3)

    def test_collector_failure_terminates_child(self):
        from transport import bounded_stream
        def reject(line):raise ValueError('bad evidence')
        with self.assertRaises(ValueError):bounded_stream([sys.executable,'-u','-c','import time;print("BAD",flush=True);time.sleep(10)'],reject,lambda:False,poll_seconds=.02)

    def test_index_requires_every_file_and_exact_hash(self):
        from transport import Collector
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            c=Collector(Path(tmp));data=b'{}';c.line(make_packet('results/0000.json',data))
            idx={'files':{'results/0000.json':{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}}
            self.assertEqual(c.missing(idx),[])
            idx['files']['results/0001.json']=idx['files']['results/0000.json']
            self.assertEqual(c.missing(idx),['results/0001.json'])

if __name__=='__main__':unittest.main()
