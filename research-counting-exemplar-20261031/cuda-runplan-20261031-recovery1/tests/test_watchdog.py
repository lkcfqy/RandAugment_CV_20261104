import contextlib,io,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch

class WatchdogTests(unittest.TestCase):
    def test_deadlines_include_failed_precheck_cleanup_and_finalization_reserve(self):
        from remote_runner import termination_reason
        common={'worker_elapsed':1,'saw_formal':False,'compute_used':100,'tree_rss':0,'cuda_peak':0,'allocation_left':1000}
        self.assertIsNone(termination_reason(**common))
        for changed in ({'worker_elapsed':360},{'compute_used':1740},{'tree_rss':8*1024**3+1},{'cuda_peak':8*1024**3+1},{'allocation_left':60}):
            self.assertIsNotNone(termination_reason(**dict(common,**changed)))
        self.assertIsNone(termination_reason(**dict(common,worker_elapsed=361,saw_formal=True)))
        self.assertIsNone(termination_reason(**dict(common,worker_elapsed=359.9)))
        self.assertIsNone(termination_reason(**dict(common,worker_elapsed=400,precheck_stage='cuda',stage_elapsed=89.9)))
        self.assertIsNotNone(termination_reason(**dict(common,worker_elapsed=400,precheck_stage='cuda',stage_elapsed=90)))
        self.assertIsNotNone(termination_reason(**dict(common,worker_elapsed=400,precheck_stage='cuda',stage_elapsed=1,compute_used=1740)))

    def test_preparation_deadline_and_disk_are_enforced_before_model_work(self):
        from remote_runner import preparation_termination_reason
        values={'compute_used':100,'allocation_left':1000,'disk_bytes':1024}
        self.assertIsNone(preparation_termination_reason(**values))
        self.assertIsNotNone(preparation_termination_reason(**dict(values,compute_used=1740)))
        self.assertIsNotNone(preparation_termination_reason(**dict(values,allocation_left=90)))
        self.assertIsNotNone(preparation_termination_reason(**dict(values,disk_bytes=6*1024**3+1)))
        self.assertIsNone(preparation_termination_reason(**dict(values,preparation_compute_seconds=59.9)))
        self.assertIsNotNone(preparation_termination_reason(**dict(values,preparation_compute_seconds=60)))

    def test_external_parent_kills_stalled_preparation_and_never_starts_worker(self):
        import remote_runner
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'payload';(root/'scripts').mkdir(parents=True)
            (root/'plan.json').write_text(json.dumps({'previous_compute_wall_seconds':1739.6}))
            (root/'scripts/remote_prepare.py').write_text('import os,time\nfrom pathlib import Path\nPath("fake_pid").write_text(str(os.getpid()))\nprint("PREPARATION_STARTED",flush=True)\ntime.sleep(10)\n')
            begin=time.monotonic()
            with patch.object(remote_runner,'ROOT',root),patch.object(remote_runner,'SCRIPT_START',begin),patch.object(sys,'argv',['runner','--allocation-seconds-left','1000','--assembly-compute-seconds','0']),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(remote_runner.main(),2)
            report=json.loads((root/'output/supervisor.json').read_text())
            self.assertTrue(report['preparation_process']['terminated_by_watchdog']);self.assertFalse(report['worker_started'])
            self.assertLess(time.monotonic()-begin,3)
            pid=int((root/'fake_pid').read_text())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_external_parent_honors_cuda_stage_origin_before_any_formal_work(self):
        import remote_runner
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'payload';(root/'scripts').mkdir(parents=True)
            (root/'plan.json').write_text(json.dumps({'previous_compute_wall_seconds':259.6864732820104}))
            (root/'scripts/remote_prepare.py').write_text('import json\nfrom pathlib import Path\np=Path("output/engineering/preparation.json");p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({"status":"READY","installation_wall_seconds":0,"download_wall_seconds":0}))\n')
            (root/'scripts/worker.py').write_text('import json,os,time\nfrom pathlib import Path\nPath("fake_pid").write_text(str(os.getpid()))\nPath("worker_state.json").write_text(json.dumps({"phase":"precheck","precheck_stage":"cuda","precheck_stage_started_monotonic":time.monotonic()-90}))\nprint("CUDA_STAGE_STARTED",flush=True)\ntime.sleep(10)\nPath("formal_ran").touch()\n')
            begin=time.monotonic()
            with patch.object(remote_runner,'ROOT',root),patch.object(remote_runner,'SCRIPT_START',begin),patch.object(sys,'argv',['runner','--allocation-seconds-left','1000','--assembly-compute-seconds','0']),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(remote_runner.main(),2)
            report=json.loads((root/'output/supervisor.json').read_text())
            self.assertTrue(report['worker_started']);self.assertTrue(report['terminated_by_watchdog'])
            self.assertIn('90 s external CUDA',report['stop_reason'])
            self.assertFalse((root/'formal_ran').exists());self.assertLess(time.monotonic()-begin,3)
            pid=int((root/'fake_pid').read_text())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

if __name__=='__main__':unittest.main()
