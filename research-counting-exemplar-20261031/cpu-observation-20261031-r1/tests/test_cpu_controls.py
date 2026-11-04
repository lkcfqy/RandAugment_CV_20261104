import contextlib,hashlib,io,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import launch_cpu
from contracts import make_packet,compute_gate

class CPUControlTests(unittest.TestCase):
    def test_original_twice_492_gate_now_uses_final_7200_cap(self):
        self.assertTrue(compute_gate(600,30,[6.3,6.2,6.1],60)['allowed'])
        self.assertFalse(compute_gate(600,30,[7.,6.2,6.1],60)['allowed'])
        self.assertEqual(compute_gate(600,30,[6.3,6.2,6.1],60)['multiplier'],2)

    def test_separate_formal_go_required_and_no_budget_reset(self):
        with self.assertRaises(ValueError):launch_cpu.validate_formal_go(None,'expected')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'go.json'
            for value in ({'formal_allowed':False},{'formal_allowed':True,'freeze_sha256':'wrong','scope':'Mac_CPU_fixed_492','max_runs':1}):
                p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):launch_cpu.validate_formal_go(p,'expected')
        for now,phase,global_at,rss,disk in [(180,180,7000,0,0),(7140,8000,7200,0,0),(1,180,7200,8*1024**3+1,0),(1,180,7200,0,6*1024**3+1)]:
            self.assertIsNotNone(launch_cpu.stop_reason(now,phase,global_at,rss,disk))

    def test_artificial_precheck_records_carry_and_refuses_second_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'plan.json').write_text(json.dumps({'prior_compute_seconds':545.6848070110136}))
            data={'status':'PASS_CPU_REPEATABILITY','freeze_sha256':'fixed','counters':{'root_forward_entered':6,'root_forward_returned':6,'precheck_attempts':6,'formal_attempts':0}}
            raw=json.dumps(data).encode();idx={'files':{'engineering/status.json':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}}
            def fake(command,on_line,should_stop,**kwargs):
                on_line(make_packet('engineering/status.json',raw)+'\n');on_line(make_packet('artifact_index.json',json.dumps(idx).encode())+'\n')
                return {'returncode':0,'terminated_by_watchdog':False,'wall_seconds':0.,'pid':999}
            with patch.object(launch_cpu,'ROOT',root),patch.object(launch_cpu,'START',time.monotonic()),patch.object(launch_cpu,'verify_freeze',return_value=({},'fixed')),patch.object(launch_cpu,'owned_size',return_value=0),patch.object(launch_cpu,'bounded_stream',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch_cpu.main(['--precheck']),0)
                row=json.loads((root/'precheck/controller_report.json').read_text())
                self.assertGreaterEqual(row['cumulative_compute_seconds'],545.6848070110136)
                self.assertEqual(row['status'],'PASS_CPU_REPEATABILITY')
                with self.assertRaises(FileExistsError):launch_cpu.main(['--precheck'])

if __name__=='__main__':unittest.main()
