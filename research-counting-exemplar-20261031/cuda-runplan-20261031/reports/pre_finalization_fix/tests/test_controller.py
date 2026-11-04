import base64,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

class ControllerTests(unittest.TestCase):
    def test_no_GO_rejected_before_any_service_call(self):
        import controller
        with patch.object(controller,'verify_freeze',return_value=({},'abc')),patch.object(controller,'bounded_stream',side_effect=AssertionError('must not contact CLI')):
            with self.assertRaises(ValueError):controller.main(['--run'])

    def test_fake_failed_precheck_persists_then_releases_without_second_allocation(self):
        import controller
        from contracts import make_packet
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);(base/'payload/scripts').mkdir(parents=True)
            (base/'payload/scripts/launch_template.py').write_text('__CUE_LAUNCH_CONFIG__')
            auth=base/'go.json';auth.write_text(json.dumps({'allocation_allowed':True,'freeze_sha256':'abc','scope':'one_T4_precheck_then_conditional_492','max_allocations':1}))
            frozen={'bundle':{'bytes':1,'parts':[],'sha256':'b'},'payload_bytes':1,'package_manifest_sha256':'c','prior_compute_wall_seconds':54.91578541700437}
            calls=[];active=False
            def fake(command,on_line,should_stop,**kwargs):
                nonlocal active
                args=command[2:];calls.append(args);op=args[0]
                if op=='--snapshot':on_line(json.dumps({'balance':40.,'rate_per_hour':1.15 if active else 0.,'active_assignments':int(active),'assignments':[{'name':'counting-fixed-abc','endpoint':'owned','accelerator':'T4'}] if active else []})+'\n')
                elif op=='new':
                    active=True;Path(kwargs['env']['CUE_ALLOCATION_RECEIPT']).write_text(json.dumps({'endpoint':'owned','session_name':'counting-fixed-abc','accelerator':'T4'}))
                elif op=='--release-receipt':active=False
                elif op=='exec' and args[args.index('--file')+1].endswith('remote_launch.py'):
                    data={'supervisor.json':{'status':'BLOCKED_OR_INCOMPLETE','compute_used_seconds':100.},'completion.json':{'complete':False}}
                    index={}
                    for path,value in data.items():
                        raw=json.dumps(value).encode();index[path]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};on_line(make_packet(path,raw)+'\n')
                    on_line(make_packet('artifact_index.json',json.dumps({'files':index}).encode())+'\n')
                return {'returncode':0,'terminated_by_watchdog':False,'wall_seconds':0.,'pid':1}
            with patch.object(controller,'BASE',base),patch.object(controller,'verify_freeze',return_value=(frozen,'abc')),patch.object(controller,'owned_size',return_value=0),patch.object(controller,'bounded_stream',side_effect=fake),patch.object(controller.subprocess,'Popen') as popen:
                popen.return_value.pid=999999
                self.assertEqual(controller.main(['--run','--authorization',str(auth)]),2)
                report=json.loads((base/'run/controller_report.json').read_text())
                self.assertEqual(report['release'],'CONFIRMED');self.assertTrue(report['all_indexed_artifacts_persisted_before_release']);self.assertFalse(active)
                self.assertEqual(sum(args[0]=='new' for args in calls),1)
                self.assertTrue((base/'run/artifacts/supervisor.json').exists())
                self.assertEqual(controller.main(['--run','--authorization',str(auth)]),2)
                self.assertEqual(sum(args[0]=='new' for args in calls),1)

if __name__=='__main__':unittest.main()
