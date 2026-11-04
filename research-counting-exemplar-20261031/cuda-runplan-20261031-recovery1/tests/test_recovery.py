import json,tempfile,unittest
from pathlib import Path

class RecoveryTests(unittest.TestCase):
    def test_stop_failure_is_followed_by_snapshot_known_endpoint_retry_and_final_snapshot(self):
        from release_guard import release_owned
        with tempfile.TemporaryDirectory() as temp:
            receipt=Path(temp)/'receipt';receipt.write_text(json.dumps({'endpoint':'owned'}))
            calls=[];states=[{'assignments':[{'name':'mine','endpoint':'owned'}]},{'assignments':[]}]
            def command(args,timeout):
                calls.append(('command',args[0]))
                if len(calls)==1:raise TimeoutError('stop outcome uncertain')
            def snapshot():calls.append(('snapshot',));return states.pop(0)
            result=release_owned(command,snapshot,receipt,'mine')
            self.assertEqual(result['release'],'CONFIRMED')
            self.assertEqual([x[0] for x in calls],['command','snapshot','command','snapshot'])
    def test_failed_reconciliation_does_not_blindly_retry(self):
        from release_guard import release_owned
        with tempfile.TemporaryDirectory() as temp:
            calls=[]
            def command(args,timeout):calls.append(args);raise TimeoutError('offline')
            def snapshot():raise TimeoutError('offline')
            self.assertEqual(release_owned(command,snapshot,Path(temp)/'absent','mine')['release'],'UNCERTAIN')
            self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
