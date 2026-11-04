import json,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parent
control=json.loads((root/'control.json').read_text())
process=subprocess.Popen(['/usr/bin/caffeinate','-i'])
(root/'local_awake_guard.json').write_text(json.dumps({'pid':process.pid,'release_marker_scoped':True,'deadline_epoch':control['deadline_epoch']}))
try:
 while not (root/'RELEASE_CONFIRMED.json').exists() and time.time()<control['deadline_epoch']:time.sleep(1)
finally:
 process.terminate();process.wait(timeout=5)
