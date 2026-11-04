import json
import numpy as np
from stheta.protocol import WORK,save_json,file_sha

if __name__=='__main__':
    lock=json.loads((WORK/'reports/encoder-comparison-lock.json').read_text())
    old=np.load(WORK/'reports/encoder-reference446.npz',allow_pickle=False)
    new=np.load(WORK/'reports/encoder-current457.npz',allow_pickle=False)
    results={};passed=True
    for k in old.files:
        a,b=old[k],new[k];shape=a.shape==b.shape;finite=np.isfinite(a).all() and np.isfinite(b).all()
        diff=np.abs(a.astype(np.float64)-b.astype(np.float64));maxabs=float(diff.max());meanabs=float(diff.mean())
        gate=shape and finite
        if k in lock['tolerances']:
            tolerance=lock['tolerances'][k];gate=gate and maxabs<=tolerance['maxabs'] and meanabs<=tolerance['meanabs']
        results[k]={'shape_identical':shape,'finite':bool(finite),'maxabs':maxabs,'meanabs':meanabs,'passed':bool(gate)};passed=passed and gate
    report={'status':'passed' if passed else 'failed','fixed_before_comparison':lock,'checks':results,
      'reference':json.loads((WORK/'reports/encoder-reference446.json').read_text()),'actual':json.loads((WORK/'reports/encoder-current457.json').read_text()),
      'scope':'Same official weights; installed official 4.46.3 and 4.57.1 CPU FP32 eager; three fixed224 inputs; no assertion about other dtypes/backends/modes',
      'formula_only_check_is_not_substitute':True,'no_global_environment_change':True}
    save_json(WORK/'reports/encoder-cross-version.json',report);print(json.dumps({'status':report['status'],'checks':results},indent=2))
    if not passed:raise SystemExit(1)
