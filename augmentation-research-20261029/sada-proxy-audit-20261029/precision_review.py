#!/usr/bin/env python3
"""Extra precision validation of the unchanged frozen case grid; stdlib only."""
import json
import time
from decimal import Decimal as D, localcontext
import audit

start=time.process_time()
protocol=json.loads((audit.ROOT/'protocol.json').read_text())
reference=[json.loads(s) for s in (audit.ROOT/'results/windows.jsonl').read_text().splitlines()]
with localcontext() as ctx:
    ctx.prec=60
    lower=[w for spec in audit.make_specs(protocol) for w in audit.windows_for(audit.trace(spec),protocol)]
with localcontext() as ctx:
    ctx.prec=90
    maximum=D(0)
    mismatches=[]
    nonzero_low_when_high_zero=[]
    for high,low in zip(reference,lower):
        assert (high['case_id'],high['window_length'],high['end_step'],high['common_eta_divisor']) == (low['case_id'],low['window_length'],low['end_step'],low['common_eta_divisor'])
        for m in audit.METRICS:
            if high['relations'][m]!=low['relations'][m]:
                mismatches.append({'case_id':high['case_id'],'L':high['window_length'],'end_step':high['end_step'],
                                   'eta_divisor':high['common_eta_divisor'],'metric':m,'rank90':high['relations'][m],'rank60':low['relations'][m]})
            for i,(a,b) in enumerate(zip(high['ssd'][m],low['ssd'][m])):
                a=D(a)
                if a:maximum=max(maximum,abs(b-a)/abs(a))
                elif b:nonzero_low_when_high_zero.append({'case_id':high['case_id'],'metric':m,'sample':i,'value60':b})
    result={'scope':'Same frozen 66-case grid, full window calculations at 60 digits compared with saved 90-digit results.',
            'window_count':len(reference),'max_relative_ssd_error':maximum,
            'rank_mismatches':mismatches,'nonzero_low_when_high_zero':nonzero_low_when_high_zero,
            'checks':{'all_window_ranks_agree':not mismatches,'zero_variances_agree':not nonzero_low_when_high_zero,
                      'all_nonzero_ssd_relative_error_below_1e_minus_20':maximum<D('1e-20')},
            'cpu_seconds':time.process_time()-start}
audit.dump(audit.ROOT/'precision-review.json',result)
print(json.dumps(audit.decjson(result),indent=2))
raise SystemExit(0 if all(result['checks'].values()) else 1)
