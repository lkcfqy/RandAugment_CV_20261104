from ctrla_adapter import *
from datetime import datetime,timezone
assert not (ROOT/'seed0-training-summary.json').exists()
write_json(ROOT/'compatibility-amendment-layout.json',{
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'reason':'First full synthetic benchmark: eval agrees CPU/MPS, backward fails with view stride incompatibility in source channels_last layout.',
 'change':'Convert complete official network to contiguous NCHW memory format before transfer to MPS. No architectural, parameter, objective, optimizer, crop or operation changes.',
 'validation_required':'All existing fixtures + CPU/MPS eval + train-mode gradient relative L2 <=1% +15 synthetic MPS train batches.',
 'protocol_original_preserved':True,'original_protocol_sha256':sha256(ROOT/'protocol.json'),
 'pre_real_training':True,'real_images_used':0,'outcome_metrics_seen':False,
 'first_benchmark_counter_correction':'The original finally block assigned planned1875 synthetic images despite failure on first step. Actual attempted first-step125 synthetic model forwards, 8 synthetic CPU/MPS eval forwards; no real training exposure. Preserve first receipt and this correction.',
 'adapter_sha256':sha256(ROOT/'ctrla_adapter.py')})
