"""Independent autograd reference, synthetic tensors, actual DeiT on MPS."""
from core import *
started=time.perf_counter();m=model();m.train();params=list(m.named_parameters());values=[p for _,p in params]
with cpu_rng(key('oracle',0)):
    w=torch.randn(2,3,224,224).to(DEVICE);s=torch.randn(2,3,224,224).to(DEVICE)
y=torch.tensor([0,99],device=DEVICE)
with torch.no_grad():actual_output_shape=list(m(w).shape)
assert actual_output_shape==[2,100]
references=[]
for x in [w,s]:
    g=torch.autograd.grad(F.cross_entropy(m(x),y),values)
    references.append([v.detach().cpu().double().numpy() for v in g]);del g
cases=[('weak','weak',0),('dual','dual',0),('ent_single','ent',0),('head_refit','head',0),('random_weak','random',0),('random_strong','random',1),('candidate','candidate',0)]
results=[]
for name,kind,route in cases:
    loss,_=losses(m,w,s,y,kind,route)
    actual=torch.autograd.grad(loss,values,allow_unused=True)
    sums={'backbone':[0.,0.,0.],'head':[0.,0.,0.]}
    for i,((pname,p),g) in enumerate(zip(params,actual)):
        part='head' if pname.startswith('head.') else 'backbone';gw,gs=references[0][i],references[1][i]
        if kind in ['weak','ent']:expected=gw
        elif part=='head' or kind=='dual':expected=(gw+gs)/2
        elif kind=='head':expected=np.zeros_like(gw)
        elif kind=='candidate' or route==0:expected=gw
        else:expected=gs
        got=np.zeros_like(expected) if g is None else g.detach().cpu().double().numpy()
        diff=got-expected;stat=sums[part]
        stat[0]=max(stat[0],float(np.abs(diff).max()));stat[1]+=float(np.sum(diff**2));stat[2]+=float(np.sum(expected**2))
    row={'case':name,'parts':{}}
    for part,(mx,num,den) in sums.items():
        rel=math.sqrt(num/max(den,1e-30));row['parts'][part]={'max_abs':mx,'relative_l2':rel,'passed':mx<=5e-5 and rel<=2e-4}
    results.append(row);print(json.dumps(row),flush=True)
    del actual,loss
passed=all(v['passed'] for r in results for v in r['parts'].values())
result={'status':'passed' if passed else 'failed','device':'mps','dtype':'float32','actual_model':'deit_tiny_patch16_224.fb_in1k',
        'oracle':'separate native autograd.grad(meanCE(model(w))) and strong; then independent weighted sums, compared to production paths',
        'tolerances':{'max_abs':5e-5,'relative_l2':2e-4},'cases':results,'seconds':time.perf_counter()-started,'head_classes':100,'output_shape_checked':actual_output_shape,'real_dataset_images_used':0,
        'scientific_accuracy_measured':False,'core_sha256':sha(ROOT/'core.py')}
dump(ROOT/'gradient-oracle.json',result)
assert passed,'VALIDATION: gradient oracle failed'
