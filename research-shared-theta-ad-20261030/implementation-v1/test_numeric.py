import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import unittest,json,time,copy
import numpy as np
import torch
import torch.nn.functional as F
from stheta.encoder import transform,linearize,srgb,color_stats,color_target,color_normalize,EPS
from stheta.methods import (distance_cache,fit_projections,cache_scores,permutations,permute_cache,shared_scores,original_winners,fit_bow,bow_histograms,bow_costs,all_readouts,baseline_names,upper_mean,post_projection_scores)
from stheta.metrics import auroc,threshold,positive,summarize
from stheta.data import Components,is_near
from stheta.protocol import THETAS,THETA_IDS,WORK,save_json,METHOD_NAMES,BASELINE_NAMES,query_manifest_records,digest

class Numerics(unittest.TestCase):
    def setUp(self):torch.manual_seed(17);torch.set_num_threads(4);self.rng=np.random.default_rng(17)
    def test_theta_identity_and_photometry(self):
        x=self.rng.random((16,16,3),dtype=np.float32)
        self.assertIs(transform(x,THETAS[0]),x)
        self.assertEqual(THETAS[0],(1.,1.));self.assertEqual(THETAS[1:],sorted(THETAS[1:]))
        np.testing.assert_allclose(srgb(linearize(x)),x,atol=3e-7)
        L=linearize(x);mu=L.mean((0,1),keepdims=True)
        np.testing.assert_allclose(linearize(transform(x,(.8,1.2))),np.clip(.8*(1.2*L-.2*mu),0,1),atol=3e-7)
        for theta in THETAS:
            y=transform(x,theta);self.assertEqual(y.dtype,np.float32);self.assertTrue(((y>=0)&(y<=1.000001)).all())
    def test_color_floor_and_target(self):
        x=np.full((8,8,3),.5,np.float32);m,s,floor=color_stats(x)
        self.assertTrue(floor.all());np.testing.assert_allclose(s,1e-4)
        y,n=color_normalize(x,color_target([x,x]));np.testing.assert_allclose(y,x,atol=2e-7);self.assertEqual(n,3)
    def test_exact_cache_bruteforce_and_ties(self):
        q=F.normalize(torch.randn(12,24),dim=-1);b=F.normalize(torch.randn(4,9,16,24),dim=-1)
        A,ix,_=distance_cache(q,b,group_chunk=3)
        direct=(q[:,None,None,None,:]-b[None,:,:,:,:]).square().sum(-1)
        np.testing.assert_allclose(A,direct.min(-1).values.numpy(),atol=6e-7)
        np.testing.assert_array_equal(ix,direct.argmin(-1).numpy())
        zero=torch.zeros_like(b);Az,iz,_=distance_cache(torch.zeros_like(q),zero,group_chunk=3)
        self.assertTrue((iz==0).all());w=original_winners(Az,iz)
        for mode in w:
            for array in w[mode]:self.assertTrue((array==0).all())
    def test_permutations_union_identity_candidate_pooled(self):
        A=self.rng.random((256,10,9),dtype=np.float32);ids=[str(i) for i in range(10)]
        pi=permutations(ids,9001);B=permute_cache(A,pi)
        np.testing.assert_array_equal(np.sort(A,2),np.sort(B,2));np.testing.assert_array_equal(A[:,:,0],B[:,:,0])
        np.testing.assert_array_equal(A.min((1,2)),B.min((1,2)))
        self.assertEqual(pi.tolist(),permutations(ids,9001).tolist())
        self.assertFalse(np.array_equal(pi,permutations(ids,9002)))
        p,_=shared_scores(A);self.assertGreaterEqual(p+1e-7,float(upper_mean(A.min((1,2)))))
        min_top=float(upper_mean(A.min(1),13,axis=0).min())
        self.assertLessEqual(float(upper_mean(A.min((1,2)))),min_top+1e-7)
        self.assertLessEqual(min_top,p+1e-7)
        # Force all nonidentity scores above identity, for all permutations.
        A[:,:,0]=0;A[:,:,1:]+=1;B=permute_cache(A,pi)
        self.assertEqual(shared_scores(A),(0.,0));self.assertEqual(shared_scores(B),(0.,0))
    def test_projected_pre_full_search_and_post(self):
        q=F.normalize(torch.randn(16,48),dim=-1);b=F.normalize(torch.randn(3,9,16,48),dim=-1)
        u,_=torch.linalg.qr(torch.randn(48,32));bases={'S0':u.numpy()}
        A,ix,pre=distance_cache(q,b,bases,group_chunk=2)
        for rank in (8,16,32):
            uk=u[:,:rank]
            pq=F.normalize(q-(q@uk)@uk.T,dim=-1,eps=EPS)
            pb=F.normalize(b-(b@uk)@uk.T,dim=-1,eps=EPS)
            for mode in ('identity','pool9'):
                selected=pb[:,0:1] if mode=='identity' else pb
                flat=selected.permute(1,0,2,3).reshape(-1,48)
                direct=(pq[:,None,:]-flat[None,:,:]).square().sum(-1)
                got=pre[f'S0/r{rank}'][mode]
                np.testing.assert_allclose(got['distances'],direct.min(-1).values.numpy(),atol=2e-6)
                np.testing.assert_array_equal(got['indices'],direct.argmin(-1).numpy())
        post=post_projection_scores(q,b,A,ix,bases,top=3)
        winners=original_winners(A,ix)
        for mode,(t,r,j) in winners.items():
            v=q-b[r,t,j.astype(int)];u8=u[:,:8];pv=v-(v@u8)@u8.T;ds=pv.square().sum(-1).numpy()
            self.assertAlmostEqual(post[f'projection/S0/r8/{mode}/post_sq'],float(upper_mean(ds,3)),places=6)
            self.assertAlmostEqual(post[f'projection/S0/r8/{mode}/post_sqrt'],float(upper_mean(np.sqrt(ds),3)),places=6)
        # Projection changes the winner; simply reranking an original winner is insufficient.
        q2=torch.tensor([[.8,.6,0.]])
        b2=F.normalize(torch.tensor([[[[.9,.1,.4],[0.,1.,0.]]]]),dim=-1)
        u2=torch.tensor([[1.],[0.],[0.]])
        a2,j2,p2=distance_cache(q2,b2,{'X':u2},ranks=(1,))
        self.assertEqual(int(j2[0,0,0]),0)
        self.assertEqual(int(p2['X/r1']['identity']['indices'][0]),1)
        self.assertTrue(np.isfinite(p2['X/r1']['identity']['distances']).all())
    def test_projection_fit_centering_nested(self):
        bank=F.normalize(torch.randn(3,9,16,48),dim=-1).numpy();bases,fit=fit_projections(bank)
        delta=(bank[:,1:].astype(np.float64)-bank[:,0:1].astype(np.float64)).reshape(-1,48)
        np.testing.assert_allclose(fit['mu'],delta.mean(0),atol=1e-15)
        np.testing.assert_allclose(fit['S0']['matrix'],delta.T@delta/len(delta),atol=1e-15)
        np.testing.assert_allclose(fit['Sc']['matrix'],np.cov(delta,rowvar=False,bias=True),atol=1e-15)
        for u in bases.values():np.testing.assert_allclose(u.T@u,np.eye(32),atol=3e-7)
    def test_bow_direction_grid_and_all_configs(self):
        b=F.normalize(torch.randn(10,9,256,48),dim=-1);q=F.normalize(torch.randn(256,48),dim=-1)
        centers,rh=fit_bow(b.numpy());self.assertEqual(rh.shape,(10,9,25,12));np.testing.assert_allclose(rh.sum(-1),1,atol=2e-7)
        h=bow_histograms(q.numpy(),centers)
        np.testing.assert_allclose(bow_costs(rh,h),np.sort((rh*(np.log(rh)-np.log(h))).sum(-1),axis=-1)[...,:20].mean(-1),atol=1e-7)
        bases,_=fit_projections(b.numpy());A,j,pre=distance_cache(q,b,bases,group_chunk=8)
        scores=all_readouts(q,b,A,j,pre,bases,[str(i) for i in range(10)],centers,rh);scores['robust_color']=0.
        self.assertEqual(len(scores),59);self.assertEqual(len(baseline_names(scores)),51)
        self.assertTrue(all(np.isfinite(list(scores.values()))));self.assertGreaterEqual(scores['primary']+1e-7,scores['pool9'])
        self.assertEqual(sum(n.startswith('projection/') for n in scores),36)
        self.assertEqual(sum(n.startswith('refselect/') for n in scores),12)
    def test_auroc_threshold_and_ties_synthetic_only(self):
        self.assertEqual(auroc([0,1],[0.,1.]),1.)
        self.assertEqual(auroc([0,1],[1.,1.]),.5)
        self.assertEqual(auroc([0,1,0,1],[0.,0.,1.,1.]),.5)
        self.assertEqual(threshold(np.arange(64)),61.)
        np.testing.assert_array_equal(positive([60.,61.,62.],61.),[False,False,True])
        self.assertEqual(threshold(np.ones(64)),1.)
    def test_near_components_and_ssim(self):
        c=Components(4);c.union(0,1);c.union(1,2);self.assertEqual(c.find(0),c.find(2));self.assertNotEqual(c.find(0),c.find(3))
        x=np.arange(256*256,dtype=np.uint8).reshape(256,256)
        self.assertTrue(is_near(0,x,15,x)[0]);self.assertFalse(is_near(0,x,31,x)[0])
    def test_summarize_requires_complete_seeds_methods_ids_labels_finite(self):
        manifest={'status':'ready','calibration':[{'cluster_id':digest(f'c{i}'),'canonical':{'path':f'train/{i}','pixelhash':digest(f'cp{i}')}} for i in range(64)],'tests':[]}
        for domain,label,count in [('same',0,230),('same',1,459),('illumination',0,75),('illumination',1,198)]:
            for i in range(count):
                path=f'{domain}/{label}/{i}';manifest['tests'].append({'path':path,'domain':domain,'label':label,'pixelhash':digest(path)})
        records=query_manifest_records(manifest);labels=np.array([r['label'] for r in records[64:]])
        values={name:np.r_[np.arange(64)/100.,labels if name=='primary' else 1-labels] for name in METHOD_NAMES}
        all_scores={str(seed):{k:v.copy() for k,v in values.items()} for seed in (11,22,33)}
        valid=summarize(all_scores,records,manifest)
        self.assertTrue(valid['continue']);self.assertEqual(len(valid['methods']),3)
        self.assertEqual(len(BASELINE_NAMES),51)
        mutations=[]
        bad=copy.deepcopy(all_scores);del bad['33'];mutations.append((bad,records,manifest))
        bad=copy.deepcopy(all_scores);del bad['11']['projection/S0/r8/identity/pre_sq'];mutations.append((bad,records,manifest))
        bad=copy.deepcopy(all_scores);bad['11']['unknown']=np.zeros(1026);mutations.append((bad,records,manifest))
        bad=copy.deepcopy(all_scores);bad['11']['primary'][100]=np.nan;mutations.append((bad,records,manifest))
        bad=copy.deepcopy(all_scores);bad['22']['primary']=np.zeros(1025);mutations.append((bad,records,manifest))
        r=copy.deepcopy(records);r[65]['query_id']=r[64]['query_id'];mutations.append((all_scores,r,manifest))
        r=copy.deepcopy(records);r[65]['path']=r[64]['path'];mutations.append((all_scores,r,manifest))
        r=copy.deepcopy(records);r[65]['label']=1;mutations.append((all_scores,r,manifest))
        r=copy.deepcopy(records);r[65]['query_id']='extra-id';mutations.append((all_scores,r,manifest))
        r=copy.deepcopy(records);r.append(r[0]);mutations.append((all_scores,r,manifest))
        m=copy.deepcopy(manifest);m['tests'][0]['label']=1;r=query_manifest_records(m);mutations.append((all_scores,r,m))
        for bad,rows,m in mutations:
            with self.subTest(case=len(rows),seeds=list(bad)):
                with self.assertRaises(ValueError):summarize(bad,rows,m)
    def test_formal_entry_requires_explicit_authorization(self):
        from formal_runner import run_formal
        with self.assertRaises(PermissionError):run_formal()

if __name__=='__main__':
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Numerics)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    save_json(WORK/'reports/synthetic-tests.json',{'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'seconds':time.monotonic()-start,'data':'synthetic only','real_AUROC_TPR_or_theta_distributions_computed':False})
    if not result.wasSuccessful():raise SystemExit(1)
