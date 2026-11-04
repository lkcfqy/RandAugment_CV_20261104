"""Small manufactured fixtures only; never open real panel images or models."""
import importlib
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np


class Oracle(unittest.TestCase):
    def module(self, name, function):
        try:
            mod = importlib.import_module('millproxy.' + name)
        except ModuleNotFoundError:
            mod = None
        self.assertIsNotNone(mod, 'Missing protocol module: ' + name)
        self.assertTrue(hasattr(mod, function), 'Missing protocol behavior: ' + function)
        return mod

    def test_q8_rounds_half_up_not_bankers(self):
        c = self.module('core', 'q8')
        np.testing.assert_array_equal(c.q8(np.array([.5,1.5,2.5,-255,510])/255)*255,[1,2,3,0,255])

    def test_gamma_grid_anchor_and_black_source_tie(self):
        a = self.module('fit', 'fit_a')
        p=a.fit_a(np.zeros((6,3,2,3)),np.ones((6,3,2,3))/255)
        self.assertEqual(p.gamma,1.0)
        np.testing.assert_array_equal(p.gain,[0,0,0])
        np.testing.assert_allclose(p.beta,0,atol=1e-30)
        self.assertEqual(len(a.gamma_grid()),4097)
        self.assertEqual(a.gamma_grid()[2048],1.0)

    def test_a_exact_shared_gamma_three_gain_and_no_mean_addback(self):
        a=self.module('fit','fit_a');s=self.module('synth','synthesize_a')
        x=np.broadcast_to(np.array([0,50,100,150,200,250])/255,(6,3,2,6)).copy()
        y=x*np.array([.2,.4,.6])[None,:,None,None]
        p=a.fit_a(x,y)
        self.assertEqual(p.gamma,1.0)
        np.testing.assert_allclose(p.gain,[.2,.4,.6],atol=1e-14)
        np.testing.assert_allclose(p.beta,0,atol=1e-28)
        p0=a.fit_a(np.zeros_like(x),np.ones_like(x)/255)
        np.testing.assert_array_equal(s.synthesize_a(x[0]*0,p0,'Test13',1,11),0)

    def test_nnls_minimum_norm_and_boundaries(self):
        a=self.module('fit','nnls2')
        np.testing.assert_allclose(a.nnls2([2],[.5],[1]),[.8,.4],atol=1e-14)
        np.testing.assert_allclose(a.nnls2([1,1],[0,1],[1,0]),[.5,0],atol=1e-14)
        np.testing.assert_allclose(a.nnls2([1,1],[.25,.75],[.2,.6]),[0,.8],atol=1e-14)
        np.testing.assert_array_equal(a.nnls2([],[],[]),[0,0])

    def test_pav_weighted_pooling_and_empty_node_interpolation(self):
        a=self.module('fit','monotone_lut')
        n=np.zeros(256);s=np.zeros(256);n[[0,128,255]]=[1,3,1];s[[0,128,255]]=[1,0,1]
        lut=a.monotone_lut(n,s)
        np.testing.assert_allclose(lut[[0,64,128,255]],[.25,.25,.25,1])
        self.assertAlmostEqual(lut[192],.25+.75*64/127)
        n[:]=0;s[:]=0;n[100]=2;s[100]=1.4
        np.testing.assert_allclose(a.monotone_lut(n,s),.7)

    def test_bins_boundaries_and_original_empty_fallback_lower_tie(self):
        c=self.module('core','source_bins');a=self.module('fit','nearest_nonempty')
        np.testing.assert_array_equal(c.equal_width_bins(np.arange(17)/16),np.minimum(np.arange(17),15))
        x=np.broadcast_to(np.array([0,128,255])/255,(3,1,3))
        np.testing.assert_array_equal(c.source_bins(x),[[0,8,15]])
        n=np.zeros(16,int);n[[0,2]]=1
        self.assertEqual(a.nearest_nonempty(n)[1],0)
        self.assertEqual(a.nearest_nonempty(n)[15],2)

    def test_b_zero_empirical_noise_and_mu_addback(self):
        a=self.module('fit','fit_b');s=self.module('synth','synthesize_b')
        x=np.zeros((6,3,4,6)); y=np.ones_like(x)*51/255
        p=a.fit_b(x,y)
        self.assertTrue(np.all(p.sigma>0))
        self.assertTrue(all(np.all(q==0) for row in p.quantiles for q in row))
        out,meta=s.synthesize_b(x[0],p,'Test13',1,11)
        np.testing.assert_array_equal(out*255,51)
        from dataclasses import replace
        pm=replace(p,mu=np.ones((16,3))*.2)
        shifted,_=s.synthesize_b(x[0],pm,'Test13',1,11)
        np.testing.assert_array_equal(shifted*255,102)
        self.assertEqual(meta['iterations'],10)

    def test_step_quantiles_and_rank_tie_priority(self):
        s=self.module('synth','rank_map_channel')
        q=tuple(np.array([0,1,3,7.]) for _ in range(16))
        u=np.zeros((1,3));bins=np.zeros((1,3),int);priority=np.array([[2,0,1]])
        np.testing.assert_array_equal(s.rank_map_channel(u,bins,q,priority),[[7,0,1]])

    def test_principal_root_and_material_negative_eigenvalue_rejected(self):
        f=self.module('spectral','principal_root')
        S=np.zeros((4,6,3,3),complex);S[:]=np.diag([4,9,0]);S[0,0]=0
        L=f.principal_root(S)
        np.testing.assert_allclose(L[1,1],np.diag([2,3,0]),atol=1e-14)
        S[1,1,2,2]=-.1
        with self.assertRaises(ValueError):f.principal_root(S)

    def test_projection_preserves_outer_product_and_only_four_self_conjugates(self):
        f=self.module('spectral','orbit_project')
        rng=np.random.Generator(np.random.PCG64(7))
        v=np.fft.fft2(rng.normal(size=(3,8,10)),norm='ortho');v[:,0,0]=0
        w=np.fft.fft2(rng.normal(size=(3,8,10)),norm='ortho')
        p=f.orbit_project(v,w)
        outer=lambda x:np.einsum('ahw,bhw->hwab',x,x.conj())
        np.testing.assert_allclose(outer(p),outer(v),atol=1e-12)
        self.assertLess(np.max(np.abs(np.fft.ifft2(p,norm='ortho').imag)),1e-12)
        self.assertTrue(np.any(np.abs(p[:,4,1].imag)>1e-5))
        for iy,ix in [(0,0),(0,5),(4,0),(4,5)]:np.testing.assert_array_equal(p[:,iy,ix].imag,0)
        zero=f.orbit_project(v,np.zeros_like(v));np.testing.assert_allclose(zero,v,atol=1e-12)

    def test_parseval_and_bad_ifft_imaginary_rejected(self):
        f=self.module('spectral','real_ifft')
        x=np.arange(3*4*6,dtype=float).reshape(3,4,6)/71
        z=np.fft.fft2(x,norm='ortho')
        self.assertAlmostEqual(float(np.sum(abs(z)**2)),float(np.sum(x*x)),places=12)
        np.testing.assert_allclose(f.real_ifft(z),x,atol=1e-14)
        z[0,1,1]+=1j
        with self.assertRaises(ValueError):f.real_ifft(z)

    def test_rng_domains_are_reproducible_and_separated(self):
        c=self.module('core','rng_for')
        a=c.rng_for('B','Test13',3,22,'gaussian').standard_normal((3,4,6))
        b=c.rng_for('B','Test13',3,22,'gaussian').standard_normal((3,4,6))
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,c.rng_for('A','Test13',3,22,'gaussian').standard_normal(a.shape)))
        self.assertFalse(np.array_equal(a,c.rng_for('B','Test13',3,22,'ties-R').standard_normal(a.shape)))

    def test_psnr_rgb_mse_zero_fails_and_lab_reference(self):
        m=self.module('metrics','psnr')
        gt=np.zeros((3,2,2));p=np.broadcast_to(np.array([.1,.2,.3])[:,None,None],gt.shape)
        self.assertAlmostEqual(m.psnr(p,gt),-10*math.log10(14/300),places=12)
        with self.assertRaises(ValueError):m.psnr(gt,gt)
        red=m.lab(np.array([1.,0.,0.])[:,None,None])[:,0,0]
        np.testing.assert_allclose(red,[53.240794,80.092460,67.203197],atol=2e-5)
        self.assertAlmostEqual(m.delta_e76(np.ones_like(gt),gt),100,places=3)

    def test_final_statistics_per_draw_absolute_errors_do_not_cancel(self):
        d=self.module('diagnostics','bin_errors')
        real=np.zeros((3,2,2));draws=np.stack([real+1/255,real-1/255,real+1/255,real-1/255]);bins=np.zeros((2,2),int)
        e=d.bin_errors(real,draws,bins)
        self.assertAlmostEqual(e['Emean'],1/255,places=15)
        self.assertEqual(e['Evar'],0)
        self.assertEqual(d.ks2(np.array([0,0,1]),np.array([0,1,1])),1/3)

    def test_stat_real_degenerate_unidentifiable_and_identical_pass(self):
        d=self.module('diagnostics','diagnose_residuals')
        rng=np.random.Generator(np.random.PCG64(9));u=rng.normal(size=(3,8,10));bins=np.zeros((8,10),int)
        same=d.diagnose_residuals(u,np.stack([u]*4),bins,synthetic_degeneracy='variance_1e-12')
        self.assertTrue(same['identifiable']);self.assertTrue(same['pass'])
        flat=d.diagnose_residuals(u*0,np.stack([u]*4),bins,synthetic_degeneracy='variance_1e-12')
        self.assertEqual(flat['status'],'STAT_UNIDENTIFIABLE');self.assertFalse(flat['pass'])
        bad=d.diagnose_residuals(u,np.zeros((4,3,8,10)),bins,synthetic_degeneracy='variance_1e-12')
        self.assertEqual(bad['EPSD'],2);self.assertEqual(bad['Ecor'],2)

    def test_pair_threshold_ties_and_sufficient_scene_coverage(self):
        d=self.module('decision','pair_agreement')
        r=np.arange(8,dtype=float)
        perfect=d.pair_agreement(r,r);self.assertTrue(perfect['sufficient']);self.assertEqual(perfect['agreement'],1)
        tie=d.pair_agreement(r,r*0);self.assertEqual(tie['agreement'],.5)
        self.assertEqual(d.pair_agreement(np.array([0,.5]),np.array([0,1e-8]))['agreement'],.5)
        self.assertFalse(d.pair_agreement(r,r,mask=np.arange(8)<5)['sufficient'])

    def test_anchor_control_exposes_shared_anchor_sorting(self):
        d=self.module('decision','responses')
        T=np.arange(8,dtype=float)+20;R=np.ones((8,4))*20;S=np.ones((8,4,4))*20
        result=d.responses(T,R,S,np.arange(6,dtype=float)+20,np.ones((6,4))*20,color=False)
        np.testing.assert_array_equal(result['real'],T[:,None]-R)
        np.testing.assert_array_equal(result['C_anchor'],result['real'])
        absolute=d.pair_agreement(R[:,0],S[:,0].mean(1))
        self.assertEqual(absolute['eligible_pairs'],0)

    def test_close_has_priority_over_stat_match_failure_candidate(self):
        d=self.module('decision','decide')
        real=np.repeat(np.arange(8.)[:,None],4,axis=1);pred={'A':real.copy(),'B':-real,'C_anchor':real+2,'C_meanD':np.zeros((8,4))}
        colors={k:np.zeros((8,4)) for k in pred}
        result=d.decide(real,np.zeros((8,4)),pred,colors,np.ones((8,4),bool),c_anchor_confirmed=True)
        self.assertEqual(result['decision'],'CLOSE_KNOWN_PROXY_SUFFICIENT');self.assertIn('A',result['close_methods'])
        blocked=d.decide(real,np.zeros((8,4)),pred,colors,np.ones((8,4),bool),validation_ok=False,c_anchor_confirmed=True)
        self.assertEqual(blocked['decision'],'VALIDATION_BLOCKED')

    def test_continue_requires_four_distinct_scenes_and_two_sufficient_levels(self):
        d=self.module('decision','decide')
        real=np.repeat(np.arange(8.)[:,None],4,axis=1)
        pred={k:-real for k in ['A','B','C_anchor','C_meanD']};colors={k:np.ones((8,4))*5 for k in pred}
        good=np.ones((8,4),bool)
        result=d.decide(real,np.zeros((8,4)),pred,colors,good,c_anchor_confirmed=True)
        self.assertEqual(result['decision'],'RETAIN_NARROW_MEASUREMENT_PENDING_REVIEW')
        good[:,1:]=False
        result=d.decide(real,np.zeros((8,4)),pred,colors,good,c_anchor_confirmed=True)
        self.assertEqual(result['decision'],'NO_UPGRADE')

    def test_missing_any_forward_id_method_seed_or_finite_metric_rejects_go(self):
        a=self.module('audit','expected_forward_keys')
        keys=a.expected_forward_keys();self.assertEqual(len(keys),326)
        records={k:{'psnr':20.,'delta_e76':2.,'input_sha256':'a'*64} for k in keys}
        a.validate_records(records)
        k=next(iter(records));v=records.pop(k)
        with self.assertRaises(ValueError):a.validate_records(records)
        records[k]=v;records[k]['psnr']=float('nan')
        with self.assertRaises(ValueError):a.validate_records(records)

    def test_cache_hash_detects_changed_parameters_without_pickle(self):
        a=self.module('audit','save_arrays')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cache';a.save_arrays(p,{'x':np.arange(6,dtype=float)}, {'method':'oracle'})
            arrays,meta=a.load_arrays(p);np.testing.assert_array_equal(arrays['x'],np.arange(6));self.assertFalse(arrays['x'].flags.writeable)
            f=p.with_suffix('.npz');content=bytearray(f.read_bytes());content[-10]^=1;f.write_bytes(content)
            with self.assertRaises(ValueError):a.load_arrays(p)

    def test_cluster_pair_weights_exclude_self_copies(self):
        d=self.module('decision','pair_agreement')
        result=d.pair_agreement(np.arange(8.),np.arange(8.),multiplicity=np.array([8,0,0,0,0,0,0,0]))
        self.assertIsNone(result['agreement']);self.assertEqual(result['pair_weight'],0)
        result=d.pair_agreement(np.arange(8.),np.arange(8.),multiplicity=np.array([2,3,0,0,0,0,0,0]))
        self.assertEqual(result['pair_weight'],6)

    def test_budget_formula_and_pending_approval_rejects_execution(self):
        a=self.module('audit','resource_bound')
        self.assertEqual(a.resource_bound(10,1,2,3),10+1.25*(326+64+3))
        with self.assertRaises(ValueError):a.require_release({'static_review':False,'protocol_confirmed':False},'a'*64)

    def test_gain_clipped_at_two_on_dark_codes(self):
        a=self.module('fit','fit_a')
        x=np.broadcast_to(np.array([0,1])/255,(6,3,2,2)).copy()
        p=a.fit_a(x,np.ones_like(x))
        np.testing.assert_array_equal(p.gain,[2,2,2])
        self.assertAlmostEqual(p.gamma,.2)

    def test_training_cross_spectrum_uses_six_scene_mean_and_ortho_fft(self):
        f=self.module('spectral','training_spectrum')
        yy,xx=np.indices((4,4));base=(-1.)**(yy+xx)
        z=np.repeat(np.stack([base,base*2,base*0])[None],6,axis=0)
        spec=f.training_spectrum(z)
        np.testing.assert_allclose(spec[2,2],[[16,32,0],[32,64,0],[0,0,0]],atol=1e-12)
        np.testing.assert_array_equal(spec[0,0],0)

    def test_b_finishes_with_rank_map_without_last_spectral_projection(self):
        a=self.module('fit','fit_b');s=self.module('synth','synthesize_b')
        from dataclasses import replace
        x=np.zeros((6,3,4,6));p=a.fit_b(x,x)
        quantiles=tuple(tuple(np.array([0.,1.]) for _ in range(3)) for _ in range(16))
        p=replace(p,sigma=np.ones((16,3))*.2,quantiles=quantiles)
        out,_=s.synthesize_b(x[0],p,'Test13',1,11)
        for c in range(3):np.testing.assert_array_equal(np.sort(out[c].ravel()),[0]*12+[.2]*12)

    def test_complex_coherence_detects_phase_change_at_equal_psd(self):
        d=self.module('diagnostics','diagnose_residuals')
        phase=np.broadcast_to(np.arange(8)*2*np.pi/8,(8,8))
        real=np.stack([np.cos(phase),np.sin(phase),np.cos(phase+np.pi/4)])
        syn=real.copy();syn[1]*=-1
        result=d.diagnose_residuals(real,np.stack([syn]*4),np.zeros((8,8),int),synthetic_degeneracy='variance_1e-12')
        self.assertTrue(result['identifiable']);self.assertAlmostEqual(result['EPSD'],0,places=12)
        self.assertAlmostEqual(result['Ecoh'],2,places=12)

    def test_b_parameters_roundtrip_and_field_hashes(self):
        a=self.module('pipeline','save_parameters');f=self.module('fit','fit_b')
        x=np.zeros((6,3,4,6));p=f.fit_b(x,x)
        with tempfile.TemporaryDirectory() as td:
            a.save_parameters(Path(td)/'B-1',p,1)
            restored=a.load_parameters(Path(td)/'B-1')
            np.testing.assert_array_equal(restored.lut,p.lut)
            np.testing.assert_array_equal(restored.root,p.root)
            self.assertFalse(restored.root.flags.writeable)
            with np.load(Path(td)/'B-1.npz',allow_pickle=False) as archive:
                self.assertEqual(sum(archive[k].size for k in archive.files if k.startswith('Q_')),x.size)

    def test_generation_api_has_no_test_target_or_gt_channel(self):
        s=self.module('synth','synthesize_a');f=self.module('fit','fit_a')
        x=np.zeros((6,3,2,2));p=f.fit_a(x,x)
        with self.assertRaises(TypeError):s.synthesize_a(x[0],p,'Test13',1,11,test_y=x[0])
        with self.assertRaises(TypeError):f.fit_a(x,x,gt=x)

    def test_panel_metadata_rejects_missing_or_renamed_scene_without_reading_images(self):
        p=self.module('pipeline','PanelReader')
        # Deliberately nonexistent file: reject identity/hash before any image read.
        with tempfile.TemporaryDirectory() as td:
            import json
            root=Path(td);(root/'reports').mkdir();(root/'reports/panel.json').write_text('[]\n')
            with self.assertRaises(ValueError):p.PanelReader(root)

    def test_preflight_uses_warmup_and_max_of_three_and_pending_receipt_blocks(self):
        r=self.module('runtime','preflight')
        from time import monotonic
        calls=[];freeze_sha=self.module('audit','current_code_hash').current_code_hash()
        receipt={'freeze_sha256':freeze_sha,'static_review':True,'protocol_confirmed':True,'c_anchor_confirmed':True,'synthetic_degeneracy':'variance_1e-12','absolute_ranking_eligibility':'absolute_R','unidentifiable_policy':'global'}
        out=r.preflight(lambda:calls.append('forward'),lambda:calls.append('cell'),lambda:None,t_spent=0,t_post=1,receipt=receipt,freeze_sha=freeze_sha)
        self.assertEqual(calls,['forward']*4+['cell']);self.assertEqual(out['t_forward'],max(out['three_forward_seconds']))
        with self.assertRaises(ValueError):r.preflight(lambda:None,lambda:None,lambda:None,t_spent=0,t_post=1,receipt={},freeze_sha='a'*64)

    def test_small_manufactured_pipeline_accounts_exact_326_input_only_calls(self):
        p=self.module('pipeline','prepare_generated');r=self.module('runtime','ContinuousBudget');c=self.module('core','q8')
        class ManufacturedReader:
            def __init__(self):self.evaluation_open=False
            def source(self,scene):
                # Distinct manufactured 4x6 arrays; no filesystem images.
                index=(list(c.TRAIN_IDS)+list(c.TEST_IDS)).index(scene)
                return c.q8(np.broadcast_to(np.linspace(.1,.7,24).reshape(1,4,6),(3,4,6))+index/255)
            def training_xy(self,level):
                xx=np.stack([self.source(s) for s in c.TRAIN_IDS]);return xx,c.q8(xx*(level+2)/12)
            def test_x(self,scene):return self.source(scene)
            def unlock_evaluation(self,cache):self.evaluation_open=True
            def evaluation_image(self,split,scene,tag):
                if not self.evaluation_open:raise ValueError('Targets opened before generation')
                if tag=='GT':return np.ones((3,4,6))*.95
                if tag=='L10':return self.source(scene)
                return c.q8(self.source(scene)*(int(tag[1:])+2)/12)
        freeze_sha=self.module('audit','current_code_hash').current_code_hash()
        receipt={'freeze_sha256':freeze_sha,'static_review':True,'protocol_confirmed':True,'c_anchor_confirmed':True,'synthetic_degeneracy':'variance_1e-12','absolute_ranking_eligibility':'absolute_R','unidentifiable_policy':'global','scope':'formal'}
        reader=ManufacturedReader()
        with tempfile.TemporaryDirectory() as td:
            budget=r.ContinuousBudget(30)
            p.prepare_generated(reader,Path(td),receipt=receipt,freeze_sha=freeze_sha,budget=budget)
            result=p.evaluate_generated(reader,Path(td),lambda x:(x*.8+.1).astype(np.float32),receipt=receipt,freeze_sha=freeze_sha,budget=budget)
            self.assertEqual(len(result['records']),326)
            self.assertEqual(result['forward_count'],326)
            self.assertEqual(result['synthetic_images'],256)
            self.assertEqual(set(result['stat_diagnostics']),{'A','B'})

    def test_close_report_compares_anchor_control_on_same_stat_subset_pairs(self):
        d=self.module('decision','decide')
        real=np.repeat(np.arange(8.)[:,None],4,axis=1);pred={'A':real,'B':real+2,'C_anchor':-real,'C_meanD':np.zeros((8,4))}
        mask=np.ones((8,4),bool);mask[7]=False;colors={k:np.zeros((8,4)) for k in pred}
        result=d.decide(real,np.zeros((8,4)),pred,colors,mask,c_anchor_confirmed=True)
        self.assertEqual(result['decision'],'CLOSE_KNOWN_PROXY_SUFFICIENT')
        self.assertIn('stat_pass_endpoint_pairs',result)
        for level in ['1','3','5','7']:
            bp=result['stat_pass_endpoint_pairs']['B'][level]['pairs'];cp=result['stat_pass_endpoint_pairs']['C_anchor'][level]['pairs']
            self.assertEqual([x[:2] for x in bp],[x[:2] for x in cp])

    def test_variance_errors_are_absolute_per_draw_before_average(self):
        d=self.module('diagnostics','bin_errors')
        real=np.broadcast_to(np.array([-1.,1.])/255,(3,1,2)).copy()
        draw=np.stack([real*0,real*np.sqrt(2),real*0,real*np.sqrt(2)])
        self.assertAlmostEqual(d.bin_errors(real,draw,np.zeros((1,2),int))['Evar'],1,places=12)

    def test_model_adapter_blocks_unreviewed_execution_before_loading_any_file(self):
        m=self.module('model','CIDNetPredictor')
        with self.assertRaises(ValueError):m.CIDNetPredictor(Path('/nonexistent-mill-assets'),device='cpu',receipt={},freeze_sha='a'*64)

    def test_stale_code_freeze_is_rejected_even_if_receipt_matches_old_string(self):
        a=self.module('audit','require_release')
        receipt={'freeze_sha256':'a'*64,'static_review':True,'protocol_confirmed':True,'c_anchor_confirmed':True,'synthetic_degeneracy':'variance_1e-12','absolute_ranking_eligibility':'absolute_R','unidentifiable_policy':'global'}
        with self.assertRaises(ValueError):a.require_release(receipt,'a'*64)

    def test_external_worker_timeout_is_resource_block_not_scientific_negative(self):
        r=self.module('runtime','run_worker_bounded')
        import sys
        out=r.run_worker_bounded([sys.executable,'-c','while True: pass'],remaining_seconds=.03,cwd=Path(__file__).parent)
        self.assertTrue(out['resource_blocked']);self.assertFalse(out['scientific_negative'])

    def test_rng_exact_utf8_first16_big_endian_pcg64_domain(self):
        c=self.module('core','rng_for')
        raw=c.rng_for('B','Test13',3,22,'gaussian').bit_generator.random_raw(3)
        np.testing.assert_array_equal(raw,np.array([16258144944760729272,6873634517694092025,15806597009266432717],dtype=np.uint64))

    def test_constant_empirical_bin_residual_is_exact_zero_mass(self):
        f=self.module('fit','fit_b')
        x=np.ones((6,3,4,6))
        for scene in range(6):x[scene].reshape(3,-1)[:,:scene+1]=0
        y=np.where(x==0,204/255,51/255)
        p=f.fit_b(x,y)
        for b in [0,15]:
            for c in range(3):np.testing.assert_array_equal(p.quantiles[b][c],0)

    def test_absolute_sorting_uses_R_eligibility_and_not_D_eligibility(self):
        d=self.module('decision','absolute_ranking')
        R=np.ones((8,4))*20;D=np.repeat(np.arange(8.)[:,None],4,axis=1)
        result=d.absolute_ranking(R,R,D,eligibility='absolute_R')
        self.assertEqual(result['levels']['1']['eligible_pairs'],0)
        with self.assertRaises(ValueError):d.absolute_ranking(R,R,D,eligibility='real_D')

    def test_persistent_compute_ledger_never_resets_oracle_or_stage_costs(self):
        r=self.module('runtime','ComputeLedger')
        with tempfile.TemporaryDirectory() as td:
            now=[1000.];path=Path(td)/'ledger.json'
            ledger=r.ComputeLedger(path,oracle_charge=12,clock=lambda:now[0])
            self.assertEqual(ledger.remaining,1788)
            session=ledger.begin('preflight');self.assertEqual(session['reserved_seconds'],120)
            now[0]+=10;ledger.finish('PASS')
            resumed=r.ComputeLedger(path,oracle_charge=15,clock=lambda:now[0])
            self.assertEqual(resumed.spent,25);self.assertEqual(resumed.remaining,1775)
            resumed.begin('formal')
            with self.assertRaises(ValueError):r.ComputeLedger(path,oracle_charge=15,clock=lambda:now[0]).begin('formal')

    def test_disk_guard_reserves_writes_before_exceeding_cap(self):
        r=self.module('runtime','DiskGuard')
        with tempfile.TemporaryDirectory() as td:
            guard=r.DiskGuard(Path(td),limit_bytes=4096)
            with self.assertRaises(TimeoutError):guard.reserve(8192)
            self.assertEqual(list(Path(td).iterdir()),[])

    def test_scheduler_formal_requires_successful_same_freeze_preflight(self):
        s=self.module('scheduler','launch');a=self.module('audit','current_code_hash')
        receipt={'freeze_sha256':a.current_code_hash(),'static_review':True,'protocol_confirmed':True,'c_anchor_confirmed':True,'synthetic_degeneracy':'variance_1e-12','absolute_ranking_eligibility':'absolute_R','unidentifiable_policy':'global','scope':'formal'}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):s.launch('formal',Path(td),receipt,device='cpu',tpost=1)

    def test_ks_exact_five_percent_boundary_does_not_round_above_gate(self):
        d=self.module('diagnostics','ks2')
        self.assertLessEqual(d.ks2(np.zeros(20),np.r_[np.zeros(19),1]),.05)

    def test_worker_requires_matching_parent_active_session_before_science(self):
        import json
        s=self.module('scheduler','validate_worker_session')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'implementation-v1/run-state';run.mkdir(parents=True)
            active={'kind':'preflight','started_monotonic':100.,'reserved_seconds':120.,'spent_before':19.}
            job={**active,'mode':'preflight','asset_root':str(root),'run_root':str(run),'receipt':{'scope':'preflight'}}
            (run/'compute_ledger.json').write_text(json.dumps({'active':active}))
            (run/'compute_ledger.lock').write_text('parent-owned fixture')
            s.validate_worker_session(job)
            with self.assertRaises(ValueError):s.validate_worker_session(job|{'mode':'formal'})
            with self.assertRaises(ValueError):s.validate_worker_session(job|{'reserved_seconds':1800.})
            (run/'compute_ledger.lock').unlink()
            with self.assertRaises(ValueError):s.validate_worker_session(job)

    def test_bootstrap_no_eligible_pairs_are_NA_without_resampling(self):
        d=self.module('decision','cluster_bootstrap')
        result=d.cluster_bootstrap(np.zeros((8,4)),np.zeros((8,4)))
        self.assertEqual(result['replicates'],999);self.assertEqual(result['agreement_NA_replicates'],999)
        self.assertEqual(result['agreement_percentile95'],[None,None]);self.assertEqual(result['mae_percentile95'],[0,0])


if __name__=='__main__':unittest.main()
