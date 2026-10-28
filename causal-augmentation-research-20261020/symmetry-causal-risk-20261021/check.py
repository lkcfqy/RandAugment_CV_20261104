"""Exact/rational population-mechanism check. No vision model or GPU."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,resource,time
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    with (ROOT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def serial(q):return dict(exact=str(q),decimal=float(q))
def quad(v,S):return sum(v[i]*S[i][j]*v[j] for i in range(2) for j in range(2))
def mv(M,v):return [sum(a*b for a,b in zip(row,v)) for row in M]
def transpose(M):return [list(x) for x in zip(*M)]
def mm(A,B):return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def inverse(M):
    det=M[0][0]*M[1][1]-M[0][1]*M[1][0];assert det>0
    return [[M[1][1]/det,-M[0][1]/det],[-M[1][0]/det,M[0][0]/det]]
def subtract(A,B):return [[A[i][j]-B[i][j] for j in range(2)] for i in range(2)]
def arr(M):return np.array(M,dtype=np.float64)

def setup():
    p=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        question='Does exact structural outcome invariance alone ensure non-increasing original-do(X)-marginal causal excess risk for population OLS under non-Haar random reflection?',
        scope='2D analytic SCM/population mechanism check; not real-image training or counterevidence to a theorem whose assumptions exclude this kernel',
        covariance=[['1','9/10'],['9/10','1']],f=['1','0'],confounding_bias_v=['1','1/10'],
        SCM='C,NX mutually independent centered Gaussians covarianceSigma/2; NY independent N(0,1). X=C+NX; Y=f^T X+2v^T C+NY. Thus xi=2v^T C+NY, c=Cov(X,xi)=Sigma v. Nondegenerate acyclic SCM.',
        augmentation='G=diag(1,S), S=-1 withprob1/5 and+1withprob4/5; independent of C,NX,NY; retainY. f(GX)=f(X)pointwise. No input-coordinate redefinition at evaluation.',
        exact_estimand='CER_original(h)=E_{Xnew~N(0,Sigma),independentC,NY}[(Ynew-h^TXnew)^2-(Ynew-f^TXnew)^2]=(h-f)^T Sigma(h-f)',
        population_OLS='h0=f+Sigma^-1 c; for A=E G,B=E G Sigma G^T, hDA=f+B^-1 A c',
        exact_controls=['identitykernel','Haaruniformreflectionprob1/2','uncorrelatedcoordinatesrho0','zero_hidden_confounding'],
        fixed_composition_counts=list(range(13)),composition='k independent copies of the same reflectionkernel; mu=(3/5)^k; no selection of bestk',
        general_condition='M=A^T B^-1 Sigma B^-1 A-Sigma^-1. CERDA-CER0=c^T M c; no harm for every feasiblec iff M negative semidefinite, when B andSigma positive definite and everyG fixesf.',
        reflection_lemma='ForSigma=[[1,rho],[rho,1]],0<abs(rho)<1, no-ridge OLS and mu=E S in[-1,1], universal non-harm holds only at mu0(Haar) ormu1(identity). T=B^-1 A Sigma fixes e1. H=T^T Sigma T-Sigma has H11=0,H12=rho*mu*(1-mu)*(1-rho^2)/(1-mu^2*rho^2); nonzero H12 makesdetH<0. This is a proposed elementary boundary lemma, originality unestablished.',
        numerical_checks='exactFraction2x2 normal equations and risk; float64 direct matrix agreement<1e-12; independent2D SCM simulation only as sanity check',
        simulation=dict(n=200000,train_seed=2026102121,doX_seed=2026102122,OLS='direct leastsquares2columns no intercept; centered population; no iteration/tuning',
            outcome_invariance_max_abs=0.,purpose='verify SCMinvariance, moments and analytical predictions; not an independent CV efficacy test',
            fixed_CER_original_population_vs_doX_MC_abs_tolerance=.03),
        resources=dict(CPU_threads=1,nice_increment=10,GPU=False,MPS=False,A100=False,ARIS=False,downloads=False,software_changes=False,other_project_access=False),
        novelty_status='boundary-observation candidate only; Haar projection, generic DA normal equations and bias shifts already studied; no exact-equivalence claim until closest2026work checked',
        model_request='GPT-6.1-Sol/max priority; actual execution receipt unavailable')
    save('protocol.json',p)
    save('protocol-lock.json',dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='analytic_protocol_locked',protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)))),flush=True)

def main():
    try:os.nice(10)
    except OSError:pass
    t=time.perf_counter();p=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'protocol-lock.json').read_text())
    assert sha(ROOT/'protocol.json')==lock['protocol_sha256'] and sha(Path(__file__))==lock['code_sha256']
    rho=Q(9,10);S=[[Q(1),rho],[rho,Q(1)]];v=[Q(1),Q(1,10)];f=[Q(1),Q(0)];c=mv(S,v);invS=inverse(S);risk0=quad(v,S)
    rows=[]
    for k in p['fixed_composition_counts']:
        mu=Q(3,5)**k;A=[[Q(1),Q(0)],[Q(0),mu]];B=[[Q(1),mu*rho],[mu*rho,Q(1)]];invB=inverse(B)
        b=mv(invB,mv(A,c));risk=quad(b,S);T=mm(mm(invB,A),S);H=subtract(mm(mm(transpose(T),S),T),S)
        M=subtract(mm(mm(mm(transpose(A),invB),S),mm(invB,A)),invS)
        d=rho*mu*(1-mu)*(1-rho*rho)/(1-mu*mu*rho*rho)
        assert H[0][0]==0 and H[0][1]==H[1][0]==d and H[0][0]*H[1][1]-H[0][1]**2==-d*d
        assert quad(c,M)==risk-risk0
        direct=np.linalg.solve(arr(B),arr(A)@arr(c));assert np.max(np.abs(direct-arr(b)))<1e-12
        rows.append(dict(k=k,mu=serial(mu),effective_flip_probability=serial((1-mu)/2),population_coefficient_bias=[serial(a) for a in b],
            CER=serial(risk),CER_minus_original=serial(risk-risk0),risk_under_augmented_do_covariance=serial(quad(b,B)),
            H12=serial(d),H_determinant=serial(-d*d),H_float_eigenvalues=np.linalg.eigvalsh(arr(H)).tolist()))
    assert rows[0]['CER']['exact']=='119/100' and Q(rows[1]['CER_minus_original']['exact'])>0
    # Independent exact controls, including endpoints not in the fixed k list.
    def risk_mu(mu,correlation=rho,bias=v):
        covariance=[[Q(1),correlation],[correlation,Q(1)]];covc=mv(covariance,bias)
        bb=[[Q(1),mu*correlation],[mu*correlation,Q(1)]];aa=[[Q(1),Q(0)],[Q(0),mu]]
        return quad(mv(inverse(bb),mv(aa,covc)),covariance)
    haar=risk_mu(Q(0));assert haar==Q(11881,10000)<risk0
    assert risk_mu(Q(1))==risk0
    assert risk_mu(Q(3,5),Q(0))<=quad(v,[[Q(1),Q(0)],[Q(0),Q(1)]])
    assert risk_mu(Q(3,5),bias=[Q(0),Q(0)])==0
    # Exact symbolic-coefficient identity checked at a fixed grid, not a search.
    lemma_cases=0
    for r in [Q(-9,10),Q(-1,2),Q(1,2),Q(9,10)]:
        for mu in [Q(-1),Q(-3,5),Q(0),Q(1,5),Q(3,5),Q(1)]:
            ss=[[Q(1),r],[r,Q(1)]];aa=[[Q(1),Q(0)],[Q(0),mu]];bb=[[Q(1),r*mu],[r*mu,Q(1)]];tt=mm(mm(inverse(bb),aa),ss)
            hh=subtract(mm(mm(transpose(tt),ss),tt),ss);d=r*mu*(1-mu)*(1-r*r)/(1-mu*mu*r*r)
            assert hh[0][0]==0 and hh[0][1]==d
            if mu not in [Q(0),Q(1)]:assert -d*d<0
            elif mu==0:assert hh==[[Q(0),Q(0)],[Q(0),-(1-r*r)]]
            else:assert all(a==0 for row in hh for a in row)
            lemma_cases+=1
    n=p['simulation']['n'];rng=np.random.default_rng(p['simulation']['train_seed']);Sf=arr(S);vf=arr(v);ff=arr(f)
    C=rng.multivariate_normal([0.,0.],Sf/2,n);NX=rng.multivariate_normal([0.,0.],Sf/2,n);NY=rng.normal(size=n);X=C+NX
    Y=X@ff+2*C@vf+NY;sign=np.where(rng.random(n)<.2,-1.,1.);Xa=X.copy();Xa[:,1]*=sign
    Ysoft=Xa@ff+2*C@vf+NY;assert np.array_equal(Y,Ysoft)
    beta0=np.linalg.lstsq(X,Y,rcond=None)[0];betaa=np.linalg.lstsq(Xa,Y,rcond=None)[0]
    pop0=ff+vf;popa=ff+np.array([q['decimal'] for q in rows[1]['population_coefficient_bias']])
    rg=np.random.default_rng(p['simulation']['doX_seed']);Xdo=rg.multivariate_normal([0.,0.],Sf,n)
    mc0=float(np.mean((Xdo@(pop0-ff))**2));mca=float(np.mean((Xdo@(popa-ff))**2))
    assert abs(mc0-float(risk0))<.03 and abs(mca-rows[1]['CER']['decimal'])<.03
    assert np.max(np.abs(beta0-pop0))<.03 and np.max(np.abs(betaa-popa))<.03
    paired=(Xdo@(popa-ff))**2-(Xdo@(pop0-ff))**2
    result=dict(status='exact_population_counterexample_and_controls_verified',at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'protocol.json'),baseline_CER=serial(risk0),one_composition_CER=rows[1]['CER'],
        one_composition_CER_increase=rows[1]['CER_minus_original'],relative_increase=rows[1]['CER']['decimal']/float(risk0)-1,
        Haar_endpoint_CER=serial(haar),composition_curve=rows,all_exact_identity_checks_passed=True,finite_grid_lemma_checks=lemma_cases,
        MC=dict(n=n,exact_Y_soft_intervention_equal=True,fitted_original_OLS=beta0.tolist(),fitted_augmented_OLS=betaa.tolist(),
            original_population_OLS=pop0.tolist(),augmented_population_OLS=popa.tolist(),
            doX_oracle_population_predictor_risks=[mc0,mca],paired_population_predictor_risk_difference=float(paired.mean()),
            paired_difference_MC_standard_error=float(paired.std(ddof=1)/np.sqrt(n)),no_actual_causal_effect_estimated_from_real_data=True),
        labels_or_outcomes_changed=False,target_function_in_linear_model_class=True,finite_sample_noise_needed_for_counterexample=False,
        actual_CV_training_evidence=False,refutation_of_Akbar_Haar_or_additive_theorem=False,new_contribution_established=False,
        next='independent proof and closest-work audit; retain only if a meaningful unresolved distinction survives; no vision training yet',
        wall_seconds=time.perf_counter()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CPU_threads_requested=1,GPU=False,MPS=False,A100=False,ARIS=False)
    save('results.json',result)
    print(json.dumps({k:x for k,x in result.items() if k!='composition_curve'},indent=2,ensure_ascii=False))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='setup':setup()
    elif sys.argv[1]=='check':main()
    else:raise ValueError('setup or check')
