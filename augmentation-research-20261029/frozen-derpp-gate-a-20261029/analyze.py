"""Frozen analysis. Runs no model and evaluates no development/test examples."""
import csv,datetime,json,time
from pathlib import Path
import numpy as np
import torch
from audit_math import scores,assign,invariants,probability,tv,spearman,cosine,rank_pairs,ols_r2,rank
from prepare import sha,savej,state_fingerprints

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'mammoth-derpp-baseline-20261029'
OUT=ROOT/'run-001'

def aggregate(p,rows):
    unique,inv=np.unique(rows,return_inverse=True)
    return np.bincount(inv,weights=p,minlength=len(unique))

def consistency(pair,coverage):
    delta=[probability(x['B'])-probability(x['A']) for x in pair]
    sa=spearman(pair[0]['Davg'],pair[1]['Davg']);sb=spearman(pair[0]['Dmean'],pair[1]['Dmean'])
    co=cosine(delta[0],delta[1]);tvs=[tv(x['A'],x['B']) for x in pair]
    return {'TVs':tvs,'Davg_cross_group_spearman':sa,'Dmean_cross_group_spearman':sb,
            'signed_identity_difference_cosine':co,
            'passed':bool(coverage>=.8 and min(tvs)>=.05 and sa>=.8 and sb>=.8 and co>=.5)}

def main():
    started=time.perf_counter();freeze=json.loads((ROOT/'protocol-freeze.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in freeze['files'].items())
    c=torch.load(BASE/'run-002/task2-final.pt',map_location='cpu',weights_only=False)
    assert state_fingerprints(c)==json.loads((ROOT/'input-state-fingerprints.json').read_text())
    inf=json.loads((OUT/'inference-summary.json').read_text());assert inf['status']=='completed_frozen_inference'
    cal=np.load(OUT/'shadow-calibration.npz',allow_pickle=False);w=cal['weights'];q=float(cal['q'])
    h=np.load(OUT/'score-logits.npy',allow_pickle=False).astype(np.float64)
    clean=np.load(OUT/'clean-logits.npy',allow_pickle=False).astype(np.float64)
    z=c['buffer']['logits'].numpy().astype(np.float64);labels=c['buffer']['labels'].numpy()
    rows=c['buffer']['rows'];arrival=c['buffer']['arrivals'];age=199999-arrival
    exp=np.exp(clean-clean.max(-1,keepdims=True));prob=exp/exp.sum(-1,keepdims=True)
    confidence=prob[np.arange(500),labels];entropy=-(prob*np.log(prob)).sum(-1)
    agebin=np.searchsorted([50000,100000,150000],age,side='right')
    strata=labels*8+agebin*2+(confidence>=.5).astype(int)
    sizes={int(s):int((strata==s).sum()) for s in np.unique(strata)}
    coverage=sum(n for n in sizes.values() if n>=2)/500
    clean_norm=np.linalg.norm(clean,axis=-1);z_norm=np.linalg.norm(z,axis=-1)
    onehot=np.column_stack([strata==s for s in np.unique(strata)]).astype(float)
    all_modes={'raw':[],'centered':[],'centered_unit_norm':[]};groups=[];arrays=[]
    for group in range(2):
        hg=h[group];mean=hg.mean(0);hc=hg-hg.mean(-1,keepdims=True);zc=z-z.mean(-1,keepdims=True)
        norm_hc=np.linalg.norm(hc,axis=-1,keepdims=True);norm_zc=np.linalg.norm(zc,axis=-1,keepdims=True)
        hu=hc/np.maximum(norm_hc,1e-12);zu=zc/np.maximum(norm_zc,1e-12)
        mode_result={};mode_arrays={}
        for mode,mh,mz in [('raw',hg,z),('centered',hc,zc),('centered_unit_norm',hu,zu)]:
            da,dm,var=scores(mh,mz);error=np.abs(da-dm-var)
            assert float(error.max())<1e-10*(1+float(da.max()))
            A=assign(w,da,strata,rows,arrival);B=assign(w,dm,strata,rows,arrival)
            inv=invariants(w,{'A':A,'B':B},strata)
            data={'Davg':da,'Dmean':dm,'V':var,'A':A,'B':B};all_modes[mode].append(data);mode_arrays[mode]=data
            pa,pb=probability(A),probability(B)
            oa=np.argsort(A,kind='stable')[:100];ob=np.argsort(B,kind='stable')[:100]
            mode_result[mode]={'TV_A_B':tv(A,B),'source_aggregated_TV':float(np.abs(aggregate(pa,rows)-aggregate(pb,rows)).sum()/2),
                'Davg_Dmean_spearman':spearman(da,dm),'within_stratum_rank_pairs':rank_pairs(da,dm,strata),
                'bottom20_weight_identity_overlap':len(np.intersect1d(oa,ob))/100,
                'identity_max_abs_error':float(error.max()),'mean_Davg':float(da.mean()),'mean_Dmean':float(dm.mean()),
                'mean_half_view_variance':float(var.mean()),'variance_fraction_of_Davg':float(var.sum()/max(da.sum(),1e-30)),
                'weight_invariants':inv}
        da=mode_arrays['raw']['Davg'];dc=mode_arrays['centered']['Davg'];v=mode_arrays['raw']['V'];vc=mode_arrays['centered']['V']
        translation=.5*np.square(hg.mean(-1)-z.mean(-1)).mean(0)
        translation_var=.5*np.square(hg.mean(-1)-mean.mean(-1)).mean(0)
        assert np.max(np.abs(da-dc-translation))<1e-10*(1+float(da.max()))
        assert np.max(np.abs(v-vc-translation_var))<1e-10*(1+float(v.max()))
        h_norm=np.linalg.norm(hg,axis=-1);mean_norm=np.linalg.norm(mean,axis=-1);norm_sd=h_norm.std(0)
        radial=.5*np.square(h_norm-z_norm).mean(0)/10
        delta=probability(mode_arrays['raw']['B'])-probability(mode_arrays['raw']['A'])
        covnames=['clean_norm','historical_norm','mean_prediction_norm','view_norm_sd','clean_true_confidence','clean_entropy','mean_prediction_logit_mean','age']
        cov=np.column_stack([clean_norm,z_norm,mean_norm,norm_sd,confidence,entropy,mean.mean(-1),age/200000])
        cov=(cov-cov.mean(0))/np.maximum(cov.std(0),1e-12)
        controls={'translation_residual_fraction':float(translation.sum()/max(da.sum(),1e-30)),
                  'translation_variance_fraction':float(translation_var.sum()/max(v.sum(),1e-30)),
                  'radial_norm_residual_fraction':float(radial.sum()/max(da.sum(),1e-30)),
                  'unit_normalization_zero_h_vectors':int((norm_hc<1e-12).sum()),
                  'unit_normalization_zero_z_vectors':int((norm_zc<1e-12).sum()),
                  'identity_difference_OLS_R2_strata_only':ols_r2(delta,onehot),
                  'identity_difference_OLS_R2_strata_and_covariates':ols_r2(delta,np.column_stack([onehot,cov])),
                  'covariates':{name:{'spearman_with_variance':spearman(cov[:,j],v),
                        'spearman_with_identity_probability_difference':spearman(cov[:,j],delta)} for j,name in enumerate(covnames)}}
        for layer,s in [('unstratified',np.zeros(500,int)),('class_only',labels)]:
            aa=assign(w,mode_arrays['raw']['Davg'],s,rows,arrival);bb=assign(w,mode_arrays['raw']['Dmean'],s,rows,arrival)
            controls[layer+'_TV_descriptive_only']=tv(aa,bb)
        arrays.append({'translation_component':translation,'translation_variance':translation_var,'radial_norm_residual':radial,**{
            mode+'_'+k:value for mode,di in mode_arrays.items() for k,value in di.items()}})
        groups.append({'group':group+1,'modes':mode_result,'controls':controls})
    consistencies={mode:consistency(pair,coverage) for mode,pair in all_modes.items()}
    passed=consistencies['raw']['passed']
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'status':'gate_A_passed_draft_only' if passed else 'gate_A_not_met_close_reference_route',
      'raw_gate_passed':passed,'checkpoint_global_step':6300,'buffer_slots':500,'unique_sources':len(np.unique(rows)),
      'stratum_sizes':sizes,'exchangeable_fraction':coverage,'singleton_slots':sum(n for n in sizes.values() if n==1),
      'weight_origin':'160 frozen-model auxiliary shadow batches, not native A2ER or baseline training trajectory',
      'Q_after_shadow_frozen':q,'shadow_visits_min_max':[int(cal['visits'].min()),int(cal['visits'].max())],
      'shadow_unvisited_count':int((cal['visits']==0).sum()),'groups':groups,
      'cross_group_consistency':consistencies,'analysis_seconds':time.perf_counter()-started,
      'optimizer_steps':0,'branches_executed':0,'variance_identity_is_known_math':True,
      'no_contribution_or_generalization_claim':True}
    np.savez_compressed(OUT/'scored-arrays.npz',rows=rows,labels=labels,arrivals=arrival,age=age,age_bin=agebin,
          confidence=confidence,entropy=entropy,strata=strata,N_shadow=w,
          **{f'g{i+1}_{n}':a for i,di in enumerate(arrays) for n,a in di.items()})
    fields=['slot','source_row','label','arrival','age','age_bin','clean_confidence','stratum','N_shadow']+[
       f'g{g+1}_{n}' for g in range(2) for n in ['Davg','Dmean','V','rank_Davg','rank_Dmean','A','B']]
    with (OUT/'all-slots.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i in range(500):
            row=dict(slot=i,source_row=int(rows[i]),label=int(labels[i]),arrival=int(arrival[i]),age=int(age[i]),
                 age_bin=int(agebin[i]),clean_confidence=float(confidence[i]),stratum=int(strata[i]),N_shadow=float(w[i]))
            for g in range(2):
                raw=all_modes['raw'][g]
                for n in ['Davg','Dmean','V','A','B']:row[f'g{g+1}_{n}']=float(raw[n][i])
                for n in ['Davg','Dmean']:row[f'g{g+1}_rank_{n}']=float(rank(raw[n])[i])
            writer.writerow(row)
    savej(OUT/'gate-A.json',result)
    print(json.dumps({'status':result['status'],'exchangeable_fraction':coverage,'cross_group_consistency':consistencies},indent=2))
    make_report(result,inf)

def make_report(r,inf):
    passed=r['raw_gate_passed']
    decision='通过预设投入门；只拟定分支草案，仍未训练' if passed else '未通过预设投入门；关闭本检查点参考组合的短分支路线'
    rows='\n'.join(f"| {name} | {s['TVs'][0]:.6f} | {s['TVs'][1]:.6f} | {s['Davg_cross_group_spearman']:.4f} | {s['Dmean_cross_group_spearman']:.4f} | {s['signed_identity_difference_cosine']:.4f} |" for name,s in r['cross_group_consistency'].items())
    proc=json.loads((ROOT/'process-timing.json').read_text())
    text=f'''# 新基础检查点门 A 诊断

**{decision}。** 仅一个预先固定checkpoint/seed/8+8普通增强组，没有挑样本、改强度或增补视图。旧760步结果保持基础学习不足；这次独立新协议也不是A2ER复现。

| 评分定义 | 组1 TV(A,B) | 组2 TV(A,B) | 跨组Davg Spearman | 跨组Dmean Spearman | 身份概率差余弦 |
|---|---:|---:|---:|---:|---:|
{rows}

主门预先固定为两组原始TV均≥.05、Davg/Dmean跨组Spearman均≥.80、概率差余弦≥.50、可交换层覆盖≥80%。实际覆盖 **{r['exchangeable_fraction']:.1%}**，单例{r['singleton_slots']}槽。raw门 **{passed}**。中心化和单位范数结果是预定解释诊断，不用于救原始门失败。

使用当前DER++第6300步最终检查点全部500槽位，实际472个source；含10类，历史logit/入库年龄来自原reservoir。分层为真实class×预定四段年龄×clean真类confidence≥.5。A/B在每层保持完整权重多重集，故各层概率质量、entropy、ESS一致。TV是归一化基础权重分布距离，不是无放回batch边际纳入概率；重复source聚合TV和所有槽位也已保存。

基线本身为uniform replay，**没有N降权或Q轨迹**。本协议结果前声明160批×32的冻结模型影子校准，按同一残差重权公式生成N_shadow；Q只按实际校准batch的median EMA更新，后续全buffer评分绝不更新Q。影子访问范围{r['shadow_visits_min_max']}，未访问{r['shadow_unvisited_count']}槽，最终Q={r['Q_after_shadow_frozen']:.8g}。影子校准不更新模型/BN/历史target/buffer，也不能当作作者方法训练轨迹。没有假设免费原生权重。

Davg和Dmean均为half-MEAN，原公式half-SUM的eps同步从1e-5除10为1e-6；这保持采样比例的单位等价。每个slot都验证Davg-Dmean=半视图方差；它是已知恒等式，无创新性。逐槽原始logit、评分、排名、权重、age和分层在 run-001/all-slots.csv、scored-arrays.npz、score-logits.npy，未删失败样本。

'''
    for g in r['groups']:
        rawg=g['modes']['raw'];con=g['controls']
        text+=f"组{g['group']}：恒等式最大误差{rawg['identity_max_abs_error']:.3g}；层内严格排序反转率{rawg['within_stratum_rank_pairs']['discordant_fraction']:.2%}；source聚合TV={rawg['source_aggregated_TV']:.6f}。公共logit平移占Davg {con['translation_residual_fraction']:.2%}、占视图方差 {con['translation_variance_fraction']:.2%}；径向范数差占Davg {con['radial_norm_residual_fraction']:.2%}。层+预定范数/置信度/entropy/年龄等协变量对身份差的in-sample OLS R²={con['identity_difference_OLS_R2_strata_and_covariates']:.4f}，只能描述解释关联，不是因果或泛化证据。\n\n"
    text+=f'''推理完整子进程 **{proc['whole_child_seconds']:.3f}秒**，影子校准{inf['shadow_calibration_seconds']:.3f}秒、8+8评分{inf['score_seconds']:.3f}秒，均同步计时。GPU推理500 clean+5120影子+8000评分=13,620图像；额外8张CPU数值核对，最大差{inf['cpu_mps_first8_max_abs']:.3g}。optimizer/backward均为0。进程peak RSS {inf['memory']['process_peak_RSS_bytes']:,}字节，MPS driver {inf['memory']['mps_driver_bytes']:,}字节，统一内存不相加。

运行前后全部模型/BN tensor与buffer内容逐位一致，原checkpoint及源码hash不变。未读取dev/test；没有训练分支、云GPU、安装、付费、推送或重置卡操作。

边界：即使raw门通过，只说明本单seed、冻结终点、声明影子权重组合具有可重复的身份分配差异；不说明原生A2ER受增强损害，不说明clean泛化损害，也不说明有原创贡献。若门未过，是停止这个指定组合的投入决定，不是证明所有模型、检查点或机制普遍无效。
'''
    with (ROOT/'REPORT_ZH.md').open('x') as f:f.write(text)
    if passed:
        draft='''# 五分支草案（仅草案，未执行、尚需单独冻结）

以门A固定第6300步checkpoint作为共同起点，只用第一组评分。N_shadow、A、B、固定seed层内随机R保留完整层内权重多重集，U为uniform。N_shadow是160批冻结影子流，不可标为A2ER原生身份。先由父任务结合中心化/范数/置信度诊断确认是否值得投入。

如果另获执行协议，最多各64步；固定buffer/targets/alpha=.3/beta=.5，关闭永久target correction和自适应，仅改变回放基础权重身份。同样model/optimizer/BN初态、当前新任务序列、batch32和两次replay32，所有臂同等诊断预算，使用共同随机数耦合；实际source曝光可不同，逐步报告。沿用checkpoint任务2末LR=.0012是否足以检验需在结果前由父任务确定，不能事后升LR追结果。BN三次forward顺序一致，并明确训练中BN状态可能随身份产生路径差异，不能声称完全锁定BN。

主终点仍为buffer外原TRAIN旧/新dev NLL与错误率。B相对N_shadow/A/R各旧NLL改善≥.01 nat、新NLL恶化≤.01，且U不能同样好或更好；单seed只作机制初筛。当前buffer含新旧两类，而旧协议只含旧类，这个适配须写入最终冻结协议。不要直接复用旧方案声称同一试验。当前未运行任何分支。
'''
        with (ROOT/'FIVE_BRANCH_DRAFT_ZH.md').open('x') as f:f.write(draft)

if __name__=='__main__':main()
