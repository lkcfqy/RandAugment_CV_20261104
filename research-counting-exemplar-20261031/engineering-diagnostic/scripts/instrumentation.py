"""Read-only snapshots/hooks; never changes model values, ordering, or decisions."""
import hashlib
import json
import torch


def snapshot(tensor):
    return tensor.detach().cpu().clone()


def tensor_hash(tensor):
    value=tensor.detach().cpu().contiguous()
    meta=json.dumps({'shape':list(value.shape),'dtype':str(value.dtype)},sort_keys=True).encode()
    return hashlib.sha256(meta+b'\0'+value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_stage(reference, actual):
    actual=actual.detach().cpu()
    if reference.shape!=actual.shape or reference.dtype!=actual.dtype:
        return {'same_shape_dtype':False,'within_tolerance':False}
    if not reference.is_floating_point():
        exact=torch.equal(reference,actual)
        return {'same_shape_dtype':True,'exact':exact,'within_tolerance':exact,'max_absolute_error':0 if exact else None}
    finite=torch.isfinite(reference)&torch.isfinite(actual)
    special=((torch.isneginf(reference)&torch.isneginf(actual))|(torch.isposinf(reference)&torch.isposinf(actual)))
    pattern=bool((finite|special).all())
    delta=(actual[finite]-reference[finite]).abs()
    limit=1e-4+1e-3*reference[finite].abs()
    maxerr=float(delta.max()) if delta.numel() else 0.
    ratio=float((delta/limit).max()) if delta.numel() else 0.
    return {'same_shape_dtype':True,'exact':torch.equal(reference,actual),'nonfinite_pattern_compatible':pattern,
            'within_tolerance':pattern and bool((delta<=limit).all()),'max_absolute_error':maxerr,
            'max_error_divided_by_tolerance':ratio}


def topk_report(cpu_scores,cpu_indices,other_scores,other_indices):
    a=cpu_scores.reshape(-1);b=other_scores.reshape(-1)
    ai=cpu_indices.reshape(-1);bi=other_indices.reshape(-1)
    if a.shape!=b.shape or ai.shape!=bi.shape:raise ValueError('top-k shape mismatch')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite top-k input')
    def describe(scores,indices):
        selected=scores[indices];remaining=torch.ones(scores.numel(),dtype=torch.bool);remaining[indices]=False
        cutoff=float(selected.min());best_excluded=float(scores[remaining].max())
        gaps=selected[:-1]-selected[1:]
        return {'actual_indices_in_native_order':indices.tolist(),'candidate_size':scores.numel(),'selected_size':indices.numel(),
                'cutoff_score':cutoff,'best_excluded_score':best_excluded,'cutoff_gap_to_best_excluded':cutoff-best_excluded,
                'ties_at_cutoff':int((scores==cutoff).sum()),'zero_adjacent_selected_gaps':int((gaps==0).sum()),
                'minimum_adjacent_selected_gap':float(gaps.min()),'score_tensor_sha256':tensor_hash(scores)}
    aset=set(ai.tolist());bset=set(bi.tolist())
    return {'cpu':describe(a,ai),'mps':describe(b,bi),'order_equal':torch.equal(ai,bi),
            'sets_equal':aset==bset,'intersection_size':len(aset&bset),'symmetric_difference_size':len(aset^bset),
            'different_positions':int((ai!=bi).sum()),'max_absolute_score_delta':float((a-b).abs().max()),
            'encoder_score_comparison':compare_stage(a,b),'posthoc_reordered_outputs':False}


class Recorder:
    def __init__(self,model,grounding_module,context,report,save_callback):
        self.model=model;self.module=grounding_module;self.context=context;self.report=report;self.save=save_callback
        self.cpu={};self.cpu_hashes={};self.input_cpu={};self.input_hashes={'cpu':{},'mps':{}};self.topk={};self.handles=[]
        self.calls={};self.original_topk=torch.topk;self.original_roi=grounding_module.roi_align
        report['diagnostic']={'stage_comparisons':[],'first_nonzero_stage':None,'first_tolerance_failure_stage':None,
                              'embed_init_tgt':bool(model.transformer.embed_init_tgt),'output_reordering_used':False}

    def stage(self,name,tensor):
        value=snapshot(tensor);device=self.context['device']
        if device=='cpu':
            if name in self.cpu:raise AssertionError('duplicate CPU diagnostic stage '+name)
            self.cpu[name]=value;self.cpu_hashes[name]=tensor_hash(value)
        else:
            comparison=compare_stage(self.cpu[name],value)
            row={'stage':name,'shape':list(value.shape),'cpu_sha256':self.cpu_hashes[name],
                 'mps_sha256':tensor_hash(value),**comparison}
            d=self.report['diagnostic'];d['stage_comparisons'].append(row)
            if not comparison.get('exact',False) and d['first_nonzero_stage'] is None:d['first_nonzero_stage']=name
            if not comparison['within_tolerance'] and d['first_tolerance_failure_stage'] is None:d['first_tolerance_failure_stage']=name
        self.save()

    def exact_input(self,name,tensor):
        sha=tensor_hash(tensor);device=self.context['device'];self.input_hashes[device][name]=sha
        if device=='cpu':self.input_cpu[name]=sha
        else:
            if sha!=self.input_cpu[name]:raise AssertionError('CPU/MPS exact input hash mismatch: '+name)
        self.report['diagnostic']['exact_input_hashes']=self.input_hashes
        self.save()

    def install(self):
        def bert_before(module,args,kwargs):
            for key,value in kwargs.items():
                if isinstance(value,torch.Tensor):self.exact_input('bert_input.'+key,value)
        self.handles.append(self.model.bert.register_forward_pre_hook(bert_before,with_kwargs=True))
        self.handles.append(self.model.bert.register_forward_hook(lambda m,a,o:self.stage('bert.last_hidden_state',o['last_hidden_state'])))
        self.handles.append(self.model.feat_map.register_forward_hook(lambda m,a,o:self.stage('bert.projected_text',o)))
        def backbone(module,args,output):
            device=self.context['device'];key=(device,'backbone');index=self.calls.get(key,0);self.calls[key]=index+1
            role=('query','positive','negative')[index]
            for i,feature in enumerate(output[0]):
                self.stage(f'backbone.{role}.level{i}',feature.tensors)
                self.exact_input(f'backbone_mask.{role}.level{i}',feature.mask)
        self.handles.append(self.model.backbone.register_forward_hook(backbone))
        def roi(*args,**kwargs):
            device=self.context['device'];key=(device,'roi');index=self.calls.get(key,0);self.calls[key]=index+1
            role=('positive','negative')[index]
            self.stage('roi.'+role+'.input_features',args[0])
            for i,value in enumerate(kwargs['boxes']):self.exact_input('roi.'+role+'.xyxy'+str(i),value)
            output=self.original_roi(*args,**kwargs)
            self.stage('roi.'+role+'.output_tokens',output)
            return output
        self.module.roi_align=roi
        def encoder_before(module,args,kwargs):
            self.stage('encoder.input_visual',args[0]);self.stage('encoder.input_text',kwargs['memory_text'])
            for key in ['pos','valid_ratios']:self.stage('encoder.'+key,kwargs[key])
            for key in ['level_start_index','spatial_shapes','key_padding_mask','text_attention_mask','position_ids','text_self_attention_masks']:
                self.exact_input('encoder.'+key,kwargs[key])
        self.handles.append(self.model.transformer.encoder.register_forward_pre_hook(encoder_before,with_kwargs=True))
        def encoder_after(module,args,output):
            self.stage('encoder.output_visual',output[0]);self.stage('encoder.output_text',output[1])
        self.handles.append(self.model.transformer.encoder.register_forward_hook(encoder_after))
        self.handles.append(self.model.transformer.enc_output_norm.register_forward_hook(lambda m,a,o:self.stage('proposal.projected_memory',o)))
        def proposal_scores(module,args,output):
            device=self.context['device'];key=(device,'proposal_scores');index=self.calls.get(key,0);self.calls[key]=index+1
            if index==0:self.stage('proposal.token_logits_before_topk',output)
        self.handles.append(self.model.transformer.enc_out_class_embed.register_forward_hook(proposal_scores))
        def topk(input,k,*args,**kwargs):
            output=self.original_topk(input,k,*args,**kwargs)
            if k==900 and input.ndim==2 and input.shape[0]==1:
                device=self.context['device']
                if device in self.topk:raise AssertionError('unexpected additional proposal top-k')
                self.topk[device]=(snapshot(input),snapshot(output.indices))
                self.stage('proposal.score_max_before_topk',input)
                if device=='mps':
                    self.report['diagnostic']['topk']=topk_report(*self.topk['cpu'],*self.topk['mps']);self.save()
            return output
        torch.topk=topk

    def finalize(self):
        unchanged=all(tensor_hash(v)==self.cpu_hashes[k] for k,v in self.cpu.items())
        self.report['diagnostic']['cpu_stage_clones_unchanged']=unchanged
        self.report['diagnostic']['all_input_key_sets_equal']=set(self.input_hashes['cpu'])==set(self.input_hashes['mps'])
        self.report['diagnostic']['stage_count_cpu']=len(self.cpu)
        self.report['diagnostic']['stage_count_mps']=len(self.report['diagnostic']['stage_comparisons'])
        assert unchanged
        self.save()

    def close(self):
        torch.topk=self.original_topk;self.module.roi_align=self.original_roi
        for handle in self.handles:handle.remove()
        self.handles.clear();self.model=None;self.cpu.clear();self.topk.clear()
