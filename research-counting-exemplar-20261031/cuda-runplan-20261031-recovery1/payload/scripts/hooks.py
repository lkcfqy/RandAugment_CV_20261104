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
    return {'cpu':describe(a,ai),'cuda':describe(b,bi),'order_equal':torch.equal(ai,bi),
            'sets_equal':aset==bset,'intersection_size':len(aset&bset),'symmetric_difference_size':len(aset^bset),
            'different_positions':int((ai!=bi).sum()),'max_absolute_score_delta':float((a-b).abs().max()),
            'encoder_score_comparison':compare_stage(a,b),'posthoc_reordered_outputs':False}

def token_contract(model,positive,negative):
    from models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
    tokens=model.tokenizer([positive+' . '+negative+' . '],padding='longest',return_tensors='pt')
    mask,pos,_=generate_masks_with_special_tokens_and_transfer_map(tokens,model.specical_tokens,None)
    td={'encoded_text':torch.zeros(1,tokens['input_ids'].shape[1],256),'text_token_mask':tokens['attention_mask'].bool(),'position_ids':pos,'text_self_attention_masks':mask}
    td,tokens=model.add_exemplar_tokens(tokens,td,torch.full((1,3,256),11.),[torch.tensor([0])])
    td,tokens=model.add_exemplar_tokens(tokens,td,torch.full((1,3,256),22.),[torch.tensor([1])])
    ids=tokens['input_ids'][0].tolist();one=ids.index(1012);two=ids.index(1012,one+1)
    a=[i for i in range(one) if ids[i]==1008];b=[i for i in range(one+1,two) if ids[i]==1008]
    if len(a)!=3 or len(b)!=3 or ids.count(1008)!=6 or not (td['encoded_text'][0,a]==11).all() or not (td['encoded_text'][0,b]==22).all():raise AssertionError('visual group insertion failed')
    if td['text_self_attention_masks'][0,a][:,b].any() or td['text_self_attention_masks'][0,b][:,a].any():raise AssertionError('visual concept groups mixed')
    if not positive and not negative and ids!=[101,1008,1008,1008,1012,1008,1008,1008,1012,102]:raise AssertionError('V token order changed')
    return ids

class NativeHooks:
    def __init__(self,model,context,counters):
        self.context=context;self.counters=counters;self.raw={};self.inputs={};self.topk={};self.original_topk=torch.topk
        self.handles=[model.register_forward_pre_hook(self.before,with_kwargs=True),model.register_forward_hook(self.after),model.bert.register_forward_pre_hook(self.bert_before,with_kwargs=True)]
        def topk(input,k,*args,**kwargs):
            output=self.original_topk(input,k,*args,**kwargs)
            if self.context.get('phase')=='precheck' and k==900 and input.ndim==2 and input.shape[0]==1:
                self.topk={'values':snapshot(input),'indices':snapshot(output.indices)}
            return output
        torch.topk=topk
    def remember(self,name,tensor):
        self.inputs[name]=tensor_hash(tensor)
    def before(self,module,args,kwargs):
        self.counters['root_forward_entered']+=1
        if self.counters['root_forward_entered']!=self.counters['root_forward_reserved']:raise AssertionError('unreserved model forward')
        if len(args[3])!=1 or len(args[4])!=1 or len(args[4][0])!=1:raise AssertionError('missing negative visual group')
        shape=tuple(args[0].tensors.shape);self.context['actual_shape']=list(shape)
        if self.context['phase']=='precheck' and shape!=(1,3,800,1333):raise AssertionError('fixture preprocessing not worst-shape size')
        for name,nested in [('query',args[0]),('positive',args[1]),('negative',args[3][0])]:
            if tuple(nested.tensors.shape)!=shape or nested.tensors.dtype!=torch.float32 or nested.tensors.device.type!=self.context['device']:raise AssertionError('input dtype/shape/device mismatch')
            self.remember('normalized_'+name,nested.tensors);self.remember('padding_'+name,nested.mask)
        if len({self.inputs['normalized_'+name] for name in ['query','positive','negative']})!=1:raise AssertionError('exemplar images differ from target image')
        width,height=self.context['original_wh'];scales=torch.tensor([shape[-1]/width,shape[-2]/height,shape[-1]/width,shape[-2]/height])
        if not torch.equal(args[2][0].cpu(),torch.tensor(self.context['positive'],dtype=torch.float32)*scales):raise AssertionError('positive boxes changed')
        if not torch.equal(args[4][0][0].cpu(),torch.tensor(self.context['negative'],dtype=torch.float32)*scales):raise AssertionError('negative boxes changed')
        self.remember('positive_xyxy_resized',args[2][0]);self.remember('negative_xyxy_resized',args[4][0][0])
        if kwargs['captions']!=[self.context['caption']]:raise AssertionError('caption changed')
    def bert_before(self,module,args,kwargs):
        for key,value in kwargs.items():
            if isinstance(value,torch.Tensor):self.remember('bert_'+key,value)
    def after(self,module,args,output):
        self.counters['root_forward_returned']+=1
        self.raw={key:snapshot(output[key]) for key in ['pred_logits','pred_boxes','input_ids','text_mask']}
        if self.raw['input_ids'][0].tolist()!=self.context['expected_ids']:raise AssertionError('native positive/negative token groups changed')
        if self.raw['pred_logits'].dtype!=torch.float32 or self.raw['pred_boxes'].dtype!=torch.float32:raise AssertionError('not FP32')
        ids=output['input_ids'][0].tolist();split=ids.index(1012);scores=output['pred_logits'].sigmoid()[0]
        pos=scores[:,:split+1].max(-1).values;neg=scores[:,split+1:].max(-1).values
        self.raw['native_filter_mask']=snapshot((pos>.23)&(pos>neg))
        self.remember('native_token_ids',output['input_ids']);self.remember('native_text_mask',output['text_mask'])
    def reset(self):
        self.raw={};self.inputs={};self.topk={}
    def close(self):
        torch.topk=self.original_topk
        for h in self.handles:h.remove()
        self.handles.clear();self.reset()

