"""Restricted AST reading, never import/exec/eval upstream code or instantiate objects."""
import ast,copy,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
CONF=ROOT/'sources/Domain-RAG/mmdetection/configs'
MAIN=CONF/'grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

def merge(base,child):
 if not isinstance(base,dict) or not isinstance(child,dict): return copy.deepcopy(child)
 if child.get('_delete_',False): return {k:copy.deepcopy(v) for k,v in child.items() if k!='_delete_'}
 out=copy.deepcopy(base)
 for k,v in child.items(): out[k]=merge(out.get(k),v)
 return out

def read_expr(n,env,base):
 if isinstance(n,ast.Constant): return n.value
 if isinstance(n,ast.List): return [read_expr(x,env,base) for x in n.elts]
 if isinstance(n,ast.Tuple): return tuple(read_expr(x,env,base) for x in n.elts)
 if isinstance(n,ast.Dict): return {read_expr(k,env,base):read_expr(v,env,base) for k,v in zip(n.keys,n.values)}
 if isinstance(n,ast.Name): return env[n.id]
 if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='_base_': return copy.deepcopy(base[n.attr])
 if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add): return read_expr(n.left,env,base)+read_expr(n.right,env,base)
 if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub): return -read_expr(n.operand,env,base)
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
  if n.func.id=='dict' and not n.args and all(k.arg is not None for k in n.keywords): return {k.arg:read_expr(k.value,env,base) for k in n.keywords}
  if n.func.id=='len' and len(n.args)==1 and not n.keywords: return len(read_expr(n.args[0],env,base))
 raise ValueError('Unsupported AST: '+ast.dump(n))

def resolve(path,fish_shot=None,stack=()):
 path=path.resolve()
 if CONF not in path.parents or path in stack: raise ValueError('Bad dependency')
 tree=ast.parse(path.read_text());bases=[]
 for n in tree.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_base_': bases=ast.literal_eval(n.value)
 if isinstance(bases,str): bases=[bases]
 if path==MAIN and fish_shot is not None: bases[0]=f'CDFSOD_detection_few-shot_FISH_{fish_shot}shot.py'
 base={}; deps=[]
 for x in bases:
  resolved,subdeps=resolve(path.parent/x,fish_shot,stack+(path,)); base=merge(base,resolved);deps+=subdeps
 env={}
 for n in tree.body:
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str): continue
  if not (isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)): raise ValueError('Unsupported statement in '+str(path))
  key=n.targets[0].id
  if key=='_base_': continue
  env[key]=read_expr(n.value,env,base)
 return merge(base,env),deps+[str(path.relative_to(ROOT))]

results={}
for name,path,shot in [('as_committed_swin_t',MAIN,None)]+[(f'static_FISH_{k}shot_swin_t',MAIN,k) for k in [1,5,10]]+[(f'automation_swin_b_if_FISH_{k}shot',CONF/'grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py',k) for k in [1,5,10]]:
 cfg,deps=resolve(path,shot)
 if name.startswith('automation_'): cfg=merge(cfg,{'train_cfg':{'max_epochs':30}})
 results[name]={'kind':'static_resolution_not_executed','hypothetical_base_substitution':shot,'dependencies':deps,'config':cfg}
 r=cfg
 print(name,'root',r['data_root'],'ann',r['train_dataloader']['dataset']['ann_file'],'epochs',r['train_cfg']['max_epochs'],'batch',r['train_dataloader']['batch_size'],'backbone',r['model']['backbone']['embed_dims'],'pipeline',[t['type'] for t in r['train_dataloader']['dataset']['pipeline']])
(ROOT/'reports/static_resolved_configs.json').write_text(json.dumps(results,indent=2))
