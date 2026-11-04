"""Foreground execution with durable commits; scientific operations from frozen AST."""
import ast,collections,hashlib,json,math,os,pickle,random,sys,tarfile,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from state_io import atomic_json,canonical_hash,cpu_tree,save_checkpoint,sha_file,validate_checkpoint

class Engine:
    def __init__(self,config):
        self.c=config;self.root=Path('/content/tta_attempt02');self.out=self.root/'output';self.out.mkdir(exist_ok=True)
        self.audit={'benchmark_steps':0,'benchmark_presentations':0,'formal_steps':0,'formal_presentations':0,'inference_views':0}
        self.phase='prepared';self.step=0;self.epoch=0;self.cursor=0;self.presentations=0;self.perm=None
        source=self.root/'science_v1.py';assert sha_file(source)==config['science_source_sha256']
        assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
        env={'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'numpy':np.__version__,
             'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'python':sys.version.split()[0]}
        assert env['torch']=='2.11.0+cu130' and env['numpy']=='2.1.3' and env['cuda']=='13.0' and env['cudnn']==92700, 'Environment differs from frozen attempt1 execution'
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.identity={'run_id':'attempt02_designated','protocol_sha256':config['protocol_sha256'],
                       'amendment_sha256':config['amendment_sha256'],'science_source_sha256':config['science_source_sha256'],
                       'engine_sha256':sha_file(Path(__file__)),'state_io_sha256':sha_file(self.root/'state_io.py'),
                       'dataset_sha256':config['dataset_sha256'],'split_manifest_sha256':config['split_manifest_sha256'],
                       'seed':0,'environment':env}
        atomic_json(self.out/'environment.json',self.identity)
        self.ns={'__file__':str(source),'np':np,'torch':torch,'nn':nn,'F':F,'random':random,'math':math,
                 'collections':collections,'hashlib':hashlib,'json':json,'os':os,'pickle':pickle,'tarfile':tarfile,
                 'time':time,'urllib':urllib,'Path':Path,'ROOT':self.root,'OUT':self.out,'SOURCE_START':time.time(),
                 'event':self.event,'save':lambda name,data:atomic_json(self.out/name,data),
                 'sha':sha_file,'watchdog':self.watchdog,'STATE':self.audit}
        tree=ast.parse(source.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='data_and_split']
        selected += [n for n in run.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in
                     {'Block','ResNet20','seed','augment','model_optim','update','infer'}]
        assert len(selected)==8
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),self.ns)
        self.event('preparing_data',identity=self.identity)
        xcpu,ycpu,roles=self.ns['data_and_split'](np)
        assert sha_file(self.out/'split-manifest.jsonl')==config['split_manifest_sha256']
        summary=json.loads((self.out/'split-summary.json').read_text())
        assert summary['archive_sha256']==config['dataset_sha256']
        for role,expected in config['role_id_hashes'].items():assert summary['roles'][role]['ids_sha256']==expected
        self.ns.update(xcpu=xcpu,ycpu=ycpu,roles=roles)
        # Execute the unchanged original normalization / transfer statement block.
        start=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t0' for t in n.targets))
        end=next(i for i,n in enumerate(run.body) if isinstance(n,ast.FunctionDef) and n.name=='augment')
        normnodes=run.body[start:end]
        exec(compile(ast.Module(body=normnodes,type_ignores=[]),str(source),'exec'),self.ns)
        self.normalization={'mean':self.ns['mean'].detach().cpu(),'std':self.ns['std'].detach().cpu()}
        self.held_ids=np.concatenate([roles[k] for k in ('pseudocal','validation','cal','eval')])
        self.held_roles=np.concatenate([np.full(len(roles[k]),i,dtype=np.int8) for i,k in enumerate(('pseudocal','validation','cal','eval'))])
        np.save(self.out/'heldout-image-ids.npy',self.held_ids);np.save(self.out/'heldout-labels.npy',ycpu[self.held_ids])
        np.save(self.out/'view-ids.npy',np.arange(10,dtype=np.int64))
        self.watchdog();self.done('prepare',{'identity':self.identity,'normalization':json.loads((self.out/'normalization.json').read_text()),
                 'scientific_AST_sha256':hashlib.sha256(ast.dump(ast.Module(body=selected+normnodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()})

    def event(self,event,**values):
        obj={'event':event,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'attempt_seconds':time.time()-self.c['allocated_at'],'previous_GPU_seconds':self.c['previous_GPU_seconds'],**values}
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
        print(json.dumps(obj,allow_nan=False),flush=True)

    def watchdog(self):
        assert self.c['previous_GPU_seconds']+time.time()-self.c['allocated_at']<6600,'Cumulative computation deadline110min'
        assert time.time()<self.c['wall_deadline']-3600,'Reserve60min CPU statistics before original6h deadline'
        assert self.audit['benchmark_steps']<=200 and self.audit['benchmark_presentations']<=25600
        assert self.audit['formal_steps']<=25040 and self.audit['formal_presentations']<=3200000
        assert self.audit['inference_views']<=205120
        assert sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())<2*1024**3

    def done(self,name,data):
        obj={'action':name,'success':True,'identity':self.identity,'audit':dict(self.audit),**data}
        atomic_json(self.out/(name+'.done.json'),obj);self.event('action_done',action=name,audit=dict(self.audit))
        return obj

    def rng(self):
        n=np.random.get_state()
        return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
                'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}

    def state(self):
        return {'phase':self.phase,'step':self.step,'epoch':self.epoch,'cursor':self.cursor,'presentations':self.presentations,
                'model':self.m.state_dict(),'optimizer':self.opt.state_dict(),'training_mode':self.m.training,
                'optimizer_parameter_names':[n for n,_ in self.m.named_parameters()],
                'scheduler':{'implementation':'manual_cosine_per_step' if self.phase=='formal' else 'constant_benchmark',
                  'base_lr':.1,'Tmax':25040,'next_global_step':self.step,'current_lr':float(self.opt.param_groups[0]['lr'])},
                'rng':self.rng(),'permutation':self.perm,'normalization':self.normalization,
                'numeric_policy':{'FP32':True,'AMP':False,'TF32':False,'cudnn_deterministic':True,'cudnn_benchmark':False}}

    def checkpoint(self,name):
        torch.cuda.synchronize();t=time.perf_counter()
        receipt=save_checkpoint(self.out/(name+'.pt'),self.state(),self.identity)
        receipt['save_seconds']=time.perf_counter()-t;receipt['audit']=dict(self.audit)
        atomic_json(self.out/(name+'.receipt.json'),receipt)
        return receipt

    def restore(self,path,receipt):
        s=validate_checkpoint(path,receipt,self.identity)['state']
        self.m,self.opt=self.ns['model_optim']()
        self.m.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.m.train(s['training_mode'])
        assert [n for n,_ in self.m.named_parameters()]==s['optimizer_parameter_names']
        assert canonical_hash(s['normalization'])==canonical_hash(self.normalization)
        self.phase=s['phase'];self.step=s['step'];self.epoch=s['epoch'];self.cursor=s['cursor'];self.presentations=s['presentations']
        self.perm=s['permutation'].to('cuda') if s['permutation'] is not None else None
        self.opt.zero_grad(set_to_none=True)
        r=s['rng'];random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        torch.set_rng_state(r['torch_cpu']);torch.cuda.set_rng_state_all(r['torch_cuda'])
        assert canonical_hash(self.state())==receipt['state_content_sha256'],'Restored complete state differs'

    def benchmark_advance(self,n):
        assert self.phase=='benchmark' and self.audit['benchmark_steps']+n<=200
        loss_values=[];segments=[];torch.cuda.synchronize();t=time.perf_counter()
        first_ids=self.perm[self.cursor*128:(self.cursor+1)*128].cpu().numpy().tobytes()
        first_rng=canonical_hash(torch.cuda.get_rng_state_all())
        for j in range(n):
            ids=self.perm[self.cursor*128:(self.cursor+1)*128]
            loss=self.ns['update'](self.m,self.opt,ids);loss_values.append(loss)
            self.step+=1;self.cursor+=1;self.presentations+=len(ids)
            self.audit['benchmark_steps']+=1;self.audit['benchmark_presentations']+=len(ids)
            if (j+1)%50==0:
                torch.cuda.synchronize();now=time.perf_counter();segments.append((now-t)/50);t=now;self.watchdog()
        losses=torch.stack(loss_values).cpu().numpy()
        return {'losses_sha256':hashlib.sha256(losses.tobytes()).hexdigest(),'first_loss':float(losses[0]),
                'first_ids_sha256':hashlib.sha256(first_ids).hexdigest(),'first_cuda_rng_sha256':first_rng,
                'seconds_per_step_segments':segments}

    def benchmark_reference(self):
        assert self.audit['benchmark_steps']==0
        self.m,self.opt=self.ns['model_optim']();self.m.train();self.phase='benchmark';self.step=self.cursor=self.epoch=self.presentations=0
        self.perm=self.ns['trainidx'][torch.randperm(40000,device='cuda')]
        first=self.benchmark_advance(100);c100=self.checkpoint('pilot100')
        continuation=self.benchmark_advance(50);c150=self.checkpoint('pilot150_reference')
        self.reference={'first100':first,'continuation50':continuation,'checkpoint100':c100,'checkpoint150':c150}
        return self.done('benchmark_reference',self.reference)

    def benchmark_restore(self,uploaded_path):
        reference=json.loads((self.out/'benchmark_reference.done.json').read_text())
        self.restore(uploaded_path,reference['checkpoint100'])
        continuation=self.benchmark_advance(50);restored=self.checkpoint('pilot150_restored')
        assert continuation['losses_sha256']==reference['continuation50']['losses_sha256']
        assert continuation['first_ids_sha256']==reference['continuation50']['first_ids_sha256']
        assert continuation['first_cuda_rng_sha256']==reference['continuation50']['first_cuda_rng_sha256']
        assert restored['state_content_sha256']==reference['checkpoint150']['state_content_sha256']
        ids=torch.from_numpy(self.held_ids[:512]).to('cuda');torch.cuda.synchronize();t=time.perf_counter()
        path=self.out/'pilot_logits.npy';arr=self.ns['infer'](self.m,ids,path);torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        receipt=self.logit_receipt(path,0,512,benchmark=True)
        assert self.audit['benchmark_steps']==200 and self.audit['inference_views']==5120
        return self.done('benchmark_restore',{'restored_complete_state_bitwise_equal':True,'next_step_and50step_sequence_equal':True,
                'continuation50':continuation,'checkpoint_restored':restored,'pilot_logits':receipt,'inference_seconds':elapsed})

    def start_formal(self,gate):
        assert self.audit['benchmark_steps']==200 and self.audit['formal_steps']==0
        assert gate['passed'] and gate['cumulative_projection_seconds']+(time.time()-gate['measured_epoch'])<=5400
        self.m,self.opt=self.ns['model_optim']();self.m.train();self.phase='formal'
        self.step=self.epoch=self.cursor=self.presentations=0;self.perm=None
        receipt=self.checkpoint('formal000');return self.done('formal_start',{'checkpoint':receipt,'gate':gate,'benchmark_weights_discarded':True})

    def train_chunk(self,checkpoint_name,expected_sha,target_epoch):
        action=f'train{target_epoch:03d}'
        if (self.out/(action+'.done.json')).exists():return json.loads((self.out/(action+'.done.json')).read_text())
        receipt=json.loads((self.out/(checkpoint_name+'.receipt.json')).read_text());assert receipt['sha256']==expected_sha
        assert receipt['phase']=='formal' and receipt['cursor']==0 and target_epoch==receipt['epoch']+5 and target_epoch<=80
        assert self.audit['formal_steps']==receipt['step'],'Unknown chunk state; cannot replay training'
        self.restore(self.out/(checkpoint_name+'.pt'),receipt)
        epochs=[];t=time.perf_counter()
        for epoch in range(self.epoch,target_epoch):
            perm=self.ns['trainidx'][torch.randperm(40000,device='cuda')];losses=[]
            for begin in range(0,40000,128):
                lr=.1*.5*(1+math.cos(math.pi*self.step/25040))
                for group in self.opt.param_groups:group['lr']=lr
                ids=perm[begin:begin+128];loss=self.ns['update'](self.m,self.opt,ids);losses.append(loss)
                self.step+=1;self.presentations+=len(ids)
                self.audit['formal_steps']+=1;self.audit['formal_presentations']+=len(ids)
                if self.step%100==0:self.watchdog()
            torch.cuda.synchronize();self.epoch=epoch+1
            record={'epoch':self.epoch,'formal_steps':self.step,'train_loss':float(torch.stack(losses).mean().cpu()),'lr':lr}
            epochs.append(record);self.event('epoch_done',**record)
        self.perm=None;self.cursor=0
        receipt=self.checkpoint(f'formal{target_epoch:03d}')
        return self.done(action,{'checkpoint':receipt,'epochs':epochs,'chunk_seconds':time.perf_counter()-t})

    def logit_receipt(self,path,start,end,benchmark=False):
        array=np.load(path,mmap_mode='r');assert array.dtype==np.float32 and array.shape==(end-start,10,100)
        receipt={'schema':1,'kind':'logits','identity':self.identity,'benchmark':benchmark,'path':str(path),'filename':path.name,
                 'row_start':start,'row_end':end,'image_ids':self.held_ids[start:end].tolist(),'roles':self.held_roles[start:end].tolist(),
                 'source':['official_train' if i<50000 else 'official_test' for i in self.held_ids[start:end]],
                 'source_indices':[int(i if i<50000 else i-50000) for i in self.held_ids[start:end]],
                 'view_ids':list(range(10)),'view_definition':[[0,0,False],[0,8,False],[8,0,False],[8,8,False],[4,4,False],
                    [0,0,True],[0,8,True],[8,0,True],[8,8,True],[4,4,True]],'class_ids':list(range(100)),
                 'shape':list(array.shape),'dtype':array.dtype.str,'bytes':path.stat().st_size,'sha256':sha_file(path),
                 'logits_bytes_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
        if not benchmark:receipt['checkpoint_sha256']=json.loads((self.out/'formal080.receipt.json').read_text())['sha256']
        atomic_json(path.with_suffix('.receipt.json'),receipt);return receipt

    def infer_chunk(self,index):
        assert self.audit['formal_steps']==25040 and self.epoch==80
        action=f'infer{index:03d}';path=self.out/(action+'.npy')
        if (self.out/(action+'.done.json')).exists():return json.loads((self.out/(action+'.done.json')).read_text())
        assert self.audit['inference_views']==5120+min(index*512,20000)*10,'Unknown inference progress; no duplicate views'
        start=index*512;end=min(start+512,20000);assert start<end
        ids=torch.from_numpy(self.held_ids[start:end]).to('cuda')
        self.ns['infer'](self.m,ids,path);receipt=self.logit_receipt(path,start,end)
        return self.done(action,{'logits':receipt})
