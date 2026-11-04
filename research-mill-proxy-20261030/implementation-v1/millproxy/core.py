"""Numerical contracts and deterministic random domains (CPU float64)."""
import hashlib
import re
import numpy as np

Q=1.0/255
LEVELS=(1,3,5,7)
SEEDS=(11,22,33,44)
TRAIN_IDS=('Scene7','Scene16','Scene10_b','Scene12','Scene23_b','Scene15')
TEST_IDS=tuple('Test'+str(i) for i in range(13,21))

def finite(a, name='array'):
    a=np.asarray(a)
    if not np.isfinite(a).all():raise ValueError(name+' contains nonfinite values')
    return a

def rgb(a, *, quantized=False):
    a=np.asarray(a,dtype=np.float64)
    finite(a)
    if a.ndim<3 or a.shape[-3]!=3 or min(a.shape[-2:])<1:raise ValueError('Expected [...,3,H,W] RGB')
    if np.any(a<0) or np.any(a>1):raise ValueError('RGB outside [0,1]')
    if quantized and np.max(np.abs(a*255-np.rint(a*255)))>1e-10:raise ValueError('Source is not original 8-bit codes / 255')
    return a

def q8(t):
    return np.floor(np.clip(finite(np.asarray(t,dtype=np.float64)),0,1)*255+.5)/255

def source_bins(x):
    x=rgb(x,quantized=True)
    # On original uint8 codes this is the exact decimal-coefficient expression
    # floor(16*(.2126 R+.7152 G+.0722 B)), without moving a boundary left
    # through floating-point cancellation. This is indexing, not radiometry.
    code=np.rint(x*255).astype(np.int64)
    numerator=2126*code[...,0,:,:]+7152*code[...,1,:,:]+722*code[...,2,:,:]
    return np.minimum(15,(16*numerator)//2550000)

def equal_width_bins(h):
    h=finite(np.asarray(h,np.float64))
    if np.any(h<0) or np.any(h>1):raise ValueError('Bin coordinate outside [0,1]')
    return np.minimum(15,np.floor(16*h).astype(np.int64))

def rng_for(method,scene,level,seed,purpose):
    if method not in ('A','B') or level not in LEVELS or seed not in SEEDS:raise ValueError('Unknown RNG method/level/seed')
    if not re.fullmatch(r'[A-Za-z0-9_]+',scene):raise ValueError('Invalid scene identifier')
    if purpose not in ('gaussian','ties-R','ties-G','ties-B'):raise ValueError('Unknown RNG purpose')
    key=f'MILL-proxy-v1|method={method}|scene={scene}|level={level}|seed={seed}|purpose={purpose}'
    seed128=int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:16],'big')
    return np.random.Generator(np.random.PCG64(seed128))

def readonly(a):
    a=np.asarray(a)
    a.setflags(write=False)
    return a

def validate_training_pair(x,y):
    x=rgb(x,quantized=True);y=rgb(y,quantized=True)
    if x.shape!=y.shape or x.ndim!=4 or x.shape[0]!=6:raise ValueError('Exactly six matched training scenes required')
    return x,y
