"""One precommitted integer texture. No tunable input or randomized retry."""
import numpy as np

SEED=20261031
WIDTH=1333
HEIGHT=800
POSITIVE=[[64,64,96,96],[160,64,192,96],[256,64,288,96]]
NEGATIVE=[[64,192,96,224],[160,192,192,224],[256,192,288,224]]
TEXTS={'F':('red toy','blue toy'),'C':('toy',''),'V':('','')}

def generate():
    y,x=np.indices((HEIGHT,WIDTH),dtype=np.uint32)
    z=x*np.uint32(374761393)+y*np.uint32(668265263)+np.uint32(SEED)
    z=(z^(z>>np.uint32(13)))*np.uint32(1274126177)
    z=z^(z>>np.uint32(16))
    result=np.stack([32+((5*x+3*y)%160)//2+(z&63),
                     24+((3*x+7*y)%176)//2+((z>>8)&63),
                     28+((11*x+2*y)%160)//2+((z>>16)&63)],axis=-1).astype(np.uint8)
    for x1,y1,x2,y2 in POSITIVE:
        result[y1:y2,x1:x2,0]=(176+(z[y1:y2,x1:x2]&63)).astype(np.uint8)
        result[y1:y2,x1:x2,1]=(40+((x[y1:y2,x1:x2]*5+y[y1:y2,x1:x2]*3)%64)).astype(np.uint8)
        result[y1:y2,x1:x2,2]=(50+((z[y1:y2,x1:x2]>>16)&63)).astype(np.uint8)
    for x1,y1,x2,y2 in NEGATIVE:
        result[y1:y2,x1:x2,2]=(176+((z[y1:y2,x1:x2]>>8)&63)).astype(np.uint8)
        result[y1:y2,x1:x2,0]=(40+((x[y1:y2,x1:x2]*7+y[y1:y2,x1:x2]*2)%64)).astype(np.uint8)
        result[y1:y2,x1:x2,1]=(50+(z[y1:y2,x1:x2]&63)).astype(np.uint8)
    return result
