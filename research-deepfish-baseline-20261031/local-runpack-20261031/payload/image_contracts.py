"""Synthetic uint8/BGR Color gate; imports no detector or model."""
def require_bgr_uint8(img,backend):
 import numpy as np
 if backend!='cv2' or img.dtype!=np.uint8 or img.ndim!=3 or img.shape[2]!=3:raise ValueError('Only explicitly declared cv2 uint8 BGR is allowed.')
def verify_color_contract(Color):
 import cv2,numpy as np
 image=(np.arange(17*19*3,dtype=np.uint16)%256).astype(np.uint8).reshape(17,19,3)
 require_bgr_uint8(image,'cv2');saved=np.random.get_state()
 def state():
  x=np.random.get_state();return (x[0],x[1].tobytes(),x[2],x[3],x[4])
 try:
  np.random.seed(1729);a=Color(prob=1.0,level=0,min_mag=1.0,max_mag=1.0)({'img':image.copy()})['img'];ra=state()
  np.random.seed(1729);b=Color(prob=1.0,level=0,min_mag=.8,max_mag=.8)({'img':image.copy()})['img'];rb=state()
  np.random.seed(1729);np.random.rand();expected_rng=state()
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);expected=cv2.addWeighted(image,.8,np.repeat(gray[...,None],3,axis=2),.2,0)
  result=dict(identity_exact=bool(np.array_equal(a,image)),rng_equal=ra==rb,exactly_one_numpy_draw=ra==expected_rng,b_matches_cv2_bgr_blend=bool(np.array_equal(b,expected)),cv2_version=cv2.__version__,numpy_version=np.__version__)
  if not all(result[k] for k in ('identity_exact','rng_equal','exactly_one_numpy_draw','b_matches_cv2_bgr_blend')):raise ValueError('Color contract failed: '+str(result))
  require_bgr_uint8(a,'cv2');require_bgr_uint8(b,'cv2');return result
 finally:np.random.set_state(saved)
