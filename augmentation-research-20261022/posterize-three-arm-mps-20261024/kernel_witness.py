import torch,json
torch.set_num_threads(1)
torch.manual_seed(20261024)
assert torch.backends.mps.is_available()
x=torch.randn(16,16)
y=(x.to("mps")@x.to("mps")).cpu()
assert torch.isfinite(y).all()
assert torch.allclose(y,x@x,rtol=1e-4,atol=1e-5)
print("BZR_MPS_KERNEL_WITNESS",json.dumps({"device":"mps","shape":list(y.shape),"torch":torch.__version__,"finite":True}))
