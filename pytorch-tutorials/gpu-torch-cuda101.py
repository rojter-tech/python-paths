#%%
import torch
import pycuda.driver as cuda
cuda.init()
torch.cuda.current_device()

torch.cuda.is_available()

#%%
torch.cuda.current_device()

#%%
cuda.Device(0).name()

#%%
