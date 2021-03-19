from Packages.SplitEbinMetric import *
from Packages.GeoPlot import *
import scipy.io as sio
import torch
import numpy as np

g00 = sio.loadmat('Data/103818_orig_tensors_masked_bg.mat')
g11 = sio.loadmat('Data/105923_orig_tensors_masked_bg.mat')
g00 = torch.from_numpy(g00['tensor']).double()
g11 = torch.from_numpy(g11['tensor']).double()
height = g00.size(-2)
width = g00.size(-1)
g0, g1 = torch.zeros(2,2,height,width,dtype=torch.double), torch.zeros(2,2,height,width,dtype=torch.double)
g0[0,0,:,:] = g00[0,:,:]
g0[0,1,:,:] = g00[1,:,:]
g0[1,0,:,:] = g00[1,:,:]
g0[1,1,:,:] = g00[2,:,:]
g1[0,0,:,:] = g11[0,:,:]
g1[0,1,:,:] = g11[1,:,:]
g1[1,0,:,:] = g11[1,:,:]
g1[1,1,:,:] = g11[2,:,:]
g0 = g0.permute(2,3,0,1)
g1 = g1.permute(2,3,0,1)

a = 0.5
Tpts = 5
geo_group = get_Geo(g0, g1, a, Tpts)
print('finish')
for i in range(Tpts):
    plot_2d_tensors(geo_group[i].permute(2,3,0,1), scale=1000, title=str(i), margin=0.05, dpi=80)