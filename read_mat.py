import numpy as np
import binvox_rw
import torch.nn.functional as F
import torch
import scipy.io
from utils import recover_voxel, data_save
from skimage import measure
from stl import mesh

# file_name = 'test/'
file_name = '55e-6_15e-7_ep10000/'
data_folder = './output/'+file_name+'gen_data/'
vox_data_folder = data_folder + 'vox_data/'
mat_data_folder = data_folder + 'mat_data/'
stl_data_folder = data_folder + 'stl_data/'
mat_data = scipy.io.loadmat(mat_data_folder+'data.mat')
data = mat_data['tmp_voxel_gen']
print("data.shape",data.shape)
data_005 = np.where(data > 0.05, 1, 0)
# data_01 = np.where(data > 0.1, 1, 0)
# data_02 = np.where(data > 0.2, 1, 0)
# data_03 = np.where(data > 0.3, 1, 0)
# data_04 = np.where(data > 0.4, 1, 0)
# data_05 = np.where(data > 0.5, 1, 0)
# data_06 = np.where(data > 0.6, 1, 0)
# print(data_01[0].shape)
# print("data",np.sum(data))
# print("data_005",np.sum(data_005[0]))
# print("data_01 ",np.sum(data_01[0]))
# print("data_02 ",np.sum(data_02[0]))
# print("data_03 ",np.sum(data_03[0]))
# print("data_04 ",np.sum(data_04[0]))
# print("data_05 ",np.sum(data_05[0]))
# print("data_06 ",np.sum(data_06[0]))

voxs_gen = data_005

vox = []
mask_margin = (13,14,12)
xmin,xmax,ymin,ymax,zmin,zmax = 13, 243, 70, 186, 116, 140
for vox_gen in voxs_gen:
    # vox_real_size = recover_voxel(vox_gen,mask_margin,256,xmin,xmax,ymin,ymax,zmin,zmax)
    # vox.append(vox_real_size)
    vox.append(vox_gen)
vox = np.array(vox)
data_list = data_save(vox, vox_data_folder)

for data_name in data_list:
    with open(vox_data_folder+data_name, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f, False)
    model.data = model.data.astype(np.float32)
    # print("Data min:", model.data.min())
    # print("Data max:", model.data.max())
    verts, faces, normals, values = measure.marching_cubes(model.data, level=0)

    # create STL mesh
    your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            your_mesh.vectors[i][j] = verts[f[j], :]
    # save as STL
    stl_name = stl_data_folder+data_name.replace('.binvox','.stl')
    your_mesh.save(stl_name)
    print("save stl file:",stl_name)