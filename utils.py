import numpy as np
import binvox_rw
import torch.nn.functional as F
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_vox_from_binvox(objname):
    #get voxel models
    voxel_model_file = open(objname, 'rb')
    voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    step_size = 2
    voxel_model_256 = voxel_model_512[0::step_size,0::step_size,0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_256 = np.maximum(voxel_model_256,voxel_model_512[i::step_size,j::step_size,k::step_size])
    return voxel_model_256

def get_voxel_bbox(vox):
    #minimap
    vox_tensor = torch.from_numpy(vox).to(device).unsqueeze(0).unsqueeze(0).float()
    smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = 1, stride = 1, padding = 0)
    smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0,0]
    smallmaskx = np.round(smallmaskx).astype(np.uint8)
    smallx,smally,smallz = smallmaskx.shape
    #x
    ray = np.max(smallmaskx,(1,2))
    xmin = 0
    xmax = 0
    for i in range(smallx):
        if ray[i]>0:
            if xmin==0:
                xmin = i
            xmax = i
    #y
    ray = np.max(smallmaskx,(0,2))
    ymin = 0
    ymax = 0
    for i in range(smally):
        if ray[i]>0:
            if ymin==0:
                ymin = i
            ymax = i
    #z
    ray = np.max(smallmaskx,(0,1))
    zmin = 0
    zmax = 0
    for i in range(smallz):
        if ray[i]>0:
            if zmin==0:
                zmin = i
            zmax = i

    return xmin,xmax+1,ymin,ymax+1,zmin,zmax+1

def crop_voxel(vox, mask_margin,xmin,xmax,ymin,ymax,zmin,zmax):
    xspan = xmax-xmin
    yspan = ymax-ymin
    zspan = zmax-zmin
    tmp = np.zeros([xspan+mask_margin[0]*2,yspan+mask_margin[1]*2,zspan+mask_margin[2]*2], np.uint8)
    tmp[mask_margin[0]:-mask_margin[0],mask_margin[1]:-mask_margin[1],mask_margin[2]:-mask_margin[2]] = vox[xmin:xmax,ymin:ymax,zmin:zmax]
    return tmp

def data_save(data, folder):
    data_list = []
    for idx, vox_data in enumerate(data):
        voxels = binvox_rw.Voxels(
        data=vox_data,
        dims=vox_data.shape,
        translate=[-0.5, -0.5, -0.5],
        scale=1.0,
        axis_order='xzy'
        )
        data_name = f'gen_data_{idx}.binvox'
        filepath = folder+data_name
        binvox_rw.write_voxel(voxels, filepath)
        print(filepath + " has been saved.")
        data_list.append(data_name)
    return data_list

def recover_voxel_uni(vox,mask_margin,real_size,xmin,xmax,ymin,ymax,zmin,zmax):
    tmpvox = np.zeros([real_size,real_size,real_size], np.float32)
    xmin_,ymin_,zmin_ = mask_margin # (13, 14, 12)
    xmax_ = vox.shape[0] - mask_margin[0]
    ymax_ = vox.shape[1] - mask_margin[1]
    zmax_ = vox.shape[2] - mask_margin[2]
    tmpvox[xmin:xmax,ymin:ymax,zmin:zmax] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
    return tmpvox
