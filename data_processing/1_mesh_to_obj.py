import trimesh
from stl import mesh
import os

def stl_to_obj(stl_file_path, obj_file_path):
    mesh_data = mesh.Mesh.from_file(stl_file_path)

    vertices = mesh_data.vectors.reshape(-1, 3)
    faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    trimesh_mesh.export(obj_file_path)
    print(f"Converted: {obj_file_path}")

import os
import shutil
data_class_u = 'Diamond'
data_class_l = 'diamond'

source_folder = '../data_ori/'+data_class_u+'/'
target_root = '../data/' + data_class_l+ '/' 
prefix = data_class_l+'_unit'
stl_folders = os.listdir(source_folder)
stl_folders = sorted(stl_folders)
if not os.path.exists(target_root):
    os.makedirs(target_root)

for stl_folder in stl_folders:
    data_folder = source_folder + stl_folder+'/'
    tar_folder = target_root + prefix + stl_folder + '_'
    data_list = os.listdir(data_folder)
    data_list = sorted(data_list)
    for data in data_list:
        source_data = data_folder + data
        target_folder = tar_folder+ data.split('.')[0]+'/'
        target_data = target_folder+'model_ori.obj'
        # print(source_data, target_data)
        # print(target_folder)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        stl_to_obj(source_data, target_data)