from collections import defaultdict
import os
import sys
from glob import glob
import tempfile
import numpy as np
import torch
import open3d as o3d
import pymeshlab as ml    # only required for decimate_mesh_meshlab_butterfly

from pytorch3d.io import load_obj, load_ply, save_obj, save_ply
from tqdm import tqdm


def remove_duplicate_vertices(ver, faces):
    # ver: numpy array (|V|, 3)
    # faces: numpy array (|F|, 3)
    # uvs: numpy array (|F|, 3, 2)
    # returns: ver_upd, faces_upd, uvs_upd

    ver_set = defaultdict(list)
    for i, v in enumerate(ver):
        ver_set[tuple(v)].append(i)
    
    ver_upd = []
    ver_upd_set = dict()
    ver_old2new_idx = []
    for i, v in enumerate(ver):
        if tuple(v) not in ver_upd_set:
            ver_upd.append(v)
            ver_upd_set[tuple(v)] = len(ver_upd) - 1
            ver_old2new_idx.append(len(ver_upd) - 1)
        else:
            ver_old2new_idx.append(ver_upd_set[tuple(v)])
    
    ver_upd = np.array(ver_upd)
    ver_old2new_idx = np.array(ver_old2new_idx)
    faces_upd = ver_old2new_idx[faces]

    return ver_upd, faces_upd


def subdivide_mesh_meshlab_butterfly(vertices, faces, uvs, ms_script_loc):
    # This script calls MeshLab to subdivide a mesh using the butterfly subdivision scheme.
    # The subdivision granularity is controlled in the .mlx file provided as `ms_script_loc`.
    # If the head mesh is in NPHM coord system and scale (120x larger than the standard FLAME scale = 4x for NPHM convention and 30x used in this work ad-hoc), this subdivision produces roughly 99K vertices.
    # Note that for every mesh, the number of produced vertices is different. For the exactly the same number, use `consistent_subdiv(...)` from demo/animate.py.

    # save a mesh to a temporary file with UVs attached to it
    tmp_file_in, filename_in = tempfile.mkstemp(suffix='.obj')
    tmp_file_out, filename_out = tempfile.mkstemp(suffix='.obj')
    os.close(tmp_file_in)    # we don't need to write to it via python interface
    os.close(tmp_file_out)    # we don't need to write to it via python interface

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().data.numpy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().data.numpy())
    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs.cpu().data.numpy().reshape(-1, 2))
    o3d.io.write_triangle_mesh(filename_in, o3d_mesh)

    # run meshlab script using pymeshlab to decimate the mesh
    ms = ml.MeshSet(verbose=0)
    ms.load_new_mesh(filename_in)
    # ms.load_filter_script('/rhome/asevastopolsky/source/uv-to-offset/data/meshlab/decimate_butterfly.mlx')
    ms.load_filter_script(ms_script_loc)
    ms.apply_filter_script()
    ms.save_current_mesh(filename_out)

    # load the decimated mesh from the temporary file
    o3d_mesh = o3d.io.read_triangle_mesh(filename_out)
    ver_upd = np.asarray(o3d_mesh.vertices)
    faces_upd = np.asarray(o3d_mesh.triangles)
    uvs_upd = np.asarray(o3d_mesh.triangle_uvs).reshape(-1, 3, 2)

    # sometimes meshlab duplicates vertices at the UV seams, which we need to avoid here
    ver_upd, faces_upd = remove_duplicate_vertices(ver_upd, faces_upd)

    ver_upd = torch.from_numpy(ver_upd).float().to(vertices.device)
    faces_upd = torch.from_numpy(faces_upd).long().to(vertices.device)
    uvs_upd = torch.from_numpy(uvs_upd).float().to(vertices.device)

    # remove the temporary file
    os.remove(filename_in)
    os.remove(filename_out)

    return ver_upd, faces_upd, uvs_upd


def process_sample(input_path, output_path, scaling_coef=30):
    # input_path: path to .ply file with a g.t. mesh
    # output_path: path to the folder where all the processed data will be saved. 
    #   The subfolder "subdivided" will be created inside of it (if it hasn't been yet).

    file_loc = os.path.dirname(os.path.realpath(__file__))
    uv_path = os.path.join(file_loc, '..', '..', 'data', 'flame', 'flame_triangle_uvs.npy')
    mlx_script_path = os.path.join(file_loc, '..', '..', 'data', 'meshlab', 'subdivide_butterfly.mlx')
    seq_name, exp_name = input_path.split(os.sep)[-3:-1]
    output_path_dir = os.path.join(output_path, seq_name, exp_name, 'subdivided')
    os.makedirs(output_path_dir, exist_ok=True)

    # load a mesh
    mesh_ver, mesh_faces = load_ply(input_path)
    # flame_mesh_ver, flame_mesh_faces, flame_mesh_prop = load_obj(flame_path)
    mesh_ver *= scaling_coef

    # load uvs
    triangle_uvs = np.load(uv_path)     # (n_faces, 3, 2)
    triangle_uvs = torch.tensor(triangle_uvs).float()

    # processing
    ver_upd, faces_upd, uvs_upd = subdivide_mesh_meshlab_butterfly(mesh_ver, mesh_faces, triangle_uvs, mlx_script_path)

    # saving
    save_ply(os.path.join(output_path_dir, 'template_decim.ply'),
             ver_upd,
             faces_upd)
    np.save(os.path.join(output_path_dir, 'template_decim_ver.npy'), ver_upd.cpu().data.numpy())
    np.save(os.path.join(output_path_dir, 'template_decim_faces.npy'), faces_upd.cpu().data.numpy())
    np.save(os.path.join(output_path_dir, 'template_decim_uv.npy'), uvs_upd.cpu().data.numpy())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Subdivide a mesh using the butterfly subdivision scheme. The script only uses the CPU.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset folder that contains flame meshes in a structure "dataset_path/subject_id/subject_expr/flame.ply"')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the folder where all the processed data will be saved. The subfolder "subdivided" will be created inside each subject_id/subject_expr subfolder (if it hasn\'t been yet).')
    parser.add_argument('--scaling_coef', type=int, default=30, help='The scaling coefficient for the mesh (the larger it is, the more subdivided the flame will be). Default: 30')
    parser.add_argument('--n_jobs', type=int, default=-1, help='If specified, will process the dataset in parallel using the specified number of processes.')
    args = parser.parse_args()

    # scraping the flame.ply inside the dataset_path
    mesh_fns = glob(os.path.join(args.dataset_path, '*', '*', 'flame.ply'))
    mesh_fns = list(sorted(mesh_fns))
    print('Found {} meshes'.format(len(mesh_fns)))

    if args.n_jobs == -1:
        for fn in tqdm(mesh_fns):
            process_sample(args.input_path, args.output_path)
    else:
        import contextlib
        import joblib
        from joblib import Parallel, delayed

        @contextlib.contextmanager
        def tqdm_joblib(tqdm_object):
            """Context manager to patch joblib to report into tqdm progress bar given as argument"""
            class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    tqdm_object.update(n=self.batch_size)
                    return super().__call__(*args, **kwargs)

            old_batch_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            try:
                yield tqdm_object
            finally:
                joblib.parallel.BatchCompletionCallBack = old_batch_callback
                tqdm_object.close()
        

        with tqdm_joblib(tqdm(desc="Progress", 
                              total=len(mesh_fns))) as progress_bar:
            Parallel(n_jobs=args.n_jobs)(
                delayed(process_sample)(
                    input_path=mesh_path, 
                    output_path=args.output_path,
                    scaling_coef=args.scaling_coef)
                for mesh_path in mesh_fns
            )
