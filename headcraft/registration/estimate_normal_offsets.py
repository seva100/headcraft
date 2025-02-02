from collections import defaultdict
import sys
from glob import glob
import os
import shutil
import time
import subprocess
import trimesh
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj, load_ply, save_ply

# To be able to import other modules from the project:
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from headcraft.loss.chamfer import pruned_chamfer_loss
from headcraft.utils.renderer import make_renderer

from headcraft.utils.mesh import triangle_uvs_to_vertex_uvs

from pytorch3d.renderer import (
    look_at_view_transform,
    TexturesVertex
)

# suppressing UserWarning from pytorch3d (edges_packed)
import warnings
warnings.filterwarnings("ignore")


def model(w, template_ver_t, normals_template, nullify_offsets_mask=None):
    # offsets: (n_template_vertices, 3)
    
    offsets = w.clone()
    
    if nullify_offsets_mask is not None:
        offsets[nullify_offsets_mask] = 0
    
    # normal space:
    offsets = (normals_template * w)
    upd_ver = template_ver_t + offsets
    
    return upd_ver


def mesh_laplacian_smoothing(meshes, method: str = "uniform", vertex_weights=None):

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()

    loss = L.mm(verts_packed)
    loss = loss.norm(p=2, dim=1)    # changed the norm type
    
    # print('loss mean:', loss.mean())
    if vertex_weights is not None:
        loss = loss * vertex_weights
    
    loss = loss * weights
    return loss.norm(p=2) / N, loss


def loss_func(pred_mesh, target, src_ver, normals_target, nullify_offsets_mask, template_faces_t, device='cuda:0'):

    # --------------------------------------------------------
    # loss function weights/params (hardcoded for simplicity)
    l_chamfer_weight = 20000.0
    pruning_dist_thr = 1.0
    pruning_normals_thr = 1.0
    l_edges_weight = 20000.0
    l_laplacian_weight = 10000.0
    # --------------------------------------------------------
    
    pred = pred_mesh.verts_list()[0]
    
    offset_mesh = pred_mesh.clone()
    offset_mesh.offset_verts_(-src_ver.float())

    chamfer_term = pruned_chamfer_loss(
        pred,
        target,
        x_normals=pred_mesh.verts_normals_packed(),
        y_normals=normals_target,
        dist_thr=pruning_dist_thr,
        normals_thr=pruning_normals_thr,
        mask_x=(~nullify_offsets_mask),
        mask_y=None
    )

    edges_len_term = torch.zeros(1).to(device)
    if l_edges_weight > 1e-8:
        for i in range(3):
            for j in range(i + 1, 3):
                edge_starts = offset_mesh.verts_packed()[template_faces_t[:, i]]
                edge_ends = offset_mesh.verts_packed()[template_faces_t[:, j]]
                edges_len_term += ((edge_ends - edge_starts) ** 2).mean()
    
    laplacian_term = torch.zeros(1).to(device)
    laplacian_per_ver = None
    if l_laplacian_weight > 1e-8:
        laplacian_term, laplacian_per_ver = mesh_laplacian_smoothing(offset_mesh)
    
    loss_val = (
        l_chamfer_weight * chamfer_term 
        + l_edges_weight * edges_len_term 
        + l_laplacian_weight * laplacian_term
    )
    return loss_val, {
        'chamfer_term': chamfer_term.item(),
        'edges_len_term': edges_len_term.item(),
        'laplacian_term': laplacian_term.item(),
        'laplacian_per_ver': laplacian_per_ver
    }


def train(template_ver_t, template_faces_t, target_ver_t, target_mesh, 
          normals_template, nullify_offsets_mask,
          n_iter=1000, visualize=False, save_every=20, save_dir=None, clear_save_dir=False,
          make_video=False, video_save_name=None, device='cuda:0', progress_bar=True):

    # --------------------------------------------------------
    # training params
    lr = 3e-4
    # --------------------------------------------------------

    # renderers
    template_center = template_ver_t.mean(axis=0, keepdim=True)
    # The first value in look_at_view_transform(...) is the distance from the camera to the object
    # that depends on the gt_scaling_coef -- adjust accordingly if required.
    renderers = [
        make_renderer(*look_at_view_transform(40, 180, 90, at=template_center, device=device)),    # left
        make_renderer(*look_at_view_transform(35, 180, 180, at=template_center, device=device)),   # frontal
        make_renderer(*look_at_view_transform(40, 0, 90, at=template_center, device=device)),      # right
        make_renderer(*look_at_view_transform(40, 90, 0, at=template_center, device=device)),      # above
        make_renderer(*look_at_view_transform(40, 180, 0, at=template_center, device=device)),      # back
    ]

    offsets = torch.randn_like(template_ver_t).float() * 0.0001
    offsets.requires_grad_(True)

    opt = torch.optim.Adam([offsets], lr=lr)

    if visualize:
        os.makedirs(save_dir, exist_ok=True)
        if clear_save_dir and len(os.listdir(save_dir)) > 0:
            print('Warning: deleting everything from visualization output dir!')
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
    
    normals_target = target_mesh.verts_normals_packed()
    
    if progress_bar:
        progress = tqdm(range(n_iter))
    else:
        progress = range(n_iter)
    for step_no in progress:
        opt.zero_grad()
    
        ver = model(offsets, template_ver_t, normals_template, nullify_offsets_mask=nullify_offsets_mask)
        pred_mesh = Meshes(verts=ver.unsqueeze(0), faces=template_faces_t.unsqueeze(0))
        
        loss, loss_dict = loss_func(pred_mesh, target_ver_t, src_ver=template_ver_t, normals_target=normals_target, nullify_offsets_mask=nullify_offsets_mask, template_faces_t=template_faces_t, device=device)
        
        loss.backward()
        opt.step()

        if visualize:
            textures = TexturesVertex(verts_features=torch.ones_like(template_ver_t).unsqueeze(0).to(device))
            pred_mesh.textures = textures
        
            verts_rgb = torch.ones_like(target_ver_t)[None] * torch.tensor([[[166 / 255., 200 / 255., 255 / 255.]]]).to(device)
            textures = TexturesVertex(verts_features=verts_rgb.to(device))
            target_mesh.textures = textures
        
            if step_no % save_every == 0:
                
                images = np.vstack([
                    np.hstack([
                        R(pred_mesh)[0, ..., :3].cpu().data.numpy()
                        for R in renderers
                    ]),
                    np.hstack([
                        R(target_mesh)[0, ..., :3].cpu().data.numpy()
                        for R in renderers
                    ])
                ])
                
                plt.figure(figsize=(10, 5))
                plt.imshow(images)
                plt.axis("off")
                plt.title(f'iteration: {step_no} / {n_iter}')
                plt.tight_layout()
                # plt.title(f'iteration: {step_no} / {n_train_steps}; loss_lapl = {loss_dict["laplacian_term"]}')
                plt.savefig(os.path.join(save_dir, f'{step_no:06d}.jpg'), dpi=300)

                # Uncomment if loss value output is needed:
                # print(step_no, 'steps', 
                #       'loss:', loss.item(), 
                #       'chamfer', f"{l_chamfer_weight * loss_dict['chamfer_term']:.4f}", 
                #       'edges', f"{l_edges_weight * loss_dict['edges_len_term']:.4f}", 
                #       'lapl', f"{l_laplacian_weight * loss_dict['laplacian_term']:.4f}")
    
    if visualize and make_video:

        # This block requires ffmpeg to be installed.
        fps = 5
        rate = 30

        cmd = f'ffmpeg -y -framerate {fps} -pattern_type glob -i "{save_dir}/*.jpg" -c:v libx264 -r {rate} -pix_fmt yuv420p {save_dir}/{video_save_name}.mp4'
        # print(cmd)
        # with open(os.devnull, 'w') as fp:    # suppressing all output for convenience
        subprocess.Popen(cmd,
                         shell=True, 
                         # stdout=subprocess.PIPE)
                         stdout=subprocess.STDOUT)
    
    return offsets, loss, loss_dict


def process_sample(ver_path, faces_path, uv_path, gt_path, uv_mask_path, output_path, 
                   gt_scaling_coef=30, device='cuda:0', progress_bar=True,
                   n_iter=1000, visualize_progress=True, vis_save_every_n_steps=200):
    
    m_template_ver = np.load(ver_path)
    m_template_faces = np.load(faces_path)
    m_template_uvs = np.load(uv_path)

    # loading target
    m_gt = trimesh.load_mesh(gt_path)
    m_gt.vertices *= gt_scaling_coef

    if uv_mask_path is None:
        uv_mask = None
    else:
        uv_mask = iio.imread(uv_mask_path)[..., 0]

    template_ver_t   = torch.tensor(np.asarray(m_template_ver)).to(device).float()
    template_ver_t.requires_grad_(False)
    target_ver_t  = torch.tensor(np.asarray(m_gt.vertices)).to(device).float()
    target_ver_t.requires_grad_(False)
    template_faces_t = torch.tensor(np.asarray(m_template_faces)).to(device)
    template_faces_t.requires_grad_(False)
    target_faces_t  = torch.tensor(np.asarray(m_gt.faces)).to(device).float()
    target_faces_t.requires_grad_(False)
    template_uvs_t = torch.tensor(m_template_uvs).to(device).float()
    _ = template_uvs_t.requires_grad_(False)

    if uv_mask is not None:
        h, w = uv_mask.shape[:2]
        seam_thr = 0.1
        vertex_uvs = triangle_uvs_to_vertex_uvs(m_template_uvs, len(m_template_ver), m_template_faces, seam_thr=seam_thr).astype(np.float32)
        nullify_offsets_mask = (uv_mask[((1 - vertex_uvs[:, 1]) * h).astype(int), 
                                        (vertex_uvs[:, 0] * w).astype(int)] < 0.5)
        nullify_offsets_mask = torch.tensor(nullify_offsets_mask).bool()
    else:
        nullify_offsets_mask = None
    
    template_mesh = Meshes(verts=template_ver_t.unsqueeze(0), faces=template_faces_t.unsqueeze(0))
    normals_template = template_mesh.verts_normals_packed()
    target_mesh = Meshes(verts=target_ver_t.unsqueeze(0), faces=target_faces_t.unsqueeze(0))

    # processing:
    offsets, loss, loss_dict = train(template_ver_t, template_faces_t, target_ver_t, target_mesh, normals_template, nullify_offsets_mask,
                                     n_iter=n_iter,
                                     visualize=visualize_progress,
                                     save_every=vis_save_every_n_steps, 
                                     save_dir=output_path,
                                     clear_save_dir=False, make_video=False, device=device, progress_bar=progress_bar)
    
    # saving npy:
    np.save(os.path.join(output_path, 'offsets.npy'), offsets.cpu().data.numpy())
    ver_upd = model(offsets, template_ver_t, normals_template, nullify_offsets_mask=nullify_offsets_mask)
    np.save(os.path.join(output_path, 'ver_upd.npy'), ver_upd.cpu().data.numpy())
    np.save(os.path.join(output_path, 'faces.npy'), template_faces_t.cpu().data.numpy())
    np.save(os.path.join(output_path, 'uvs.npy'), template_uvs_t.cpu().data.numpy())

    # saving a mesh with estimated offsets
    save_ply(os.path.join(output_path, 'result.ply'),
             ver_upd,
             template_faces_t)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    
    # most important args
    parser.add_argument('--root_dir', type=str, 
                        help='path to the dataset folder that contains meshes in a structure "root_dir/seq_name/expr_name/subdivided/template_decim_ver.npy" and vector offsets "root_dir/seq_name/expr_name/estimated_vector_offsets/{ver_upd.npy, faces.npy, uvs.npy}"')
    parser.add_argument('--target_root_dir', type=str, 
                        help='path to the dataset folder that contains meshes in a structure "target_root_dir/seq_name/expr_name/scan.ply"')
    parser.add_argument('--output_root_dir', type=str, 
                        help='path to the folder where all the processed data will be saved. '
                             'The subfolder "estimated_vector_offsets" will be created inside each seq_name/expr_name subfolder (if it hasn\'t been yet).')
    parser.add_argument('--gt_scaling_coef', type=float, default=30, help='scaling coefficient for the ground truth mesh used in subdiv_flame.py')
    parser.add_argument('--uv_mask_path', type=str, default=None,
                        help='UV mask specifying which vertices to block (in the original FLAME UV space)')
    
    # optional: processing only a part of the dataset (useful when e.g. parallelizing onto several GPUs)
    parser.add_argument('--n_parts', type=int, default=None, help='whether to process a dataset in a few parts in parallel')
    parser.add_argument('--part_idx', type=int, default=None, help='index of the part to process; makes sense to use only with --n_parts')

    # optional: process only N first IDs or N first expressions of each ID
    parser.add_argument('--only_n_first_ids', type=int, default=None, help='if specified, only first n names will be processed')
    parser.add_argument('--only_n_first_expr', type=int, default=None, help='if specified, only first n expressions per person will be processed')

    # optional: individual optimization params
    parser.add_argument('--disable_inner_progress_bar', action='store_true', help='whether to hide the progress bar for the individual optimization of each sample')
    parser.add_argument('--n_iter', type=int, default=1000, help='number of iterations for the individual optimization')
    parser.add_argument('--visualize_progress', action='store_true', help='whether to visualize the optimization progress')
    parser.add_argument('--vis_save_every_n_steps', type=int, default=200, help='save visualization every this number of steps (only if visualize_progress is True)')
    
    # optional: whether to skip already processed samples
    parser.add_argument('--skip_saved', action='store_true', help='whether to skip already processed samples')
    args = parser.parse_args()

    if args.uv_mask_path is None:
        args.uv_mask_path = os.path.join(os.path.dirname(__file__), 
                                         '..', '..', 'data', 'flame', 'flame_uv_mask_stage2_allowed_no_face.png')

    all_inp = glob(os.path.join(args.root_dir, '*', '*', 'estimated_vector_offsets', 'ver_upd.npy'))
    all_inp = list(sorted(all_inp))
    print(len(all_inp))

    if args.n_parts is not None and args.part_idx is not None:
        all_inp = all_inp[args.part_idx::args.n_parts]
        print('processing part', args.part_idx, 'of', args.n_parts, f'({len(all_inp)} samples)')

    if args.only_n_first_ids is not None:
        available_seq = list(set(
            [int(name.split(os.sep)[-4]) for name in all_inp]
        ))
        available_seq.sort()
        suitable_seq = set(available_seq[:args.only_n_first_ids])
        all_inp = [name for name in all_inp
                   if int(name.split(os.sep)[-4]) in suitable_seq]
    
    if args.only_n_first_expr is not None:
        seq2available_exps = defaultdict(list)
        for name in all_inp:
            seq_name = int(name.split(os.sep)[-4])
            exp_name = int(name.split(os.sep)[-3])
            seq2available_exps[seq_name].append(exp_name)
        for seq_name in seq2available_exps.keys():
            seq2available_exps[seq_name].sort()
        all_inp = []
        for seq_name, exps in seq2available_exps.items():
            suitable_exps = exps[:args.only_n_first_expr]
            for exp_name in suitable_exps:
                all_inp.append(os.path.join(args.root_dir, f'{seq_name:03}', f'{exp_name:03}', 'estimated_vector_offsets', 'ver_upd.npy'))

    for ver_path in tqdm(all_inp):

        # loading template
        seq_name = ver_path.split(os.sep)[-4]
        exp_name = ver_path.split(os.sep)[-3]

        # output path for visualization and estimated offsets
        output_path = os.path.join(args.output_root_dir, seq_name, exp_name, 'estimated_normal_offsets')
        os.makedirs(output_path, exist_ok=True)

        if args.skip_saved and os.path.exists(os.path.join(output_path, 'offsets.npy')) and os.path.exists(os.path.join(output_path, 'result.ply')):
            print('skipping', seq_name)
            continue
        
        m_template_ver_path = ver_path
        m_template_faces_path = os.sep.join(ver_path.split(os.sep)[:-1] + ['faces.npy'])
        m_template_uvs_path = os.sep.join(ver_path.split(os.sep)[:-1] + ['uvs.npy'])

        # loading target
        m_gt_path = os.path.join(args.target_root_dir, seq_name, exp_name, 'scan.ply')

        # output path for visualization and estimated offsets

        process_sample(m_template_ver_path, m_template_faces_path, m_template_uvs_path, 
                       m_gt_path, args.uv_mask_path, output_path, 
                       gt_scaling_coef=args.gt_scaling_coef, progress_bar=not args.disable_inner_progress_bar,
                       n_iter=args.n_iter, visualize_progress=args.visualize_progress,
                       vis_save_every_n_steps=args.vis_save_every_n_steps)
