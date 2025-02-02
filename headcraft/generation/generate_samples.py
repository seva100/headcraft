from copy import deepcopy
import os
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from einops import rearrange
from tqdm import tqdm
import imageio
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torch_ema import ExponentialMovingAverage
from torchvision.utils import make_grid
from pytorch_lightning import Trainer
from pytorch3d.renderer import (
    look_at_view_transform,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, load_ply, save_ply

import sys
# sys.path.insert(0, '/rhome/asevastopolsky/Downloads/stylegan2-ada-lightning-v2')    # TODO replace with the submodule path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'stylegan2-ada-lightning'))    # submodule path

from model.generator import Generator

# To be able to import other modules from the project:
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from headcraft.utils.mesh import dilate_vertex_mask, find_seam_ver, duplicate_seam_ver, find_stretched_triangles, find_subdiv_ver_in_region
from headcraft.utils.uv import sample_map_with_coords
from headcraft.utils.renderer import make_ortho_renderer

# for --sample_flame only:
from omegaconf import OmegaConf
import yaml
from modules.flame.flame_model import FLAME, FLAMETex


def get_bary_face_inds(orig_flame_ver, orig_flame_faces, orig_flame_tri_uvs,
                       res=2048, device='cuda:0'):
    
    # rendering vertices to UV space
    center = torch.tensor([[0, 0, 0]], device=device).float()
    R, T = look_at_view_transform(1, 180, 180, at=center, device=device)

    rasterizer = make_ortho_renderer(R, T, device=device, res=res, make_rasterizer=True)

    # rasterizing vertices to the standard UV space to obtain barycentric coords and face indices 
    #  w.r.t. the original coarse template
    orig_flame_seam_ver_mask, orig_flame_vertex_uvs_prelim = \
        find_seam_ver(orig_flame_tri_uvs, orig_flame_ver.shape[0], orig_flame_faces)
    orig_flame_verts_upd, orig_flame_vertex_uvs, orig_flame_faces, _ = \
        duplicate_seam_ver(orig_flame_tri_uvs, orig_flame_ver, orig_flame_faces, 
                            orig_flame_seam_ver_mask, orig_flame_vertex_uvs_prelim)
    orig_flame_faces_t = torch.tensor(orig_flame_faces, device=device).long()
    orig_flame_verts_t = torch.tensor(orig_flame_verts_upd, device=device).float()
    orig_flame_vertex_uvs_t = torch.tensor(orig_flame_vertex_uvs, device=device).float()
    orig_flame_vertex_uvs_t = torch.cat([orig_flame_vertex_uvs_t, 
                                        torch.zeros(orig_flame_vertex_uvs_t.shape[0], 1, device=device)], dim=1)    # (n_ver, 3)

    mesh = Meshes(verts=[orig_flame_vertex_uvs_t], faces=[orig_flame_faces_t])

    fragments_orig_flame_std_layout = rasterizer(mesh)

    pix_to_face = fragments_orig_flame_std_layout.pix_to_face.cpu().data.numpy()    # (1, H, W, 2)
    bary_coords = fragments_orig_flame_std_layout.bary_coords.cpu().data.numpy()    # (N, H, W, 2, 3)

    return pix_to_face, bary_coords


def uvs_to_custom_layout(ver_upd, faces, uvs, 
                         orig_flame_ver, orig_flame_faces, orig_flame_tri_uvs,
                         mod_tri_uvs,
                         res=2048, device='cuda:0'):

    # # rendering vertices to UV space
    # center = torch.tensor([[0, 0, 0]], device=device).float()
    # R, T = look_at_view_transform(1, 180, 180, at=center, device=device)

    # rasterizer = make_ortho_renderer(R, T, device=device, res=res, make_rasterizer=True)

    # # rasterizing vertices to the standard UV space to obtain barycentric coords and face indices 
    # #  w.r.t. the original coarse template
    # orig_flame_seam_ver_mask, orig_flame_vertex_uvs_prelim = \
    #     find_seam_ver(orig_flame_tri_uvs, orig_flame_ver.shape[0], orig_flame_faces)
    # orig_flame_verts_upd, orig_flame_vertex_uvs, orig_flame_faces, _ = \
    #     duplicate_seam_ver(orig_flame_tri_uvs, orig_flame_ver, orig_flame_faces, 
    #                         orig_flame_seam_ver_mask, orig_flame_vertex_uvs_prelim)
    # orig_flame_faces_t = torch.tensor(orig_flame_faces, device=device).long()
    # orig_flame_verts_t = torch.tensor(orig_flame_verts_upd, device=device).float()
    # orig_flame_vertex_uvs_t = torch.tensor(orig_flame_vertex_uvs, device=device).float()
    # orig_flame_vertex_uvs_t = torch.cat([orig_flame_vertex_uvs_t, 
    #                                     torch.zeros(orig_flame_vertex_uvs_t.shape[0], 1, device=device)], dim=1)    # (n_ver, 3)

    # mesh = Meshes(verts=[orig_flame_vertex_uvs_t], faces=[orig_flame_faces_t])

    # fragments_orig_flame_std_layout = rasterizer(mesh)

    # pix_to_face = fragments_orig_flame_std_layout.pix_to_face.cpu().data.numpy()    # (1, H, W, 2)
    # bary_coords = fragments_orig_flame_std_layout.bary_coords.cpu().data.numpy()    # (N, H, W, 2, 3)

    pix_to_face, bary_coords = get_bary_face_inds(orig_flame_ver, orig_flame_faces, orig_flame_tri_uvs,
                                                  res=res, device=device)

    # since some UVs of the dense mesh can fall 1 pixel out of the rendered region,
    # we apply distance transform to pix_to_face and bary_coords to fill in the background with the nearest values
    std_layout_uv_mask = pix_to_face[0, ..., 0] != -1
    _, edt_indices = distance_transform_edt(std_layout_uv_mask == 0, return_indices=True)
    pix_to_face_dt = pix_to_face[0, ..., 0][edt_indices[0], edt_indices[1]]
    bary_coords_dt = bary_coords[0][edt_indices[0], edt_indices[1]]

    # querying the barycentric coords and face indices of the standard template (coarse mesh) 
    #  in the dense mesh vertex locations

    uvs_for_grid_sample = rearrange(uvs, 'i j k -> (i j) k')
    dense_uvs_orig_flame_faces = sample_map_with_coords(pix_to_face_dt[..., None], 
                                                        uvs_for_grid_sample, 
                                                        mode='nearest').astype(int)[:, 0]
    dense_uvs_orig_flame_bary  = sample_map_with_coords(bary_coords_dt[ ..., 0, :], 
                                                        uvs_for_grid_sample, 
                                                        # mode='bilinear')
                                                        mode='nearest')
    dense_uvs_custom_layout = (
        dense_uvs_orig_flame_bary[:, [0]] * mod_tri_uvs[dense_uvs_orig_flame_faces][:, 0]
        + dense_uvs_orig_flame_bary[:, [1]] * mod_tri_uvs[dense_uvs_orig_flame_faces][:, 1]
        + dense_uvs_orig_flame_bary[:, [2]] * mod_tri_uvs[dense_uvs_orig_flame_faces][:, 2]
    )
    dense_uvs_custom_layout = rearrange(dense_uvs_custom_layout, '(n c) d -> n c d', c=3)
    
    # similarly to the standard layout, we find the seam vertices and duplicate them
    dense_uvs_custom_layout_seam_ver_mask, dense_vertex_uvs_custom_layout_prelim = \
        find_seam_ver(dense_uvs_custom_layout, ver_upd.shape[0], faces)
    dense_uvs_custom_layout_verts_upd, dense_vertex_uvs_custom_layout, dense_uvs_custom_layout_faces_upd, _ = \
        duplicate_seam_ver(dense_uvs_custom_layout, ver_upd, faces, 
                           dense_uvs_custom_layout_seam_ver_mask, dense_vertex_uvs_custom_layout_prelim)

    # in the custom layout, some of the triangles may correspond to the triangles spanning through the new seam
    #  (since the original triangle_uvs were created with the original seam in mind and don't know the new seam location).
    # we eliminate these stretched triangles from the rendering since their omission won't influence the result in any case
    #  but leaving them would cause the rendering mistakes.
    stretched_mask = find_stretched_triangles(dense_uvs_custom_layout, seam_thr=0.15)    # the seam_thr is adjusted to our custom layout

    return (
        dense_uvs_custom_layout, 
        dense_uvs_custom_layout_verts_upd, 
        dense_vertex_uvs_custom_layout, 
        dense_uvs_custom_layout_faces_upd, 
        stretched_mask,
        dense_uvs_orig_flame_faces,
        dense_uvs_orig_flame_bary
    )


def uv_map_to_3d(uvmap, 
                 dense_uvs_custom_layout_verts_upd,
                 dense_vertex_uvs_custom_layout,
                 dense_uvs_custom_layout_faces_upd,
                 uv_layout_type='custom_layout_v1',
                 uv_mask_inner_mouth=None,
                 smooth_seam=True,
                 seam_ver_mask=None,
                 n_lapl_iter=10,
                 device='cuda:0'):
    
    # uvmap: numpy array (res, res, 3)   (for custom_layout_2+, already shrunk to square)

    h, w = uvmap.shape[:2]

    # setting eyeballs offsets to zero
    # if uv_layout_type == 'custom_layout_v1':
    #     uvmap[:int(0.05 * h), -int(0.165 * w):] = 0

    uvmap_t = torch.tensor(uvmap)
    uvmap_t = rearrange(uvmap_t, 'h w c -> 1 c h w')

    if uv_mask_inner_mouth is not None:
        uvmap *= (uv_mask_inner_mouth < 0.5).astype(float)

    # uvmap_orig *= (uv_mask > 0.5).astype(float)

    out = sample_map_with_coords(uvmap, dense_vertex_uvs_custom_layout, mode='bilinear')

    print('out:', out)
    
    # out[ver_subdiv_belong_inner_mouth] = 0

    mesh_raw_p3d = Meshes(verts=[torch.tensor(dense_uvs_custom_layout_verts_upd + out).to(device).float()],
                          faces=[torch.tensor(dense_uvs_custom_layout_faces_upd).to(device)])
    if not smooth_seam:
        mesh = mesh_raw_p3d
    else:
        mesh_lapl = deepcopy(mesh_raw_p3d)
        L = mesh_lapl.laplacian_packed()
        seam_ver_mask_t = torch.tensor(seam_ver_mask, dtype=bool, device=device)

        # applying the laplacian smoothing
        mesh_lapl_ver = mesh_lapl.verts_packed()
        L = mesh_lapl.laplacian_packed()
        for _ in range(n_lapl_iter):
            mesh_lapl_ver[seam_ver_mask_t] += L.mm(mesh_lapl_ver)[seam_ver_mask_t]    # in-place modification
        mesh = mesh_lapl

    return mesh


def consistent_subdiv(coarse_ver, coarse_faces, n_subdiv_ver, subdiv_faces, sample_subdiv_which_tri, sample_subdiv_barycentric):
    # coarse_ver: (n_coarse_ver, 3)
    # coarse_faces: (n_coarse_faces, 3)
    # n_subdiv_ver: int
    # subdiv_faces: (n_subdiv_tri, 3)
    # sample_subdiv_which_tri: (n_subdiv_tri * 3, 3) -- which triangle j-th point of the i-th sample subdiv triangle belongs to,
    #    j=1..n_subdiv_tri, i=1..3
    # sample_subdiv_barycentric: (n_subdiv_tri * 3, 3) -- barycentric coords of j-th point of the i-th sample subdiv triangle in the space of coarse_faces,
    #    j=1..n_subdiv_tri, i=1..3
    
    sample_subdiv_ver_inds_of_the_corr_orig_tri = coarse_faces[sample_subdiv_which_tri]
    
    sample_subdiv_ver_of_the_corr_orig_tri = coarse_ver[sample_subdiv_ver_inds_of_the_corr_orig_tri]
    
    uv2dist_to_first, uv2dist_to_second, uv2dist_to_third = np.split(sample_subdiv_barycentric, 3, axis=-1)

    # coordinates of each (tri, ver) pair (we do like this first to handle seams accurately)
    tri_ver_subdiv = (
        (sample_subdiv_ver_of_the_corr_orig_tri[:, 0] * uv2dist_to_first
         + sample_subdiv_ver_of_the_corr_orig_tri[:, 1] * uv2dist_to_second
         + sample_subdiv_ver_of_the_corr_orig_tri[:, 2] * uv2dist_to_third)
        /
        (uv2dist_to_first
         + uv2dist_to_second
         + uv2dist_to_third)
    )
    
    tri_ver_subdiv = rearrange(tri_ver_subdiv, '(f d) c -> f d c', d=3)

    # now can just set the vertices based on tri_ver_subdiv -- 
    # -- even for the "duplicate" seam vertices, the 3d coordinates should be the same.
    ver_subdiv = np.full((n_subdiv_ver, 3), -1.0)
    ver_subdiv[subdiv_faces[:, 2]] = tri_ver_subdiv[:, 2]
    ver_subdiv[subdiv_faces[:, 0]] = tri_ver_subdiv[:, 0]
    ver_subdiv[subdiv_faces[:, 1]] = tri_ver_subdiv[:, 1]
    # ver_subdiv[subdiv_faces[:, 0]] = tri_ver_subdiv[:, 0]
    # ver_subdiv[subdiv_faces[:, 1]] = tri_ver_subdiv[:, 1]
    # ver_subdiv[subdiv_faces[:, 2]] = tri_ver_subdiv[:, 2]
    
    return ver_subdiv


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Unconditional sampling from the trained StyleGAN + conversion of the result to 3D meshes.')
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file.')
    parser.add_argument('--ckpt_path_ema', type=str, default=None, help='Path to the EMA checkpoint file (optional). If provided, the EMA model will be used for sampling, but --ckpt_path must also be provided (its weights will not matter but it is used for initialization). Using the EMA checkpoint is highly recommended for better results.')
    parser.add_argument('--output_path', type=str, help='Output directory for the generated samples.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate.')
    # parser.add_argument('--uv_layout_type', type=str, default='standard_layout', help='Type of the UV layout that was used for StyleGAN training.')
    parser.add_argument('--uv_layout_type', type=str, default='custom_layout', help='Type of the UV layout that was used for StyleGAN training.')
    parser.add_argument('--truncation_psi', type=float, default=0.7, help='Truncation psi.')
    parser.add_argument('--sample_flame', action='store_true', help='Sample the underlying FLAME template randomly. Otherwise, constant FLAME will be used.')
    parser.add_argument('--gt_scaling_coef', type=float, default=120.0, help='Scaling coefficient to match flame template scale to the right offsets scale. Only relevant when --sample_flame.')
    parser.add_argument('--orig_flame_tri_uvs_path', type=str, default=None, help='path to the original flame triangle UVs')
    parser.add_argument('--custom_layout_mesh_path', type=str, default=None, help='path to the custom layout mesh')
    parser.add_argument('--save_flame', action='store_true', help='Save underlying template as a separate file.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for generation.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--to_3d', action='store_true', help='Convert the samples to 3D meshes.')

    args = parser.parse_args()

    # Load the model
    ckpt_path = args.ckpt_path
    ckpt_path_ema = args.ckpt_path_ema

    # if args.uv_layout_type == 'standard_layout':
    #     from trainer.train_stylegan import StyleGAN2Trainer
    # elif args.uv_layout_type == 'custom_layout_v1':
    #     from trainer.train_stylegan_uv2offset import StyleGAN2Trainer
    # elif args.uv_layout_type == 'custom_layout_v2':
    #     from trainer.train_stylegan_uv2offset_layout_v2 import StyleGAN2Trainer
    # elif args.uv_layout_type == 'custom_layout_v3':
    #     from trainer.train_stylegan_uv2offset_layout_v2 import StyleGAN2Trainer

    # if args.uv_layout_type == 'standard_layout':
    #     from trainer.train_stylegan import StyleGAN2Trainer
    # elif args.uv_layout_type == 'custom_layout':
    #     from trainer.train_stylegan_uv2offset_layout_v2 import StyleGAN2Trainer

    from trainer.train_stylegan import StyleGAN2Trainer

    model = StyleGAN2Trainer.load_from_checkpoint(ckpt_path)
    model.on_train_start()

    if ckpt_path_ema is not None:
        ckpt_ema = torch.load(ckpt_path_ema, map_location=args.device)
        model.ema.load_state_dict(ckpt_ema)
        # model.ema.copy_to([p for p in model.G.parameters() if p.requires_grad])
        model.ema.copy_to([p for p in model.G.parameters()])

    _ = model.G.eval()
    model = model.to(args.device)

    # if args.uv_layout_type == 'standard_layout':
    #     noise_type = 'const'
    # elif args.uv_layout_type == 'custom_layout_v1':
    #     noise_type = 'const-none'    # TODO change to const-none-custom-layout-v1
    # elif args.uv_layout_type == 'custom_layout_v2':
    #     noise_type = 'const'    # we're doing two passes in this case: one with const and one with none -- and blending the result.
    #     # noise_type = 'none'    # NOTE DEBUG
    # elif args.uv_layout_type == 'custom_layout_v3':
    #     noise_type = 'const'    # we're doing two passes in this case: one with const and one with none -- and blending the result.
    # else:
    #     raise ValueError(f'Unknown UV layout type: {args.uv_layout_type}')
    noise_type = 'const'

    if args.to_3d:
        
        # loading standard layout and custom layout meshes
        orig_flame_tri_uvs_fn = args.orig_flame_tri_uvs_path
        custom_layout_mesh_fn = args.custom_layout_mesh_path
        if orig_flame_tri_uvs_fn is None:
            orig_flame_tri_uvs_fn = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'flame_triangle_uvs.npy')
        if custom_layout_mesh_fn is None:
            # if args.uv_layout_type == 'custom_layout_v1':
            #     # custom_layout_mesh_fn = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'flame_uv_mod.obj')
            #     custom_layout_mesh_fn =  '/rhome/asevastopolsky/source/uv-to-offset/data/flame_uv_mod1.obj'    #TODO temporarily hardcoded, replace with the proper path!
            # elif args.uv_layout_type == 'custom_layout_v2':
            #     custom_layout_mesh_fn =  '/rhome/asevastopolsky/source/uv-to-offset/data/custom_seam2_uv_transformed.obj'    #TODO temporarily hardcoded, replace with the proper path!
            # elif args.uv_layout_type == 'custom_layout_v3':
            #     custom_layout_mesh_fn =  '/rhome/asevastopolsky/source/uv-to-offset/data/custom_seam3_uv_transformed.obj'
            custom_layout_mesh_fn = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'flame_uv_mod.obj')
        orig_flame_tri_uvs = np.load(orig_flame_tri_uvs_fn)
        
        mod_flame_mesh = load_objs_as_meshes([custom_layout_mesh_fn], device='cpu')
        faces_to_vt_inds = mod_flame_mesh.textures.faces_uvs_list()[0]
        vt = mod_flame_mesh.textures.verts_uvs_list()[0]
        mod_tri_uvs = vt[faces_to_vt_inds]
        mod_tri_uvs = mod_tri_uvs.data.numpy()
        orig_flame_ver = mod_flame_mesh.verts_packed().cpu().data.numpy()
        orig_flame_faces = mod_flame_mesh.faces_packed().cpu().data.numpy()

        # loading sample subdivided ver, faces, triangle UVs
        #TODO replace with the paths from the data folder that we're going to provide
        uv_tri_subdiv = np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'layout', 'template_decim_uv.npy'))
        faces_subdiv = np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'layout', 'template_decim_faces.npy'))
        ver_subdiv = np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'layout', 'template_decim_ver.npy'))    # not using vertices per se, that's just for the calculate the number of them
        n_ver_subdiv = ver_subdiv.shape[0]

        dense_uvs_custom_layout, \
            dense_uvs_custom_layout_verts_upd, \
            dense_vertex_uvs_custom_layout, \
            dense_uvs_custom_layout_faces_upd, \
            stretched_mask, \
            dense_uvs_orig_flame_faces, \
            dense_uvs_orig_flame_bary = uvs_to_custom_layout(ver_subdiv, faces_subdiv, uv_tri_subdiv,
                                                             orig_flame_ver, orig_flame_faces, orig_flame_tri_uvs,
                                                             mod_tri_uvs, 
                                                            #  res=512,
                                                            #  res=2048,
                                                             res=2048,
                                                             device='cuda:0')

        seam_ver_mask = np.zeros(dense_uvs_custom_layout_verts_upd.shape[0], dtype=bool)
        seam_ver_mask[dense_uvs_custom_layout_faces_upd[stretched_mask].ravel()] = True
        dense_uvs_custom_layout_edges_upd = dense_uvs_custom_layout_faces_upd[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        # n_dilate_iter = 2
        n_dilate_iter = 10
        for _ in range(n_dilate_iter):
            seam_ver_mask = dilate_vertex_mask(dense_uvs_custom_layout_edges_upd, seam_ver_mask)
        
        # if args.uv_layout_type == 'custom_layout_v1':
        #     uv_mask_inner_mouth = imageio.imread('/rhome/asevastopolsky/source/uv-to-offset/data/flame_inner_mouth_custom_layout1.png')    #TODO replace with the proper path!
        #     uv_mask_inner_mouth = uv_mask_inner_mouth.astype(float) / 255  
        # elif args.uv_layout_type == 'custom_layout_v2':
        #     # uv_mask_inner_mouth = imageio.imread('/rhome/asevastopolsky/source/uv-to-offset/data/flame_inner_mouth_custom_layout2.png')
        #     uv_mask_inner_mouth = np.zeros((256, 256, 3), dtype=float)
        # elif args.uv_layout_type == 'custom_layout_v3':
        #     uv_mask_inner_mouth = np.zeros((256, 256, 3), dtype=float)

        uv_mask_inner_mouth = np.zeros((256, 256, 3), dtype=float)
    
        if args.sample_flame:
            
            flame_config_fn = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'flame_config_for_sampling.yaml')
            sample_flame_fn = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'sample_flame.ply')

            # loading FLAME lib
            with open(flame_config_fn, "r") as yamlfile:
                conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
                config = OmegaConf.create(conf)

            _flame = FLAME(config)
            _flame = _flame.to(args.device)

            _, sample_flame_mesh_faces = load_ply(sample_flame_fn)
            flame_faces = sample_flame_mesh_faces.cpu().data.numpy()
            flame_edges = flame_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
            # leave only unique edges:
            flame_edges = np.sort(flame_edges, axis=1)
            flame_edges = np.unique(flame_edges, axis=0)

            shape_mean = torch.tensor(np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'nersemble_statistics', 'nersemble_flame_shape_mean.npy'))).to(args.device)
            shape_std = torch.tensor(np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'nersemble_statistics', 'nersemble_flame_shape_std.npy'))).to(args.device)
            expr_mean = torch.tensor(np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'nersemble_statistics', 'nersemble_flame_expr_mean.npy'))).to(args.device)
            expr_std = torch.tensor(np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'nersemble_statistics', 'nersemble_flame_expr_std.npy'))).to(args.device)
            jaw_mean = torch.tensor(np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'nersemble_statistics', 'nersemble_flame_jaw_mean.npy'))).to(args.device)
            jaw_std = torch.tensor(np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'nersemble_statistics', 'nersemble_flame_jaw_std.npy'))).to(args.device)

            cache_dense_uvs_custom_layout_seam_ver_mask = None
            cache_dense_vertex_uvs_custom_layout_prelim = None

            # for parts smoothing
            with open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'flame_masks.pkl'), 'rb') as fin:
                flame_region_masks = pk.load(fin)

            kwargs = {
                'region_masks': flame_region_masks,
                'faces': flame_faces,
                'edges': flame_edges,
                'uvs_which_orig_tri': dense_uvs_orig_flame_faces,
                'n_ver_subdiv': n_ver_subdiv,
                'faces_subdiv': faces_subdiv,
            }
            ver_subdiv_belong_face = find_subdiv_ver_in_region('face', **kwargs)
            ver_subdiv_belong_inner_mouth = find_subdiv_ver_in_region('inner_mouth', dilate_mask=2, **kwargs)    # this call requires 'inner_mouth' mask to be present in flame_masks.pkl which is only available in the file shipped with the repo, not in the one shipped with FLAME.
            ver_subdiv_belong_scalp_neck = find_subdiv_ver_in_region('scalp', **kwargs) | find_subdiv_ver_in_region('neck', dilate_mask=2, **kwargs)
            ver_subdiv_belong_face_features = (
                find_subdiv_ver_in_region('mouth', **kwargs)
                | find_subdiv_ver_in_region('lips', **kwargs)
                | find_subdiv_ver_in_region('left_eyeball', **kwargs)
                | find_subdiv_ver_in_region('right_eyeball', **kwargs)
                | find_subdiv_ver_in_region('left_eye_region', **kwargs)
                | find_subdiv_ver_in_region('right_eye_region', **kwargs)
            )
            ver_subdiv_belong_lips = find_subdiv_ver_in_region('lips', **kwargs)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate samples and save them to the output directory
    # out = []
    out_dir = args.output_path
    os.makedirs(os.path.join(out_dir, 'uv_maps'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'meshes'), exist_ok=True)

    batch_size = 1    # the only supported value for model.latent(...) at the moment
    # n_lapl_iter = 1
    n_lapl_iter = 10

    for i in tqdm(range(args.num_samples)):
        with torch.no_grad():
            z = model.latent(batch_size)
            z = (z[0] * args.truncation_psi, z[1] * args.truncation_psi)
            w = model.get_mapped_latent(z, style_mixing_prob=0.9)    # style_mixing_prob can be omitted here
            cur_out = model.G.synthesis(w, noise_mode=noise_type)    # (1, 6, h, w)
            if args.uv_layout_type in ('custom_layout',):
                # this block works for our layout that consists of a separate scalp and face part
                cur_out_noise_none = model.G.synthesis(w, noise_mode='none')    # (1, 6, h, w)
                cur_out = torch.cat([cur_out_noise_none[:, :3], cur_out[:, 3:]], dim=1)    # face from none, scalp from const
        
        # out.append(cur_out.cpu().data.numpy())
        pred = cur_out.cpu().data.numpy()

        # Save the samples
        pred = rearrange(pred, '1 c h w -> h w c')
        np.save(os.path.join(out_dir, 'uv_maps', f'{i:06}.npy'), pred)

        # Save the version for visualization
        vis_min, vis_max = -1, 1
        uv_img_offset_vis = (np.clip(pred, vis_min, vis_max) - vis_min) / (vis_max - vis_min + 1e-10)
        if args.uv_layout_type in ('custom_layout',):
            # this block works for our layout that consists of a separate scalp and face part
            uv_img_offset_vis = np.concatenate([uv_img_offset_vis[:, :, :3], uv_img_offset_vis[:, :, 3:]], axis=1)    # (h, w, 6) -> (h, 2*w, 3)
        imageio.imwrite(os.path.join(out_dir, 'uv_maps', f'{i:06}_vis.jpg'), 
                        (uv_img_offset_vis * 255).astype(np.uint8))
        
        if args.to_3d:

            if args.sample_flame:
                shape = (torch.randn((1, 300)).float().to(args.device) - shape_mean) * shape_std
                exp = (torch.randn((1, 100)).float().to(args.device) - expr_mean) * expr_std * 3
                # exp = (torch.tensor(sample_exp).unsqueeze(0).float().to(device)) 
                pose = torch.zeros((1, 6)).float().to(args.device)
                pose[:, 3:] = (torch.randn((1, 3)).float().to(args.device) - jaw_mean) * jaw_std

                novel_ver = _flame(shape_params=shape, 
                                expression_params=exp,
                                pose_params=pose)[0].cpu().data.numpy()
                
                # dense_uvs_orig_flame_ver_inds = flame_faces[dense_uvs_orig_flame_faces] 
                # dense_uvs_orig_flame_ver = novel_ver[dense_uvs_orig_flame_ver_inds]

                # novel_ver_subdiv = (
                #     (dense_uvs_orig_flame_ver[:, 0] * dense_uvs_orig_flame_bary[:, [0]]
                #     + dense_uvs_orig_flame_ver[:, 1] * dense_uvs_orig_flame_bary[:, [1]]
                #     + dense_uvs_orig_flame_ver[:, 2] * dense_uvs_orig_flame_bary[:, [2]])
                #     /
                #     (dense_uvs_orig_flame_bary[:, [0]]
                #     + dense_uvs_orig_flame_bary[:, [1]]
                #     + dense_uvs_orig_flame_bary[:, [2]])
                # )

                # dense_uvs_custom_layout_verts_upd = novel_ver_subdiv

                # novel_ver_subdiv = consistent_subdiv(novel_ver, flame_faces, 
                #                                     #  n_ver_subdiv, 
                #                                      dense_uvs_custom_layout_verts_upd.shape[0],
                #                                     #  faces_subdiv, dense_uvs_orig_flame_faces, dense_uvs_orig_flame_bary)
                #                                      dense_uvs_custom_layout_faces_upd, dense_uvs_orig_flame_faces, dense_uvs_orig_flame_bary)

                novel_ver_subdiv = consistent_subdiv(novel_ver, flame_faces, 
                                                    #  n_ver_subdiv, 
                                                     n_ver_subdiv,
                                                    #  faces_subdiv, dense_uvs_orig_flame_faces, dense_uvs_orig_flame_bary)
                                                     faces_subdiv, dense_uvs_orig_flame_faces,dense_uvs_orig_flame_bary)
                
                novel_subdiv_mesh_p3d = Meshes(verts=torch.tensor(novel_ver_subdiv).to(args.device).unsqueeze(0), 
                                            faces=torch.tensor(faces_subdiv).to(args.device).unsqueeze(0))
                L = novel_subdiv_mesh_p3d.laplacian_packed()
                novel_ver_subdiv_t = torch.tensor(novel_ver_subdiv).to(args.device).float()
                for _ in range(10):
                    novel_ver_subdiv_t[ver_subdiv_belong_scalp_neck] += L.mm(novel_ver_subdiv_t)[ver_subdiv_belong_scalp_neck]
                for _ in range(5):
                    novel_ver_subdiv_t[(ver_subdiv_belong_face & (~ver_subdiv_belong_face_features))] \
                        += L.mm(novel_ver_subdiv_t)[(ver_subdiv_belong_face & (~ver_subdiv_belong_face_features))]
                for _ in range(3):
                    novel_ver_subdiv_t[ver_subdiv_belong_lips] \
                        += L.mm(novel_ver_subdiv_t)[ver_subdiv_belong_lips]
                novel_ver_subdiv = novel_ver_subdiv_t.cpu().data.numpy()

                if cache_dense_uvs_custom_layout_seam_ver_mask is None:
                    # similarly to the standard layout, we find the seam vertices and duplicate them
                    dense_uvs_custom_layout_seam_ver_mask, dense_vertex_uvs_custom_layout_prelim = \
                        find_seam_ver(dense_uvs_custom_layout, n_ver_subdiv, faces_subdiv)
                    cache_dense_uvs_custom_layout_seam_ver_mask = dense_uvs_custom_layout_seam_ver_mask.copy()
                    cache_dense_vertex_uvs_custom_layout_prelim = dense_vertex_uvs_custom_layout_prelim.copy()
                else:
                    dense_uvs_custom_layout_seam_ver_mask = cache_dense_uvs_custom_layout_seam_ver_mask.copy()
                    dense_vertex_uvs_custom_layout_prelim = cache_dense_vertex_uvs_custom_layout_prelim.copy()
                
                # dense_uvs_custom_layout_verts_upd, dense_vertex_uvs_custom_layout, dense_uvs_custom_layout_faces_upd, _ = \
                dense_uvs_custom_layout_verts_upd, dense_vertex_uvs_custom_layout, dense_uvs_custom_layout_faces_upd, _ = \
                    duplicate_seam_ver(dense_uvs_custom_layout, novel_ver_subdiv, faces_subdiv, 
                                       dense_uvs_custom_layout_seam_ver_mask, dense_vertex_uvs_custom_layout_prelim)
                
                dense_uvs_custom_layout_verts_upd = dense_uvs_custom_layout_verts_upd.copy() * args.gt_scaling_coef
            
            # if args.uv_layout_type in ('custom_layout_v3',):
            if args.uv_layout_type in ('custom_layout',):
                # this block works for our layout that consists of a separate scalp and face part
                pred = np.concatenate([pred[:, :, :3], pred[:, :, 3:]], axis=1)    # (h, w, 6) -> (h, 2*w, 3)
                pred = (pred[:, ::2] + pred[:, 1::2]) / 2    # (h, 2*w, 3) -> (h, w, 3)

            # Additional face smoothing:
            hor_thr = 0.5
            # average pooling to twice lower res
            uvmap_ds = (pred[::2, ::2] + pred[1::2, ::2] + pred[::2, 1::2] + pred[1::2, 1::2]) / 4
            # upsampling back via bilinear interpolation from scipy
            from scipy.ndimage import zoom
            uvmap_ds_up = zoom(uvmap_ds, (2, 2, 1), order=1)
            # replace the face part
            pred[:, :int(hor_thr * uvmap_ds_up.shape[1])] = uvmap_ds_up[:, :int(hor_thr * uvmap_ds_up.shape[1])]
            
            mesh = uv_map_to_3d(pred,
                                dense_uvs_custom_layout_verts_upd,
                                dense_vertex_uvs_custom_layout,
                                dense_uvs_custom_layout_faces_upd,
                                uv_layout_type=args.uv_layout_type,
                                uv_mask_inner_mouth=uv_mask_inner_mouth,
                                smooth_seam=True,
                                seam_ver_mask=seam_ver_mask,
                                n_lapl_iter=n_lapl_iter,
                                device=args.device)
            save_ply(os.path.join(out_dir, 'meshes', f'{i:06}.ply'), mesh.verts_packed(), mesh.faces_packed())

            if args.save_flame:
                save_ply(os.path.join(out_dir, 'meshes', f'{i:06}_template.ply'), 
                         torch.tensor(dense_uvs_custom_layout_verts_upd),
                         torch.tensor(dense_uvs_custom_layout_faces_upd))
        
        del cur_out
        torch.cuda.empty_cache()    # might be not necessary and can affect the performance -- check if it's the case

