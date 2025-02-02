from glob import glob
import os
from copy import deepcopy
from collections import defaultdict
import sys
from tqdm import tqdm
import imageio
import numpy as np
from scipy.ndimage import distance_transform_edt
from einops import rearrange, repeat
import torch
import torch.nn as nn

from pytorch3d.renderer import (
    look_at_view_transform,
    TexturesVertex
)

from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes

# To be able to import other modules from the project:
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from headcraft.utils.mesh import find_seam_ver, duplicate_seam_ver, find_stretched_triangles
from headcraft.utils.uv import sample_map_with_coords
from headcraft.utils.renderer import make_ortho_renderer


def process_standard_layout(ver_upd, faces, uvs, res=2048, device='cuda:0'):

    # rendering vertices to UV space

    center = torch.tensor([[0, 0, 0]], device=device).float()
    R, T = look_at_view_transform(1, 180, 180, at=center, device=device)
    renderer = make_ortho_renderer(R, T, device=device, res=res)

    seam_ver_mask, vertex_uvs_prelim = find_seam_ver(uvs, ver_upd.shape[0], faces)
    new_verts, new_vertex_uvs, new_faces, ver2new_ver = \
        duplicate_seam_ver(uvs, ver_upd, faces, seam_ver_mask, vertex_uvs_prelim)

    faces_t = torch.tensor(new_faces, device=device).long()
    verts_t = torch.tensor(new_verts, device=device).float()
    vertex_uvs_t = torch.tensor(new_vertex_uvs, device=device).float()
    vertex_uvs_t = torch.cat([vertex_uvs_t, torch.zeros(vertex_uvs_t.shape[0], 1, device=device)], dim=1)    # (n_ver, 3)

    textures = TexturesVertex(verts_features=[verts_t])

    mesh = Meshes(verts=[vertex_uvs_t], faces=[faces_t], textures=textures)

    image = renderer(mesh)
    image = image[0, ..., :3].cpu().data.numpy()

    return image


def process_custom_layout(ver_upd, faces, uvs, 
                          orig_flame_ver, orig_flame_faces, orig_flame_tri_uvs,
                          mod_tri_uvs,
                          res=2048, device='cuda:0', 
                          reuse_dense_uvs=None):

    # rendering vertices to UV space
    center = torch.tensor([[0, 0, 0]], device=device).float()
    R, T = look_at_view_transform(1, 180, 180, at=center, device=device)

    if reuse_dense_uvs is not None:
        dense_uvs_custom_layout = reuse_dense_uvs
    else:
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

    # rendering the vertices in the custom layout with obtained dense_uvs_custom_layout
    renderer = make_ortho_renderer(R, T, device=device, res=res)

    faces_t = torch.tensor(dense_uvs_custom_layout_faces_upd, device=device).long()
    faces_t = faces_t[~stretched_mask]
    verts_t = torch.tensor(dense_uvs_custom_layout_verts_upd, device=device).float()
    vertex_uvs_custom_t = torch.tensor(dense_vertex_uvs_custom_layout, device=device).float()
    vertex_uvs_custom_t = torch.cat([vertex_uvs_custom_t, 
                                    torch.zeros(vertex_uvs_custom_t.shape[0], 1, device=device)], dim=1)    # (n_ver, 3)

    textures = TexturesVertex(verts_features=[verts_t])

    mesh = Meshes(verts=[vertex_uvs_custom_t], faces=[faces_t], textures=textures)

    image = renderer(mesh)
    image = image[0, ..., :3].cpu().data.numpy()
    return image, dense_uvs_custom_layout


def save_uint16(img, path_prefix):
    vis_min, vis_max = -20, 20
    img_vis = (np.clip(img, vis_min, vis_max) - vis_min) / (vis_max - vis_min + 1e-10)

    for ch_no in range(3):
        imageio.imsave(path_prefix + f'_{vis_min}_to_{vis_max}_ch{ch_no}.png', 
                       (img_vis[..., ch_no] * 65536).astype(np.uint16))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    # most important args
    parser.add_argument('--root_dir', type=str, 
                        help='path to the dataset folder that contains data in a structure "<root_dir>/<seq_name>/<expr_name>/subdivided/template_decim_ver.npy", "<root_dir>/<seq_name>/<expr_name>/estimated_normal_offsets/\{faces.npy,uvs.npy,ver_upd.npy\}"')
    parser.add_argument('--output_root_dir', type=str,
                        help='path to the folder where all the processed data will be saved (in "<output_root_dir>/<seq_name>/<expr_name>/estimated_normal_offsets/").')
    parser.add_argument('--res', type=int, default=1024)
    parser.add_argument('--uv_layout_type', type=str, default='standard_layout', help='type of the UV layout to use')
    parser.add_argument('--use_mask', action='store_true', default=False, help='access vertex masks "<root_dir>/<seq_name>/<expr_name>/estimated_normal_offsets/final_mask.npy" to mask out some parts of the UV map. Can be useful when offsets were registered to only a part of a mesh/cloud.')

    # optional: processing only a part of the dataset (useful when e.g. parallelizing onto several GPUs)
    parser.add_argument('--n_parts', type=int, default=None, help='whether to process a dataset in a few parts in parallel')
    parser.add_argument('--part_idx', type=int, default=None, help='index of the part to process; makes sense to use only with --n_parts')

    # optional: important paths
    parser.add_argument('--orig_flame_tri_uvs_path', type=str, default=None, help='path to the original flame triangle UVs')
    parser.add_argument('--custom_layout_mesh_path', type=str, default=None, help='path to the custom layout mesh')

    # optional: individual optimization params
    parser.add_argument('--ignore_existing', action='store_true')

    args = parser.parse_args()

    all_inp = glob(os.path.join(args.root_dir, '*', '*', 'estimated_normal_offsets', 'ver_upd.npy'))
    all_inp = list(sorted(all_inp))
    print(len(all_inp))

    if args.n_parts is not None and args.part_idx is not None:
        all_inp = all_inp[args.part_idx::args.n_parts]
        print('processing part', args.part_idx, 'of', args.n_parts, f'({len(all_inp)} samples)')
    
    # loading the layout
    
    orig_flame_tri_uvs_fn = args.orig_flame_tri_uvs_path
    custom_layout_mesh_fn = args.custom_layout_mesh_path
    if orig_flame_tri_uvs_fn is None:
        orig_flame_tri_uvs_fn = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'flame', 'flame_triangle_uvs.npy')
    if custom_layout_mesh_fn is None:
        custom_layout_mesh_fn = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'layout', 'flame_uv_mod.obj')
    orig_flame_tri_uvs = np.load(orig_flame_tri_uvs_fn)

    mod_flame_mesh = load_objs_as_meshes([custom_layout_mesh_fn], device='cpu')
    faces_to_vt_inds = mod_flame_mesh.textures.faces_uvs_list()[0]
    vt = mod_flame_mesh.textures.verts_uvs_list()[0]
    mod_tri_uvs = vt[faces_to_vt_inds]
    mod_tri_uvs = mod_tri_uvs.data.numpy()

    orig_flame_ver = mod_flame_mesh.verts_packed().cpu().data.numpy()
    orig_flame_faces = mod_flame_mesh.faces_packed().cpu().data.numpy()

    dense_uvs_custom_layout = None

    for ver_path in tqdm(all_inp):

        # loading template
        seq_name = ver_path.split(os.sep)[-4]
        exp_name = ver_path.split(os.sep)[-3]
        print(seq_name)

        save_name_suffix = f'_{args.uv_layout_type}.npy'

        faces = np.load(os.sep.join(ver_path.split(os.sep)[:-1] + ['faces.npy']))
        uvs = np.load(os.sep.join(ver_path.split(os.sep)[:-1] + ['uvs.npy']))

        # output_mode: updated vertices
        ver_upd = np.load(ver_path)
        
        out_basename = 'uv_map_ver_upd' + save_name_suffix

        output_path = os.sep.join(ver_path.split(os.sep)[:-1] + [out_basename])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if args.ignore_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print('skipping', output_path)
        else:
            if args.uv_layout_type == 'standard_layout':
                uv_img_ver_upd = process_standard_layout(ver_upd, faces, uvs, 
                                                         res=args.res)
            else:
                uv_img_ver_upd, dense_uvs_custom_layout = process_custom_layout(
                    ver_upd, faces, uvs,
                    orig_flame_ver=orig_flame_ver,
                    orig_flame_faces=orig_flame_faces,
                    orig_flame_tri_uvs=orig_flame_tri_uvs,
                    res=args.res,
                    mod_tri_uvs=mod_tri_uvs,
                )
            np.save(output_path, uv_img_ver_upd)    # TODO replace with uint16 saving

        # output mode: template vertices
        ver_template = np.load(os.sep.join(ver_path.split(os.sep)[:-2] + ['subdivided', 'template_decim_ver.npy']))
        out_basename = 'uv_map_template' + save_name_suffix
        output_path = os.sep.join(ver_path.split(os.sep)[:-1] + [out_basename])

        if args.ignore_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print('skipping', output_path)
        else:
            if args.uv_layout_type == 'standard_layout':
                uv_img_template = process_standard_layout(ver_template, faces, uvs, 
                                                          res=args.res)
            else:
                uv_img_template, _ = process_custom_layout(
                    ver_template, faces, uvs,
                    orig_flame_ver=orig_flame_ver,
                    orig_flame_faces=orig_flame_faces,
                    orig_flame_tri_uvs=orig_flame_tri_uvs,
                    mod_tri_uvs=mod_tri_uvs,
                    res=args.res, 
                    reuse_dense_uvs=dense_uvs_custom_layout
                )
            np.save(output_path, uv_img_template)
        
        # output mode: offsets
        out_basename = 'uv_map_offset' + save_name_suffix
        output_path = os.sep.join(ver_path.split(os.sep)[:-1] + [out_basename])

        if args.ignore_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print('skipping', output_path)
        else:
            uv_img_offset = uv_img_ver_upd - uv_img_template

            if args.use_mask:
                mask_path = os.sep.join(ver_path.split(os.sep)[:-1] + ['final_mask.npy'])
                mask = np.load(mask_path)
                mask = np.repeat(mask[..., None], 3, axis=-1)

                if args.uv_layout_type == 'standard_layout':
                    uv_img_mask = process_standard_layout(mask, faces, uvs, 
                                                          res=args.res)
                else:
                    uv_img_mask, _ = process_custom_layout(
                        mask, faces, uvs,
                        orig_flame_ver=orig_flame_ver,
                        orig_flame_faces=orig_flame_faces,
                        orig_flame_tri_uvs=orig_flame_tri_uvs,
                        mod_tri_uvs=mod_tri_uvs,
                        res=args.res, 
                        reuse_dense_uvs=dense_uvs_custom_layout
                    )
                
                uv_img_offset *= uv_img_mask

                # saving the mask separately
                imageio.imwrite(os.path.join(os.path.dirname(output_path),
                                             f'{os.path.splitext(out_basename)[0]}_mask.png'), 
                                (uv_img_mask * 255).astype(np.uint8))
        
            np.save(output_path, uv_img_offset)
            # saving rescaled uint8 version for quick visualization
            vis_min, vis_max = -1, 1
            uv_img_offset_vis = (np.clip(uv_img_offset, vis_min, vis_max) - vis_min) / (vis_max - vis_min + 1e-10)
            imageio.imwrite(os.path.join(os.path.dirname(output_path), 
                                         f'{os.path.splitext(out_basename)[0]}_vis.png'), 
                            (uv_img_offset_vis * 255).astype(np.uint8))
            
            # saving uint16 version for more precise visualization
            path_prefix = os.sep.join(ver_path.split(os.sep)[:-1] + ['uv_map_offset' + save_name_suffix])
            save_uint16(uv_img_offset, path_prefix)
