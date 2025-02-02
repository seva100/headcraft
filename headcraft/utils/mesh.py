from collections import defaultdict
from einops import rearrange
import os
import tempfile
import numpy as np
import torch
import open3d as o3d
from numba import jit


def triangle_uvs_to_vertex_uvs(triangle_uvs, n_verts, faces, seam_thr=0.1):

    # Note: this implementation may not be accurate in finding vertex UVs at the seams 
    # (however, it works well for e.g. offsets estimation procedure).
    # To handle the seam better, use the functions find_seam_ver(...) and duplicate_seam_ver(...) below.

    vertex_uvs = np.zeros((n_verts, 2))
    vertex_uvs_count = np.zeros((n_verts, 1))
    for face_no in range(faces.shape[0]):
        
        if seam_thr is not None and np.any(np.max(triangle_uvs[face_no], axis=0) - np.min(triangle_uvs[face_no], axis=0) > seam_thr):
            continue

        for j in range(3):
            if vertex_uvs_count[faces[face_no, j]] == 0 or np.linalg.norm(vertex_uvs[faces[face_no, j]] - triangle_uvs[face_no, j]) < seam_thr:
                vertex_uvs[faces[face_no, j]] += triangle_uvs[face_no, j]
                vertex_uvs_count[faces[face_no, j]] += 1

    vertex_uvs /= vertex_uvs_count
    return vertex_uvs


def find_seam_ver(triangle_uvs, n_verts, faces, seam_thr=0.1):
    vertex_uvs = np.zeros((n_verts, 2))
    vertex_uvs_set_once = np.zeros(n_verts, dtype=bool)

    for face_no in range(faces.shape[0]):
        face = faces[face_no]
        for i in range(3):
            v = face[i]
            if not vertex_uvs_set_once[v]:
                vertex_uvs[v] = triangle_uvs[face_no, i]
                vertex_uvs_set_once[v] = True
            else:
                vertex_uvs[v] = min(vertex_uvs[v], triangle_uvs[face_no, i], 
                                    key=lambda x: np.linalg.norm(x))

    seam_ver = np.zeros(n_verts, dtype=bool)
    for face_no in range(faces.shape[0]):
        face = faces[face_no]
        for i in range(3):
            v = face[i]
            if np.linalg.norm(vertex_uvs[v] - triangle_uvs[face_no, i]) > seam_thr:
                seam_ver[v] = True

    return seam_ver, vertex_uvs    # these uvs do not account for the seam yet (i.e. duplicate vertices not yet added)


@jit
def _duplicate_seam_ver_get_new(triangle_uvs, verts, faces, seam_ver, vertex_uvs_prelim, seam_thr=0.1):
    # assumption: every seam vertex should have exactly one duplicate (typically true for 3D meshes with seams, if there are no divergence points with several seams colliding at the point, etc.).
    # vertex_uvs_prelim is the vertex uvs before the seam handling

    n_verts = verts.shape[0]

    new_verts = np.full((np.sum(seam_ver), 3), np.nan, dtype=np.float32)
    new_vertex_uvs = np.full((np.sum(seam_ver), 2), np.nan, dtype=np.float32)
    new_faces = faces.copy()
    ver2new_ver = np.zeros(n_verts, dtype=np.int32)
    ver2new_ver[seam_ver] = np.arange(np.sum(seam_ver))

    for face_no in range(faces.shape[0]):
        face = faces[face_no]
        for i in range(3):
            v = face[i]
            if np.linalg.norm(vertex_uvs_prelim[v] - triangle_uvs[face_no, i]) > seam_thr:
                # seam vertex
                new_ver_idx = ver2new_ver[v]
                if np.isnan(new_verts[new_ver_idx].sum()):
                    new_verts[new_ver_idx] = verts[v]
                    new_vertex_uvs[new_ver_idx] = triangle_uvs[face_no, i]
                else:
                    assert np.all(np.isclose(new_verts[new_ver_idx], verts[v]))
                    assert np.all(np.isclose(new_vertex_uvs[new_ver_idx], triangle_uvs[face_no, i]))
                new_faces[face_no, i] = n_verts + new_ver_idx
    return new_verts, new_vertex_uvs, new_faces, ver2new_ver


def duplicate_seam_ver(triangle_uvs, verts, faces, seam_ver, vertex_uvs_prelim, seam_thr=0.1):
    new_verts, new_vertex_uvs, new_faces, ver2new_ver = _duplicate_seam_ver_get_new(
        triangle_uvs, verts, faces, seam_ver, vertex_uvs_prelim, seam_thr)

    # now we need to add the duplicates to the mesh
    new_verts = np.concatenate([verts, new_verts], axis=0)
    new_vertex_uvs = np.concatenate([vertex_uvs_prelim, new_vertex_uvs], axis=0)
    new_faces = new_faces
    return new_verts, new_vertex_uvs, new_faces, ver2new_ver


# def duplicate_seam_ver_faster(triangle_uvs, verts, faces, seam_ver, vertex_uvs_prelim, seam_thr=0.1):
#     mismatching_np.linalg.norm(vertex_uvs_prelim[faces] - triangle_uvs) > seam_thr    # (F, 3, 2)


def find_stretched_triangles(triangle_uvs, seam_thr=0.05):
    # triangle_uvs: (V, 3, 2)
    faces_perimeter = np.linalg.norm(np.roll(triangle_uvs, 1, axis=1) - triangle_uvs, axis=2).sum(axis=1)
    stretched_triangles_mask = faces_perimeter > seam_thr
    return stretched_triangles_mask


def dilate_vertex_mask(edges, mask):
    # edges: (M, 2) numpy array long
    # mask: (N,) numpy array bool (1 = keep; 0 = omit)

    mask_dilated = mask.copy().astype(bool)
    edges_crossing = mask_dilated[edges[:, 0]] != mask_dilated[edges[:, 1]]
    mask_dilated[edges[edges_crossing, 0]] = True
    mask_dilated[edges[edges_crossing, 1]] = True
    return mask_dilated


def find_subdiv_ver_in_region(region_name, region_masks, faces, edges, uvs_which_orig_tri, n_ver_subdiv, faces_subdiv, n_ver_coarse=5023, dilate_mask=None):
    # edges can be straightforwardly created from faces but added for simplicity here as a parameter
    # example usage: mask = find_subdiv_ver_in_region('inner_mouth', ..., dilate_mask=2)

    # NOTE better rewriting through splatting the mask onto UV and then querying with subdiv uvs?

    inds_coarse = region_masks[region_name].astype(int)
    mask_coarse = np.zeros((n_ver_coarse,)).astype(bool)
    mask_coarse[inds_coarse] = True

    if dilate_mask is not None and dilate_mask > 0:
        for _ in range(dilate_mask):
            mask_coarse = dilate_vertex_mask(edges, mask_coarse)

    uvs_which_orig_ver_inds = faces[uvs_which_orig_tri]     # (n_subdiv_tri * 3, 3)

    uvs_belong_to_face_region = mask_coarse[uvs_which_orig_ver_inds]
    uvs_belong_to_face_region = rearrange(uvs_belong_to_face_region, '(f d) c -> f d c', d=3)
    uvs_belong_to_face_region = np.any(uvs_belong_to_face_region, axis=2)

    ver_subdiv_belong_to_face_region = np.zeros((n_ver_subdiv,)).astype(bool)
    ver_subdiv_belong_to_face_region[faces_subdiv[:, 0]] = uvs_belong_to_face_region[:, 0]
    ver_subdiv_belong_to_face_region[faces_subdiv[:, 1]] = uvs_belong_to_face_region[:, 1]
    ver_subdiv_belong_to_face_region[faces_subdiv[:, 2]] = uvs_belong_to_face_region[:, 2]
    return ver_subdiv_belong_to_face_region
