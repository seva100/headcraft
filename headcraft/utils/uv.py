import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def sample_map_with_coords(img, uv, mode='bilinear'):
    # img: numpy array (h, w, c)
    # uv: numpy array (n_coords, 2)
    
    uv_h, uv_w = img.shape[:2]
    
    ii = uv[:, 0] / (uv_h - 1) * uv_h
    ii = ii * 2 - 1

    jj = (1 - uv[:, 1]) / (uv_w - 1) * uv_w
    jj = jj * 2 - 1

    grid = np.stack([ii, jj], axis=1)

    out = F.grid_sample(rearrange(torch.tensor(img).float(), 'h w c -> 1 c h w'), 
                        rearrange(torch.tensor(grid).float(), 'm d -> 1 m 1 d'),
                        mode=mode)
    out = rearrange(out, '1 c m 1 -> m c')
    out = out.cpu().data.numpy()
    
    return out
