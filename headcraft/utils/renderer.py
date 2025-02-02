import torch
import torch.nn as nn
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    SoftPhongShader,
    PointsRenderer,
    PointsRasterizer,
    PointsRasterizationSettings,
    AlphaCompositor,
    TexturesUV,
    TexturesVertex
)

from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


def make_renderer(R, T, device='cuda:0', res=256, pointcloud=False, point_radius=0.003):
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=100.0)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    if not pointcloud:
        raster_settings = RasterizationSettings(
            image_size=res, 
            blur_radius=0.0, 
            faces_per_pixel=2, 
            bin_size=0,
        )
    else: 
        raster_settings = PointsRasterizationSettings(
            image_size=res, 
            radius=point_radius,
            points_per_pixel=10
        )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    # lights = PointLights(device=device_render, location=[[0.0, -3.0, -5.0]])
    # lights = PointLights(device=device, location=[[0.0, 10.0, 50.0]])
    # lights = -cameras.R[0].t() @ cameras.T[0].unsqueeze(1)
    # lights = lights.reshape(1, 3)
    lights = cameras.get_camera_center()
    lights = PointLights(device=device, location=lights)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    if not pointcloud:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
            # shader=HardFlatShader(device=device_render, cameras=cameras, lights=lights)
        )
    else:
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            compositor=AlphaCompositor(background_color=(1, 1, 1)),
            # shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
        )
    return renderer



def make_ortho_renderer(R, T, device='cuda:0', res=256, make_rasterizer=False):
    
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=100.0)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01, zfar=100.0, min_y=0, max_x=0) #, min_y=0, min_x=0)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    
    raster_settings = RasterizationSettings(
        image_size=res, 
        blur_radius=0.0, 
        faces_per_pixel=2, 
        bin_size=0,
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    if make_rasterizer:
        return rasterizer
    
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SimpleShader()
        # shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
        # shader=HardFlatShader(device=device_render, cameras=cameras, lights=lights)
    )
    return renderer

