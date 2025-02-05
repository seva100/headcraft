conda create -n headcraft python==3.9

source ${CONDA_PREFIX}/etc/profile.d/conda.sh    # to be able to activate conda env in a bash script

conda activate headcraft

pip install numpy==1.23     # this version is needed for chumpy to work correctly
                            # https://github.com/NVIDIA/TensorRT/issues/2567 
pip install jupyter \
    pymeshlab \
    trimesh \
    imageio \
    trimesh \
    einops \
    tqdm \
    omegaconf \
    chumpy \
    numba

# for parallel processing on CPU only (registration stage):
pip install joblib

# pytorch
# NOTE: CUDA version needs to be properly adjusted for your machine.
# Here we install PyTorch for CUDA 11.8. 
# Check https://pytorch.org/ install section for more detail.

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# pytorch3d
# NOTE: in case PyTorch3D installation fails, check the following link for various different ways to install it:
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

# CUDA_HOME needs to be set to the right path manually.

FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/remote/cuda-11.8 \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"    # this might take up to 1h.

# stylegan2-ada-pytorch
# These requirements can be ommited if you're not using generation (train or inference).

pip install click \
    requests \
    pyspng \
    ninja

# stylegan2-ada-lightning
# See https://github.com/nihalsid/stylegan2-ada-lightning/blob/main/requirements.txt
# These requirements can be ommited if you're not using generation (train or inference).

pip install ballpark \
    argparse \
    pyyaml \
    typing \
    omegaconf \
    numpy \
    torchvision \
    pillow \
    wandb \
    hydra-core \
    lightning \
    pyyaml \
    torchmetrics \
    timm \
    piq \
    torch_ema \
    clean-fid \
    albumentations

pip install scipy==1.11.1    # for cleanfid with more than 2048 images
                             # https://github.com/GaParmar/clean-fid/issues/53 

pip install open3d

# alternative way is to install Open3D from source:
# pip install cmake
# git clone https://github.com/isl-org/Open3D
# mkdir build
# cd build
# cmake ..

# nproc=1    # specify according to the number of CPU cores and available RAM
# make -j$(nproc)    # will take significant number of minutes, even when running multi-threaded
# make install-pip-package

# can rm -r Open3D if needed
