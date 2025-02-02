### Registration pipeline

We assume that the NPHM dataset (or any other suitable dataset) is downloaded to the folder `<root_dir>` and has a structure `<root_dir>/<seq_name>/<exp_name>/{scan.ply,flame.ply}` (example: `/home/user/Downloads/NPHM/017/000/scan.ply`).
The folder `<output_root_dir>` could be either the same as `<root_dir>` (the simplest way) or a new folder that will contain the processed files (will be created automatically when specified).

`cd headcraft/registration`

1. Subdivide each FLAME fit in the dataset (`headcraft/registration/subdiv_flame.py`). The script only uses the CPU and calls MeshLab under the hood to perform subdivision for each mesh individually. Specify number of jobs for parallel processing.

```python
python subdiv_flame.py \
    --dataset_path "<root_dir>" \
    --output_path "<output_root_dir>" \
    --n_jobs "<number of jobs for parallel processing>"
```

2. Regress vector displacements (`headcraft/registration/estimate_vector_offsets.py`).

```python
python estimate_vector_offsets.py \
    --root_dir "<root_dir>" \
    --target_root_dir "<root_dir>" \
    --output_root_dir "<output_root_dir>" \
    [--only_n_first_ids "<number of the first IDs to process, if only a part of the dataset needs to be processed>"] \
    [--only_n_first_expr "<number of the first exprs to process, if only a part of the dataset needs to be processed>"] \
    [--n_iter <number of optimization steps>] \
    [--visualize_progress] \
    [--vis_save_every_n_steps <show vis image this each number of steps>]
```

3. Regress normal displacements. (`headcraft/registration/estimate_normal_offsets.py`)

```python
python estimate_normal_offsets.py \
    --root_dir "<root_dir>" \
    --target_root_dir "<root_dir>" \
    --output_root_dir "<output_root_dir>" \
    [--only_n_first_ids "<number of the first IDs to process, if only a part of the dataset needs to be processed>"] \
    [--only_n_first_expr "<number of the first exprs to process, if only a part of the dataset needs to be processed>"] \
    [--n_iter <number of optimization steps>] \
    [--visualize_progress] \
    [--vis_save_every_n_steps <show vis image this each number of steps>]
```

4. Bake the resulting displacements into UV maps (`headcraft/registration/save_uv_map.py`).

```python
python save_uv_map.py \
    --root_dir "<root_dir>" \
    --output_root_dir "<output_root_dir>" \
    [--res <resolution of UV maps -- should be at least 256 for training StyleGAN after>] \
    [--uv_layout_type standard_layout|custom_layout] \
    [--custom_layout_mesh_fn "<path to the mesh, UV coords of which will define the layout; if omitted, our layout is automatically loaded>"]    # --uv_layout_type=custom_layout should be specified to use our layout; otherwise, standard UV layout of FLAME is applied
```

All steps can be run via a single script:

```python
python process_scans.py \
    --root_dir "<root_dir>" \
    --output_root_dir "<output_root_dir>"
```

that uses default (recommended) registration hyperparameters. Modify the hyperparameters of the individual stages in `process_scans.py` if required.