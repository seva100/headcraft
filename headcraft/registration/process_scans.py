import subprocess

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Subdivide a mesh using the butterfly subdivision scheme. The script only uses the CPU.')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the dataset folder that contains meshes in a structure "dataset_path/subject_id/subject_expr/mesh.ply"')
    parser.add_argument('--output_root_dir', type=str, required=True, help='Path to the folder where all the processed data will be saved. Can be the same as --root_dir.')
    args = parser.parse_args()

    # subdiv the meshes
    # print('##### STEP #1. Subdividing the meshes #####')
    # cmd = (
    #     'python subdiv_flame.py '
    #     f'--dataset_path "{args.root_dir}" '
    #     f'--output_path "{args.output_root_dir}" '
    #     f'--n_jobs 8'
    # )
    # print('Executing cmd:', cmd)
    # subprocess.run(cmd, shell=True)
    # print('##### STEP #1. Done #####')

    # # regress vector displacements
    # print('##### STEP #2. Regressing the vector displacements #####')
    # cmd = (
    #     'python estimate_vector_offsets.py '
    #     f'--root_dir "{args.output_root_dir}" '
    #     f'--target_root_dir "{args.root_dir}" '
    #     f'--output_root_dir "{args.output_root_dir}" '
    #     '--visualize_progress '
    # )
    # print('Executing cmd:', cmd)
    # subprocess.run(cmd, shell=True)
    # print('##### STEP #2. Done #####')

    # # regress normal displacements
    # print('##### STEP #3. Regressing the normal displacements #####')
    # cmd = (
    #     'python estimate_normal_offsets.py '
    #     f'--root_dir "{args.output_root_dir}" '
    #     f'--target_root_dir "{args.root_dir}" '
    #     f'--output_root_dir "{args.output_root_dir}" '
    #     '--visualize_progress '
    # )
    # print('Executing cmd:', cmd)
    # subprocess.run(cmd, shell=True)
    # print('##### STEP #3. Done #####')

    # bake the results into UV maps
    print('##### STEP #4. Baking the results into UV maps #####')
    cmd = (
        'python save_uv_map.py '
        f'--root_dir "{args.output_root_dir}" '
        f'--output_root_dir "{args.root_dir}" '
        '--uv_layout_type custom_layout '
    )
    print('Executing cmd:', cmd)
    subprocess.run(cmd, shell=True)
    print('##### STEP #4. Done #####')
