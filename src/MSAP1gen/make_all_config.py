#!/usr/bin/env python

import os
import shutil
import tqdm
import multiprocessing as mp

from run import *

base_input_dir = '../../configs/'
base_output_dir = '../../input'

def worker(config_name):
    if config_name.startswith('1'):
        raise ValueError(f'Special case in general config file directory: {config_name}')
    i = config_name.split('.')[0]
    config_path = os.path.join(input_dir, config_name)
    try:
        cg.setup(f'../../input/{i}', config_path, None, 0, 'psls.yaml')
    except Exception as e:
        print(i, e)
        raise

if __name__ == "__main__":
    nworker = mp.cpu_count()
    os.environ['OMP_NUM_THREADS'] = '1'

    skip_existing = True
    for kind in ['general']:#, 'special']:
        print(f'Making {kind} inputs.')
        output_dir = os.path.join(base_output_dir, kind)
        input_dir = os.path.join(base_input_dir, kind)
        if not skip_existing:
            shutil.rmtree(output_dir, ignore_errors=True)

        os.makedirs(output_dir, exist_ok=True)

        config_files = sorted(os.listdir(input_dir))
        existing = sorted(os.listdir(output_dir))

        if skip_existing:
            config_files = [c for c in config_files if c[:8] not in existing]

        with mp.Pool(nworker) as pool, tqdm.tqdm(total=len(config_files)) as pbar:
            for res in pool.imap_unordered(worker, config_files):
                pbar.update(1)
