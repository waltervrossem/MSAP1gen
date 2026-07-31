#!/usr/bin/env python
import os
import shutil
import numpy as np
from platoconstants import cgs
from platoconstants import cs

from pytransit import RRModel

import create_config as cg
import common

# Transits beginning or ending with a gap
# Flares at the edges of a quarter
# Constant lightcurve with very long gap(s)
# LC with only Nan
# Very wrong transit model or with weird values
# Change the quality of random observational points at the level of groups of cameras (L1) and remove groups of cameras at L1 level.
# Include random gaps of different lengths and at different positions. 2 or 3 versions of the same LC with different gaps
# Eclipsing binaries

baseline_model = 1783  # Sun-like
n_quarter = 8
len_quarter = 90

baseline_yaml = f'../../configs/baseline/{baseline_model:010}.yaml'
output_dir = '../../input/special'


def make_EB(i, j):
    star_id = int(j * 1e9 + i)

    cg.setup(f'{output_dir}/{star_id}', baseline_yaml, None, 0, 'psls.yaml')
    config = common.read_yaml(f'{output_dir}/{star_id}/psls.yaml')
    config['Star']['ID'] = star_id

    p_orb = 20
    flux_ratio = 1.5
    t_lores = np.arange(0, n_quarter * len_quarter + 2, 1 / (24 * 10), )
    tm = RRModel('power-2')
    tm.set_data(time=t_lores)

    flux1 = tm.evaluate(k=0.51, ldc=[0.6, 0.5], t0=0.0, p=p_orb, a=4.2, i=0.5 * np.pi, e=0.0, w=0.0)
    flux2 = tm.evaluate(k=0.75, ldc=[0.6, 0.5], t0=p_orb / 2, p=p_orb, a=4.2, i=0.5 * np.pi, e=0.0, w=0.0)
    flux = (flux1 + flux2 / flux_ratio) / (1 + 1 / flux_ratio)

    dat = np.stack([t_lores * 86400, (flux - 1) * 1e6], axis=1)
    os.makedirs(f'../../input/special/{star_id:010}', exist_ok=True)
    with open(f'../../input/special/{star_id:010}/ecl_bin.csv', 'w') as handle:
        for line in dat.tolist():
            handle.write(' '.join([str(_) for _ in line]) + '\n')
    config['External']['Enable'] = 1
    config['External']['FilePath'] = 'ecl_bin.csv'
    s = common.make_yaml_str(config)
    with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
        handle.write(s)


def make_special(j):
    i = 0
    make_EB(i, j)
    i += 1

if __name__ == '__main__':
    shutil.rmtree('../../input/special', ignore_errors=True)
    make_special(9)
