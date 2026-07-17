#!/usr/bin/env python

import os
import shutil

import numpy as np
import itertools
from wsssss import load_data as ld
from wsssss import functions as uf
from platoconstants import cgs
from platoconstants import cs

import common

rng = np.random.default_rng(42)

out_dir = f'{os.path.dirname(__file__)}/../../output'

Gmags = [7, 9, 11, 13]
rel_rotations = [0.85, 1, 1.15]
inclinations = [15, 30, 60, 90]
num_spots = [0, ]
transits = ['single_deep', 'single_shallow', 'triple']


hists = [ld.History(f'../MESA/grid/{i:04}/LOGS/history.data') for i in range(11)]
profs = []
for h in hists:
    profs.extend(ld.load_profs(h))

iters_general = itertools.product(profs, Gmags, rel_rotations, inclinations, num_spots, transits)
iters_special_case = []


def calc_rotation(p, rel_rotation):
    return rel_rotation * 30


def get_transit_config(p, transit_type):
    mass = p.header['star_mass']
    if transit_type == 'single_deep':
        num = 1
        radius = 1
        period = 20
        phase_deg = rng.uniform(0, 360)

    elif transit_type == 'single_shallow':
        num = 1
        radius = 0.1
        period = 100
        phase_deg = rng.uniform(0, 360)

    elif transit_type == 'triple':
        num = 3
        radius = [0.5, 1.0, 0.1]
        period = np.array([10., 50, 300])
        phase_deg = rng.uniform(0, 360, 3)

    elif transit_type == 'special1':
        raise NotImplementedError

    elif transit_type == 'special2':
        raise NotImplementedError

    elif transit_type == 'ttv':
        raise NotImplementedError

    else:
        raise Exception(f'Unknown transit type: {transit_type}')

    semi_major_axis = (mass * (period/365.25636)**2) ** (1/3)

    return {'Enable': num,
            'PlanetRadius':radius,
            'OrbitalPeriod': period,
            'PlanetSemiMajorAxis': semi_major_axis,
            'OrbitalAngle': phase_deg}


def get_spot_config(n_spot):
    config = {}
    if n_spot == 0:
        config['Enable'] = 0
    else:
        config['Enable'] = 1
        raise NotImplementedError
    return config


def make_psls_config(path, i, profile, Vmag, rot_period, inclination, spot_config, transit_config):
    config = {'Observation': {},
              'Star': {},
              'Activity': {}}
    config['Star']['ID'] = i
    config['Star']['Mag'] = Vmag
    config['Star']['SurfaceRotationPeriod'] = rot_period
    config['Star']['Inclination'] = inclination
    pnum = profile.fname.split('.')[0].replace('profile', '')
    hnum = profile.LOGS.split('/')[-2]
    config['Star']['ModelName'] = f'{hnum}_{pnum}_{rot_period:.1f}.modes'

    config['Activity']['Spot'] = spot_config
    config['Transit'] = transit_config

    # Generate hash
    config['Observation']['MasterSeed'] = hash(f'{i}{Vmag}{rot_period}{inclination}') % 2**32

    with open(path, 'w') as f:
        f.write(common.make_yaml_str(config))


def generate_gyre_configs():
    grid_dir = f'{os.path.dirname(__file__)}/../MESA/grid'
    gyre_out_dir = f'{grid_dir}/../base_gyre_in_rot'
    if os.path.exists(gyre_out_dir):
        shutil.rmtree(gyre_out_dir)
    os.makedirs(gyre_out_dir)
    with open(f'{grid_dir}/../INPUT_GYRE_9_ad.in', 'r') as handle:
        default_gyre_in = handle.read()
    for profile in profs:
        pnum = profile.fname.split('.')[0].replace('profile', '')
        hnum = profile.LOGS.split('/')[-2]
        for i, rel_rot in enumerate(rel_rotations):
            rot_period = calc_rotation(profile, rel_rot)
            fname = f'{hnum}_{pnum}_{rot_period:.1f}.in'
            # fname = f'{hnum}_{pnum}_{["lo", "med", "hi"][i]}.in'
            rot_freq = 1/rot_period
            new_omega_rot_str = f'omega_rot = {rot_freq:g}'.replace('e-', 'd-').replace('e+', 'd+')
            gyre_in = default_gyre_in.replace('omega_rot = 0d0', new_omega_rot_str)
            with open(f'{gyre_out_dir}/{fname}', 'w') as handle:
                handle.write(gyre_in)


generate_gyre_configs()
for j, iters in enumerate([iters_general, iters_special_case]):
    for i, (p, Vmag, rel_rotation, inclination, num_spot, transit_type) in enumerate(iters):
        star_id = int(j * 1e8) + i
        rot_period = calc_rotation(p, rel_rotation)
        spot_config = get_spot_config(num_spot)
        transit_config = get_transit_config(p, transit_type)

        make_psls_config(f'{out_dir}/general/{star_id:08}.yaml', star_id, p, Vmag, rot_period, inclination, spot_config, transit_config)
