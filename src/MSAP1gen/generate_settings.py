#!/usr/bin/env python

import os
import shutil

import numpy as np
import itertools
from wsssss import load_data as ld
from wsssss import functions as uf
from platoconstants import cgs
from platoconstants import cs
from scipy import interpolate as ip

import common

rng = np.random.default_rng(42)

out_dir = f'{os.path.dirname(__file__)}/../../configs'

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
def S_logP_l(logP, wP, logP_break):
    return 1 / (1 + np.exp(-(logP_break - logP)/wP))


def S_logP_h(logP, wP, logP_break):
    return 1 / (1 + np.exp(-(-logP_break + logP) / wP))


def f_period(logP, logP_break, wP, dP1, dP2):
    P = 10**logP
    Pbreak = 10**logP_break
    return S_logP_h(logP, wP, logP_break) * P**dP1 + S_logP_l(logP, wP, logP_break) * P**dP2 * Pbreak ** (dP1 - dP2)


def S_T_h(Tn, wT):
    return 1 / (1 + np.exp(-(1 - Tn)/wT))


def S_T_l(Tn, wT):
    return 1 / (1 + np.exp(-(-1 + Tn)/wT))


def g_temp(Tn, wT, c_T, dT1, dT2):
    return S_T_h(Tn, wT) * (Tn - c_T)**dT1 + S_T_l(Tn, wT) * (Tn - c_T)**dT2 * (1 - c_T)**(dT1 - dT2)

def gyro_mean_function(logP, Teff, fully_convective):
    if fully_convective:
        a = 0.774
        dP1 = 0.376
        dP2 = 1.811
        c_T = -0.223
        dT1 = -0.687
        dT2 = dT1
        Pbreak = 73.322
        Tbreak = 0
        wT = 1
        wP = 0.068
        Tn = (3500 - Teff) / 500
    else:
        a = 118.969
        dP1 = -0.405
        dP2 = 1.822
        c_T = -0.399
        dT1 = 1.646
        dT2 = -17.779
        Pbreak = 100.836
        Tbreak = 3713.699
        wT = 0.062
        wP = 0.111
        Tn = (7000 - Teff) / (7000 - Tbreak)
    logP_break = np.log10(Pbreak)
    return a * f_period(logP, logP_break, wP, dP1, dP2) * g_temp(Tn, wT, c_T, dT1, dT2)


def calc_rotation(p, rel_rotation):
    mass = p.header['star_mass']
    L = p.header['photosphere_L']
    age = p.header['star_age']
    Teff = p.header['Teff']
    radius = np.sqrt(L / (Teff/5772)**4)
    fully_convective = np.all(p.get('mixing_type')[p.get('radius') < radius] == c_mesa.convective_mixing)

    # Lu+2024 https://iopscience.iop.org/article/10.3847/1538-3881/ad28b9/pdf
    logP = np.linspace(0.5, 2.3, 101)
    ages = gyro_mean_function(logP, Teff, fully_convective)
    rot = rel_rotation * 10**ip.interp1d(ages, logP, bounds_error=False, fill_value=tuple(logP[[0, -1]]))(min(age/1e9, 14))
    return rot


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
    gyre_out_dir = f'{out_dir}/gyre'
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


if __name__ == "__main__":
    generate_gyre_configs()
    kind = ['general', 'special']
    for j, iters in enumerate([iters_general, iters_special_case]):
        os.makedirs(f'{out_dir}/{kind[j]}', exist_ok=False)
        for i, values in enumerate(iters):
            if isinstance(values, dict):
                pass
            else:
                p, Vmag, rel_rotation, inclination, spot_options, transit_type = values
            star_id = int(j * 1e8) + i
            rot_period = calc_rotation(p, rel_rotation)
            spot_config = get_spot_config(spot_options, rot_period)
            transit_config = get_transit_config(p, transit_type)

            make_psls_config(f'{out_dir}/{kind[j]}/{star_id:08}.yaml', star_id, p, Vmag, rot_period, inclination, spot_config, transit_config)
