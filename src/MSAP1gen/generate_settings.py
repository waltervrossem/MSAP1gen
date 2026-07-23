#!/usr/bin/env python

import os
import shutil
import random

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
transits = ['single_deep', 'single_shallow', 'triple', 'ttv']


hists = [ld.History(f'../MESA/grid/{i:04}/LOGS/history.data') for i in range(11)]
profs = []
for h in hists:
    profs.extend(ld.load_profs(h))
c_mesa = uf.get_constants(hists[0])

iters_general = itertools.product(profs, Vmags, rel_rotations, inclinations, num_spots, transits)
num_general = np.prod(list(map(len, [profs, Vmags, rel_rotations, inclinations, num_spots, transits])))
iters_special_case = []
def mass_to_radius(mass):
    # https://www.aanda.org/articles/aa/full_html/2024/06/aa48690-23/aa48690-23.html
    if mass < 4.37:
        radius = 1.02 * mass ** 0.27
    else:# mass >= 4.37 and mass < 127:
        radius = 0.56 * mass ** 0.67
    # else:
    #     radius = 18.6 * mass ** -0.06
    return radius


def radius_to_mass(radius):
    if radius < 1.504:
        mass = (radius/1.02) ** (1/0.27)
    else:# radius >= 1.504 and radius < 13.91:
        mass = (radius / 0.56) ** (1/0.67)
    return mass


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
    TTV_period = 0.0
    TTV_amplitude = 0.0
    TTV_phase = 0.0
    if transit_type == 'single_deep':
        num = 1
        radius = rng.uniform(1, 2.5)
        period = rng.uniform(1, 20)
        phase_deg = rng.uniform(0, 360)

    elif transit_type == 'single_shallow':
        num = 1
        radius = rng.uniform(0.5, 3) * cgs.terrestrial.EARTH_RADIUS / common.JUPITER_RADIUS
        period = rng.uniform(3, 200)
        phase_deg = rng.uniform(0, 360)

    elif transit_type == 'triple':
        num = 3
        radius = []
        period = []
        phase_deg = []
        for i in range(num):
            transit = get_transit_config(p,  ['single_shallow', 'single_shallow', 'single_deep'][i])
            radius.append(transit['PlanetRadius'])
            period.append(transit['OrbitalPeriod'])
            phase_deg.append(transit['OrbitalAngle'])
        period = np.array(period)

    elif transit_type == 'ttv':
        num = 1
        radius = rng.uniform(0.5, 3)  # Earth radii
        period = rng.uniform(1, 20)
        phase_deg = rng.uniform(0, 360)

        period_small = rng.uniform(1, 20)
        radius_small = rng.uniform(0.5, 3)
        radius_large = rng.uniform(1, 2.5) * common.JUPITER_RADIUS/cgs.terrestrial.EARTH_RADIUS
        mass_ratio = radius_to_mass(radius_large) * cgs.terrestrial.EARTH_MASS / (mass * cgs.solar.SOLAR_MASS)
        j = int(rng.choice([2, 3], 1)[0])
        Delta = rng.uniform(-0.06, 0.06)  # https://arxiv.org/pdf/1308.0996  after eq 1

        inout = rng.choice([0, 1], 2, replace=False)
        radius_in, radius_out = np.array([radius_small, radius_large])[inout]
        if inout[0] == 0:  # Small planet is inner planet
            period_large = (Delta + 1) * period_small * j / (j - 1)
            period_in = period_small
            period_out = period_large
            f = 1
            TTV_amplitude = period_in * mass_ratio * abs(f/Delta) / (np.pi * j ** (2 / 3) * (j - 1) ** (1 / 3))
        else:
            period_large = period_small / (Delta+1) * (j - 1) / j
            period_out = period_small
            period_in = period_large
            g = 1
            TTV_amplitude = period_out * mass_ratio * abs(g/Delta) / (np.pi * j)

        TTV_period = period_out / abs(j*Delta)
        TTV_phase = rng.uniform(0, 360)
    else:
        raise Exception(f'Unknown transit type: {transit_type}')

    semi_major_axis = (mass * (period/365.25636)**2) ** (1/3)

    if transit_type != 'triple':
        radius = [radius]
        period = [period]
        semi_major_axis = [semi_major_axis]
        phase_deg = [phase_deg]
        TTV_period = [TTV_period]
        TTV_amplitude = [TTV_amplitude]
        TTV_phase = [TTV_phase]

    return {'Enable': num,
            'PlanetRadius':radius,
            'OrbitalPeriod': period,
            'PlanetSemiMajorAxis': semi_major_axis,
            'OrbitalAngle': phase_deg,
            'TTV_Period': TTV_period,
            'TTV_Amplitude': TTV_amplitude,
            'TTV_Phase': TTV_phase}

def gen_spots(activity, prot, nquarters):
    """
    Generate random spots, statistics depend on the selected 'activity' level
    From gen_yaml_spots.py by Jordan Philidet
    """
    if activity == 'norot':
        maxTimes, radiiValidated, lifetimesValidated, latitudes, longitudes, contrasts = [], [], [], [], [], []
    else:
        # Defining spot statistics
        if activity in ['active', 'diffrot']:
            meanRadius = 2.5
            stdRadius = 0.5
            nspotPerRot = 4
            meanLifetime = 2.0
            stdLifetime = 0.5
            minLatitudes = 0.0
            maxLatitudes = 60.0
        else:
            meanRadius = 1.5
            stdRadius = 0.5
            nspotPerRot = 1
            meanLifetime = 1.0
            stdLifetime = 0.5
            minLatitudes = 0.0
            maxLatitudes = 30.0
        meanContrast = 0.5
        stdContrast = 0.1
        # Generating spots
        durationLC = nquarters * 90.0
        nrot = durationLC / prot
        nspotFloat = nspotPerRot * nrot
        nspot = int(nspotFloat)
        maxTimes = [durationLC * ispot/(nspot-1) for ispot in range(nspot)]
        radii = [random.gauss(mu=meanRadius, sigma=stdRadius) for _ in range(nspot)]
        lifetimes = [prot * random.gauss(mu=meanLifetime, sigma=stdLifetime) for _ in range(nspot)]
        if activity == 'diffrot':
            randInts = np.random.choice(2, size=nspot)
            latitudes = [0.0 if r == 0 else 60.0 for r in randInts]
        else:
            latitudes = [random.uniform(minLatitudes, maxLatitudes) for _ in range(nspot)]
        longitudes = [random.uniform(0.0, 360.0) for _ in range(nspot)]
        contrasts = [random.gauss(mu=meanContrast, sigma=stdContrast) for _ in range(nspot)]
        radiiValidated = [radii[i] if radii[i]>=0.0 else meanRadius for i in range(nspot)]
        lifetimesValidated = [lifetimes[i] if lifetimes[i]>=0.0 else prot * meanLifetime for i in range(nspot)]
    return maxTimes, radiiValidated, lifetimesValidated, latitudes, longitudes, contrasts


def get_spot_config(spot_options, prot):
    config = {}
    if spot_options == 0:
        config['Enable'] = 0
    else:
        config['Enable'] = 1
        spotTimes, spotRadii, spotLifetimes, spotLatitudes, spotLongitudes, spotContrasts = gen_spots(spot_options, prot, 8)

        config['Radius'] = spotRadii
        config['Latitude'] = spotLatitudes
        config['Longitude'] = spotLongitudes
        config['Lifetime'] = spotLifetimes
        config['TimeMax'] = spotTimes
        config['Contrast'] = spotContrasts

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
