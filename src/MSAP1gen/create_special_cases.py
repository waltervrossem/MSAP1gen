#!/usr/bin/env python
import os
import sys
import shutil
import tqdm
import numpy as np
from platoconstants import cgs
from platoconstants import cs

from wsssss import load_data as ld
from wsssss.constants import post15140 as c_mesa

from pytransit import RRModel

from MSAP1gen import create_config as cg
from MSAP1gen import common

sys.path.append(f'{os.path.dirname(__file__)}/../psls/psls-1.9')
import psls

baseline_model = 1783  # Sun-like
n_quarter = 8
len_quarter = 90

baseline_yaml = f'../../configs/baseline/{baseline_model:010}.yaml'
output_dir = '../../input/special'
hnum, pnum = common.read_yaml(baseline_yaml)['Star']['ModelName'].split('_')[:2]
prof = ld.Profile(f'../MESA/grid/{hnum}/LOGS/profile{pnum}.data')


def write_external(star_id, data, fname):
    dat = np.stack(data, axis=1)
    os.makedirs(f'../../input/special/{star_id:010}', exist_ok=True)
    with open(f'../../input/special/{star_id:010}/{fname}', 'w') as handle:
        for line in dat.tolist():
            handle.write(' '.join([str(_) for _ in line]) + '\n')
    return {'Enable': 1,
            'FilePath': fname}


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

    fname = 'ecl_bin.csv'
    config['External'] = write_external(star_id, [86400*t_lores, (flux - 1) * 1e6], fname)
    s = common.make_yaml_str(config)
    with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
        handle.write(s)
    return 1


def make_only_nan(i, j):
    star_id = int(j * 1e9 + i)

    cg.setup(f'{output_dir}/{star_id}', baseline_yaml, None, 0, 'psls.yaml')
    config = common.read_yaml(f'{output_dir}/{star_id}/psls.yaml')
    config['Star']['ID'] = star_id

    fname = 'nan_only.csv'
    t_lores = np.arange(0, n_quarter * len_quarter + 2, 1 / (24 * 10), )
    config['External'] = write_external(star_id, [t_lores * 86400, np.full_like(t_lores, np.nan)], fname)
    s = common.make_yaml_str(config)
    with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
        handle.write(s)
    return 1


def make_gap_transit(i, j):
    star_id = int(j * 1e9 + i)

    cg.setup(f'{output_dir}/{star_id}', baseline_yaml, None, 0, 'psls.yaml')
    config = common.read_yaml(f'{output_dir}/{star_id}/psls.yaml')
    config['Star']['ID'] = star_id

    mass = 1
    TTV_period = 0.0
    TTV_amplitude = 0.0
    TTV_phase = 0.0

    num = 1
    radius = 1.5
    period = 89
    phase_deg = 0

    semi_major_axis = (mass * (period / 365.25636) ** 2) ** (1 / 3)
    config['Transit'].update({'Enable': num,
                              'PlanetRadius': [radius],
                              'OrbitalPeriod': [period],
                              'PlanetSemiMajorAxis': [semi_major_axis],
                              'OrbitalAngle': [phase_deg],
                              'TTV_Period': [TTV_period],
                              'TTV_Amplitude': [TTV_amplitude],
                              'TTV_Phase': [TTV_phase]})
    s = common.make_yaml_str(config)
    with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
        handle.write(s)
    return 1


def make_gap_flares(i, j):
    star_id = int(j * 1e9 + i)

    cg.setup(f'{output_dir}/{star_id}', baseline_yaml, None, 0, 'psls.yaml')
    config = common.read_yaml(f'{output_dir}/{star_id}/psls.yaml')
    config['Star']['ID'] = star_id

    config['Observation']['Gaps']['Enable'] = 1

    config['Activity']['Flare']['Enable'] = 0
    Flare = config['Activity']['Flare']

    t = np.arange(0, 86400*(n_quarter * len_quarter + 0.125), 25)
    fname = 'flares.csv'
    LC_flares_ = psls.AddFlare(t, Flare['MeanPeriod'], Flare['UpDown'],
                             Flare['Amplitude'], Flare['MeanDuration'],
                             Flare['DurationDispersion'], 43486, config['Star'])

    incl = config['Star']['Inclination']
    prot = config['Star']['SurfaceRotationPeriod']

    rng = np.random.default_rng(4546546)

    # Long duration flares
    # Flare should be visible at peak time
    t_flares = 86400*(90*np.arange(0, 8, 1))
    FlareNumbers = len(t_flares)
    amp_flares = rng.uniform(5000, 10000, size=FlareNumbers)
    duration_flares = 60*np.exp(rng.normal(loc=4, scale=0.5, size=FlareNumbers))
    duration_flares[4] = 3600 * 24 * 2  # Include one very long flare
    amp_flares[4] = 50000

    lon = (rng.uniform(-45, 45, FlareNumbers) + np.degrees(t_flares * (2*np.pi/(prot*86400))) % 360) % 360
    lat = rng.uniform(0, 60, FlareNumbers)

    t_flares = t_flares + duration_flares*rng.uniform(-0.5, 0.5, size=FlareNumbers) + 86400 * np.array([3, *rng.choice([0, 3], FlareNumbers-2, replace=True), 0])

    LC_flares = np.ones_like(t)
    LC_flares = psls.flares.add_flares(LC_flares, t, t_flares, amp_flares, duration_flares, Flare['UpDown'],
                           prot=prot, incl=incl, lat=lat, lon=lon)
    LC_flares += (LC_flares_ - 1)/1e-6  # Needs to be in ppm
    config['External'] = write_external(star_id, [t, LC_flares], fname)

    s = common.make_yaml_str(config)
    with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
        handle.write(s)
    return 1


def make_constant_long_gaps(i, j):
    star_id = int(j * 1e9 + i)

    cg.setup(f'{output_dir}/{star_id}', baseline_yaml, None, 0, 'psls.yaml')
    config = common.read_yaml(f'{output_dir}/{star_id}/psls.yaml')
    config['Star']['ID'] = star_id

    config['Observation']['Gaps']['Enable'] = 1
    config['Observation']['Gaps']['RandomGapDuration'] = 17
    config['Observation']['Gaps']['RandomGapTimeFraction'] = 30
    config['Observation']['Gaps']['PeriodicGapCadence'] = 0.5
    config['Observation']['Gaps']['PeriodicGapJitter'] = 4

    config['Instrument']['RandomNoise']['Enable'] = 0
    config['Instrument']['Systematics']['Enable'] = 0
    config['Oscillations']['Enable'] = 0
    config['Activity']['Spot']['Enable'] = 0
    for name in ['Radius', 'Latitude', 'Longitude', 'Lifetime', 'TimeMax']:
        config['Activity']['Spot'][name] = []
    config['Activity']['Flare']['Enable'] = 0
    config['Granulation']['Enable'] = 0
    config['Transit']['Enable'] = 0
    s = common.make_yaml_str(config)
    with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
        handle.write(s)
    return 1


def make_wrong_transit_model(i, j):
    star_id = int(j * 1e9 + i)
    cg.setup(f'{output_dir}/{star_id}', baseline_yaml, None, 0, 'psls.yaml')
    config = common.read_yaml(f'{output_dir}/{star_id}/psls.yaml')
    config['Star']['ID'] = star_id

    Star = config['Star']
    StarRadius = prof.header['photosphere_r'] * c_mesa.rsun / 1e5
    Instrument = config['Instrument']
    IntegrationTime = Instrument['IntegrationTime']
    Sampling = Instrument['Sampling']
    gamma = np.array(config['Transit']['LimbDarkeningCoefficients'], dtype=np.float64)

    config['Transit'].update({'Enable': 1,
            'PlanetRadius': [1],
            'OrbitalPeriod': [50],
            'PlanetSemiMajorAxis': [(1 * (50/365.25636)**2) ** (1/3)],
            'OrbitalAngle': [40],
            'TTV_Period': [0],
            'TTV_Amplitude': [0],
            'TTV_Phase': [0]})

    fname = 'extra_transits.csv'

    t = np.arange(0, 86400*(n_quarter * len_quarter + 1), Sampling)
    p_orb = 35
    Transit = {'Enable': 1,
             'PlanetRadius': 1,
             'OrbitalPeriod': p_orb,
             'PlanetSemiMajorAxis': (1 * (p_orb / 365.25636) ** 2) ** (1 / 3),
             'OrbitalAngle': 40,
             'TTV_Period': 0,
             'TTV_Amplitude': 0,
             'TTV_Phase': 0}

    p = 1 * psls.jupiterRadius / StarRadius
    _, z, _, _ = psls.generateZ(Transit['OrbitalPeriod'] * 86400., Transit['PlanetSemiMajorAxis'] * psls.ua2Km,
                                     StarRadius, Sampling, IntegrationTime, 0, t.size,
                                     Transit['OrbitalAngle'] * np.pi / 180., p,
                                     Transit['TTV_Period']*86400, Transit['TTV_Amplitude']*86400, Transit['TTV_Phase'])
    flux = psls.tr.occultquad(z, p, gamma)

    config['External'] = write_external(star_id, [t, (flux-1)/1e-6], fname)

    s = common.make_yaml_str(config)
    with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
        handle.write(s)
    return 1


def make_random_spikes(i, j):
    star_id = int(j * 1e9 + i)
    rng = np.random.default_rng(98463)
    for num_cam_group in [1,2,3,4]:
        cg.setup(f'{output_dir}/{star_id}', baseline_yaml, None, 0, 'psls.yaml')
        config = common.read_yaml(f'{output_dir}/{star_id}/psls.yaml')
        config['Star']['ID'] = star_id
        config['Instrument']['RandomSpikes']['Enable'] = 1
        config['Instrument']['GroupID'] = np.sort(rng.choice([1,2,3,4], num_cam_group, replace=False)).tolist()

        s = common.make_yaml_str(config)
        with open(f'{output_dir}/{star_id}/psls.yaml', 'w') as handle:
            handle.write(s)
        star_id += 1
    return 4


def make_special(j):
    print('Making special cases')
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    with tqdm.tqdm(total=10) as pbar:
        for func in [make_EB, make_only_nan, make_gap_transit, make_constant_long_gaps, make_gap_flares, make_wrong_transit_model, make_random_spikes]:
            num = func(i, j)
            pbar.update(num)
            i += num


if __name__ == '__main__':
    shutil.rmtree(output_dir, ignore_errors=True)
    make_special(9)
