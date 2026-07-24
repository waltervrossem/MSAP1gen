#!/usr/bin/env python

import os
import sys
import shutil
import tqdm
import multiprocessing as mp

import numpy as np
import itertools
from wsssss import load_data as ld
from wsssss import functions as uf
from platoconstants import cgs
from platoconstants import cs
from scipy import interpolate as ip

import common

sys.path.append(f'{os.path.dirname(__file__)}/../psls/psls-1.9')
import psls

rng = np.random.default_rng(42)

out_dir = f'{os.path.dirname(__file__)}/../../configs'

Gmags = [7, 9, 11, 13]
rel_rotations = [0.85, 1, 1.15]
inclinations = [15, 30, 60, 90]
num_spots = [0, 'active']
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
    config = {}
    config['Radius'] = []
    config['Latitude'] = []
    config['Longitude'] = []
    config['Lifetime'] = []
    config['TimeMax'] = []
    config['Contrast'] = []
    if activity == 'norot':
        pass
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
        maxTimes = durationLC / (nspot - 1) * np.arange(nspot)
        radii = rng.normal(meanRadius, stdRadius, nspot)
        lifetimes = prot * rng.normal(meanLifetime, stdLifetime, nspot)
        if activity == 'diffrot':
            latitudes = rng.choice([minLatitudes, maxLatitudes], size=nspot)
        else:
            # Draw from a cos distribution
            latitudes = rng.uniform(np.sin(np.radians(minLatitudes)), np.sin(np.radians(maxLatitudes)), nspot)
            latitudes = np.degrees(np.asin(latitudes))
        longitudes = rng.uniform(0.0, 360.0, nspot)
        contrasts = rng.normal(meanContrast, stdContrast, nspot)

        # Only positive radii and lifetimes
        radii[radii <= 0] = meanRadius
        lifetimes[lifetimes <= 0] = prot * meanLifetime

        config['Radius'] = radii
        config['Latitude'] = latitudes
        config['Longitude'] = longitudes
        config['Lifetime'] = lifetimes
        config['TimeMax'] = maxTimes
        config['Contrast'] = contrasts
        config = check_spots(config, prot, minLatitudes, maxLatitudes, activity)

    return config


def check_spots(config, prot, minLatitudes, maxLatitudes, activity):
    # Check for and resolve overlapping spots as psls can't handle them
    Spot = config.copy()
    Spot['dOmega'] = 0
    Spot['MuStar'] = 0.59
    Spot['MuSpot'] = 0.78
    Spot['Modulation'] = 0
    Star = {'SurfaceRotationPeriod': prot,
            'Inclination': 90}
    Duration = 8 * 90
    spot = psls.prepare_spot_parameters(Star, Spot, Duration, seed=0, verbose=False)
    defoo = params = spot
    prot = np.exp(params[1])

    # cadence used for the spot modelling (in days)
    cadence_lc = prot / 100.  # in days

    # number of points in the long cadence  light curve
    n_lc = int(np.ceil(Duration / cadence_lc))

    # time in days (for the long cadence LC)
    t = np.arange(n_lc) * cadence_lc

    nspots = int(params[0])
    inispots = [psls.spotintime.OneSpot(t, defoo[1], Domega=defoo[3], rsp=defoo[4], latsp=defoo[4 + nspots],
                                        lonsp=defoo[4 + 2 * nspots], t0=defoo[4 + 3 * nspots],
                                        lifetime=defoo[4 + 4 * nspots],
                                        fs=defoo[4 + 5 * nspots])]
    if nspots > 1:
        for i in range(1, nspots):
            ispot = psls.spotintime.OneSpot(t, defoo[1], Domega=defoo[3], rsp=defoo[4 + i], latsp=defoo[4 + nspots + i],
                                            lonsp=defoo[4 + 2 * nspots + i], t0=defoo[4 + 3 * nspots + i],
                                            lifetime=defoo[4 + 4 * nspots + i], fs=defoo[4 + 5 * nspots + i])
            if activity == 'diffrot':
                pass
            else:  # Save memory and time by only keeping start and end times
                mask = ispot.duree >= 1e-9
                idx = [0, *np.where(mask)[0][[0,-1]]]
                ispot.t = ispot.t[idx]
                ispot.psi = ispot.psi[idx]
                ispot.duree = ispot.duree[idx]
            inispots.append(ispot)
    else:
        return config
    counter = {_:0 for _ in range(nspots)}
    have_overlap = []
    for (s1, s2) in itertools.combinations(range(nspots), 2):
        if psls.spotintime.testoverlap(inispots[s1], inispots[s2]):
            have_overlap.append((s1, s2))
            counter[s1] = 1
    while len(have_overlap) > 0:
        for s1, s2 in have_overlap:
            new_lon = (rng.uniform(5, 355) + np.rad2deg(inispots[s1].psi0)) % 360  # move away from overlap
            config['Longitude'][s1] = new_lon
            inispots[s1].psi0 = np.radians(new_lon)
            if (counter[s1] % 5) == 0:
                latitudes = rng.uniform(np.sin(np.radians(minLatitudes)), np.sin(np.radians(maxLatitudes)))
                latitudes = np.degrees(np.asin(latitudes))
                config['Latitude'][s1] = latitudes
                inispots[s1].chi = np.radians(config['Latitude'][s1])
            Omspot = 2.0 * np.pi * (
                        1.0 - inispots[s1].Domega * np.sin(inispots[s1].chi) * np.sin(inispots[s1].chi)) / prot
            inispots[s1].psi = inispots[s1].psi0 + (inispots[s1].t - inispots[s1].t0) * Omspot

        have_overlap_new = []
        for s1, _ in have_overlap:
            for s2 in range(nspots):
                if s1 == s2:
                    continue
                if psls.spotintime.testoverlap(inispots[s1], inispots[s2]):
                    have_overlap_new.append((s1, s2))
                    counter[s1] += 1
        if counter[s1] >= 100:  # Can't find suitable spot location so will remove it
            inispots[s1].alpha = 0
            config['Radius'][s1] = 0
        have_overlap = have_overlap_new
    mask = config['Radius'] != 0
    config['Radius'] = config['Radius'][mask]
    config['Latitude'] = config['Latitude'][mask]
    config['Longitude'] = config['Longitude'][mask]
    config['Lifetime'] = config['Lifetime'][mask]
    config['TimeMax'] = config['TimeMax'][mask]
    config['Contrast'] = config['Contrast'][mask]
    return config


def get_spot_config(spot_options, prot):
    config = {}
    if spot_options == 0 or spot_options is None:
        config['Enable'] = 0
    else:
        config['Enable'] = 1
        config.update(gen_spots(spot_options, prot, 8))
    return config


def make_psls_config(path, i, profile, Vmag, rot_period, inclination, spot_config, transit_config):
    config = {'Observation': {},
              'Star': {},
              'Activity': {}}
    config['Star']['ID'] = f'{i:08}'
    config['Star']['Mag'] = Vmag
    config['Star']['SurfaceRotationPeriod'] = rot_period
    config['Star']['Inclination'] = inclination
    pnum = profile.fname.split('.')[0].replace('profile', '')
    hnum = profile.LOGS.split('/')[-2]
    config['Star']['ModelName'] = f'{hnum}_{pnum}_{rot_period:.1f}.modes'

    config['Activity']['Spot'] = spot_config
    config['Transit'] = transit_config

    # Generate hash
    # config['Observation']['MasterSeed'] = hash(f'{i}{Vmag}{rot_period}{inclination}') % 2**32
    config['Observation']['MasterSeed'] = hash(f'{pnum}{hnum}') % 2 ** 32

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
            rot_freq = 1/rot_period
            new_omega_rot_str = f'omega_rot = {rot_freq:g}'.replace('e-', 'd-').replace('e+', 'd+')
            gyre_in = default_gyre_in.replace('omega_rot = 0d0', new_omega_rot_str)
            with open(f'{gyre_out_dir}/{fname}', 'w') as handle:
                handle.write(gyre_in)


def worker(values):
    i, p, Vmag, rel_rotation, inclination, spot_options, transit_type = values
    star_id = int(j * 1e8) + i
    rot_period = calc_rotation(p, rel_rotation)
    spot_config = get_spot_config(spot_options, rot_period)
    transit_config = get_transit_config(p, transit_type)

    make_psls_config(f'{out_dir}/{kind[j]}/{star_id:08}.yaml', star_id, p, Vmag, rot_period, inclination, spot_config,
                     transit_config)


if __name__ == "__main__":
    nworker = mp.cpu_count()
    os.environ['OMP_NUM_THREADS'] = '1'
    generate_gyre_configs()
    kind = ['general', 'special']
    for j, iters in enumerate([iters_general]):#, iters_special_case]):
        print(f'Making {kind[j]} configs.')
        os.makedirs(f'{out_dir}/{kind[j]}', exist_ok=True)
        if kind == 'general':
            num = num_general
        else:
            num = None

        args = [[i, *a] for i, a in enumerate(iters)]
        with mp.Pool(nworker) as pool, tqdm.tqdm(total=len(args)) as pbar:
            for res in pool.imap_unordered(worker, args):
                pbar.update(1)
