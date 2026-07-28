#!/usr/bin/env python
import math
import os
import shutil

import numpy as np
import yaml
import copy

from scipy import interpolate as ip
from scipy.special import assoc_legendre_p
from wsssss import load_data as ld
from platoconstants import cgs
from platoconstants import cs

from MSAP1gen import common
from common import PSLS_DIR, read_yaml

astero_TEFF_SUN = 5777
LOGG_SUN = math.log10(cgs.SOLAR_MASS_PARAMETER) - 2.*math.log10(cgs.SOLAR_RADIUS)
FWHM_to_sigma = 2*np.sqrt(2*np.log(2))
MAX_ELL = 3
this_dir = f'{os.path.dirname(__file__)}'

def replace_PSLS_DIR(config):
    config['Instrument']['Systematics']['Table'] = config['Instrument']['Systematics']['Table'].replace('PSLS_DIR', PSLS_DIR)
    config['Star']['ModelDir'] = config['Star']['ModelDir'].replace('PSLS_DIR', PSLS_DIR)
    config['External']['FilePath'] = config['External']['FilePath'].replace('PSLS_DIR', PSLS_DIR)
    return config


def new_default_yaml():
    new = read_yaml(f'{this_dir}/default.yaml')
    new = replace_PSLS_DIR(new)
    return new


def flatten(input_dict, parents=None):
    items = []
    for key, value in input_dict.items():
        if isinstance(value, dict):
            if parents is None:
                items.extend(flatten(value, parents=[key]))
            else:
                items.extend(flatten(value, parents=[*parents, key]))
        else:
            if parents is None:
                items.append([key, value])
            else:
                items.append([*parents, key, value])
    return items


def update_config(new_config, old_config=None):
    """
    Nested dict update. If old_config is None, load the default yaml and then update using new_config.
    :param new_config:
    :param old_config:
    :return:
    """
    if old_config is None:
        config = new_default_yaml()
    else:
        config = copy.deepcopy(old_config)
    if isinstance(new_config, str):
        new_config = read_yaml(new_config)
    to_update = flatten(new_config)
    for option in to_update:
        keys = option[:-1]
        value = option[-1]

        depth = len(keys)

        # Deepest nesting in config has depth=4
        if depth == 0:
            raise ValueError(f'Zero depth: {option}')
        elif depth == 1:
            config[keys[0]] = value
        elif depth == 2:
            config[keys[0]][keys[1]] = value
        elif depth == 3:
            config[keys[0]][keys[1]][keys[2]] = value
        elif depth == 4:
            config[keys[0]][keys[1]][keys[2]][keys[3]] = value
        else:
            raise ValueError(f'Maximum depth exceeded: {option}')
    config = replace_PSLS_DIR(config)
    return config


def get_stellar_params_from_gyre(gs):
    M = gs.get('M_star')[0] / cgs.SOLAR_MASS
    R = gs.get('R_star')[0] / cgs.SOLAR_RADIUS
    L = gs.get('L_star')[0] / cgs.SOLAR_LUMINOSITY
    Teff = (L * cgs.SOLAR_LUMINOSITY / (4 * np.pi * cgs.STEFAN_BOLTZMANN * R**2 * cgs.SOLAR_RADIUS**2))**0.25

    nu_max = cs.solar_seismic.nu_max * (M/R**2) / np.sqrt(Teff/astero_TEFF_SUN)
    delta_nu = cs.solar_seismic.Delta_nu * np.sqrt(M/R**3)

    return M, R, L, Teff, nu_max, delta_nu


# Table 1 in Ball et al. 2018
_table1_Ball2018 = np.array([
            [-3.710e0, 1.073e-3, 1.883e-4],
            [-7.209e1, 1.543e-2, 9.101e-4],
            [-2.266e-1, 5.083e-5, 2.715e-6],
            [-2.190e3, 4.302e-1, 8.427e-1],
            [-5.639e-1, 1.138e-4, 1.312e-4]
                ])
def calc_width_params(Teff, nu_max):
    return np.dot(_table1_Ball2018, np.array([1, Teff, nu_max]))


_eps_ml_part = np.zeros((MAX_ELL+1, 2*MAX_ELL+1))
for l, _m in enumerate(np.linspace(-l, l, 2*l+1, dtype=int) for l in range(MAX_ELL+1)):
    for m in _m:
        _eps_ml_part[l][m] = math.factorial(l - abs(m)) / math.factorial(l + abs(m))


def calc_mode_width_heights_Samadi19(gs, inclination):
    M, R, L, Teff, nu_max, delta_nu = get_stellar_params_from_gyre(gs)
    gamma_envelope = 0.66 * nu_max ** 0.88  # Mosser 2012

    nu = gs.get('Re(freq)')
    ell = gs.get('l')
    if 'm' not in gs.columns:
        m = np.zeros_like(ell)
    else:
        m = gs.get('m')
    mask_l0 = ell == 0

    # Samadi 2019 section 3.1

    # Gaussian envelope
    G_nu = np.exp(-(nu - nu_max)**2/(gamma_envelope**2 / (4*np.log(2))))
    # Mode visibilities Lund et al. 2026
    V_l = np.array([1.0, 1.54, 0.51, 0.10])[ell]
    # Relative azimuthal mode visibility
    eps_lm = _eps_ml_part[ell, m] * assoc_legendre_p(ell, m, np.cos(inclination)).flatten() ** 2
    # Corsaro 2013 scaling relations
    A_max_bol_sol = 2.53
    s = 0.748
    r = 3.47
    t = 1.27
    lnbeta = 0.321
    A_max_bol = A_max_bol_sol * np.exp((2*s - 3*t)*np.log(nu_max/cs.solar_seismic.nu_max) +
                              (4*t - 4*s)*np.log(delta_nu/cs.solar_seismic.Delta_nu) +
                              (5*s - 1.5*t - r + 0.2)*np.log(Teff/astero_TEFF_SUN) +
                              lnbeta)
    # RMS mode amplitude
    A_max = A_max_bol * (Teff / 5934)**-0.8

    # Mode widths
    Gamma_max = 0.20 + 0.97 * (Teff/astero_TEFF_SUN)**13.0  # eq. 17
    A = np.array([2, 6])[(nu<nu_max).astype(int)]
    gamma_nu_nlm = 1 + A * (1 - np.exp(-(nu - nu_max)**2/((2*gamma_envelope)**2 / (4*np.log(2)))))
    Gamma_nlm = Gamma_max * (ip.interp1d(gs.get('Re(freq)')[mask_l0], gs.get('E_norm')[mask_l0])(nu_max) / gs.get('E_norm')) * gamma_nu_nlm

    # Mode height at nu_max
    H_max = 2 * A_max**2 / (np.pi * Gamma_max)

    H_nlm = G_nu * V_l**2 * eps_lm * H_max

    return nu, H_nlm, Gamma_nlm

def calc_mode_width_heights_Ball18(gs, inclination):
    M, R, L, Teff, nu_max, delta_nu = get_stellar_params_from_gyre(gs)
    gamma_envelope = 0.66 * nu_max ** 0.88  # Mosser 2012

    nu = gs.get('Re(freq)')
    ell = gs.get('l')
    if 'm' not in gs.columns:
        m = np.zeros_like(ell)
    else:
        m = gs.get('m')
    mask_l0 = ell == 0
    Q_nl = gs.get('E_norm') / ip.interp1d(gs.get('Re(freq)')[mask_l0], gs.get('E_norm')[mask_l0])(gs.get('Re(freq)'))

    # Using Ball et al. 2018 for heights and widths
    # https://ui.adsabs.harvard.edu/link_gateway/2018ApJS..239...34B/arxiv:1809.09108

    alpha, gamma_alpha, Delta_gamma_dip, nu_dip, W_dip = calc_width_params(Teff, nu_max)
    if gamma_alpha <= 0:
        print(M, R, L, Teff, nu_max, delta_nu)
        print(Teff, nu_max, alpha, gamma_alpha, Delta_gamma_dip, nu_dip, W_dip)
        raise ValueError(f'gamma_alpha < 0')
    if Delta_gamma_dip <= 0:
        print(Teff, nu_max, alpha, gamma_alpha, Delta_gamma_dip, nu_dip, W_dip)
        raise ValueError(f'Delta_gamma_dip < 0')
    # Mode width Eq. 13
    ln_gamma = alpha * np.log(nu/nu_max) + np.log(gamma_alpha) + np.log(Delta_gamma_dip) / \
               (1 + ((2 * np.log(nu/nu_dip))/np.log(W_dip/nu_max))**2)

    ln_gamma = ln_gamma - np.log(Q_nl)
    gamma_nl = np.exp(ln_gamma)

    # Mode height

    Teff_red = 8907 * L ** -0.093
    Delta_T = 1250
    beta = 1 - np.exp((Teff - Teff_red) / Delta_T)

    # Mode height Eq. 16
    A_rms_max_sun = 2.53
    A_rms_max = A_rms_max_sun * beta * (L/M) * (astero_TEFF_SUN / Teff)**2

    # Mode visibilities Lund et al. 2026
    Vl = np.array([1.0, 1.54, 0.51, 0.10])

    eps_lm = _eps_ml_part[ell, m] * assoc_legendre_p(ell, m, np.cos(inclination)).flatten()**2

    H_nl = 2 * Vl[ell] * eps_lm * A_rms_max**2 / (np.pi * gamma_nl) * np.exp(-4*math.log(2) * ((nu - nu_max)/gamma_envelope)**2)
    H_nl = H_nl / Q_nl

    return nu, H_nl, gamma_nl


def convert_gyre(inclination, gs_path, out_path):
    gs = ld.GyreSummary(gs_path)

    if check_gs(gs):
        raise ValueError(f'GyreSummary {gs_path} has missing modes.')
    # Keep only modes between min and max freq l0
    mask_l0 = gs.get('l') == 0
    nu = gs.get('Re(freq)')
    nu_0_min, nu_0_max = nu[mask_l0][[0, -1]]
    mask = (nu >= nu_0_min) & (nu <= nu_0_max)
    gs.data = gs.data[mask]

    M, R, L, Teff, nu_max, delta_nu = get_stellar_params_from_gyre(gs)

    if os.path.exists(out_path):  # Already done
        return M, R, L, Teff, nu_max, delta_nu

    nu, H_nl, gamma_nl = calc_mode_width_heights_Samadi19(gs, inclination)  # More stable
    # nu, H_nl, gamma_nl = calc_mode_width_heights_Ball18(gs, inclination)

    freq = nu
    width = gamma_nl
    height = H_nl

    dat = np.stack([freq, width, height], axis=1)
    np.savetxt(out_path, dat, delimiter=' ', fmt='%.6f', header='# nu [muHz]  Gamma [muHz]  H [ppm2/muHz]')
    return M, R, L, Teff, nu_max, delta_nu


def check_gs(gs):
    if gs is None:
        return True
    have_missing = False
    for l in [1, 2, 3]:
        for m in np.arange(-l, l+1):
            mask = (gs.data.l == l) & (gs.data.m == m)
            n_p = gs.data.n_p[mask]
            n_g = gs.data.n_g[mask]
            n_pg = gs.data.n_pg[mask]
            if l == 1:
                mask_ng = n_p < n_g
                have_missing = not (np.all(np.diff(n_pg[mask_ng]) == 1) and np.all(np.diff(n_pg[~mask_ng]) == 1))
            else:
                have_missing = not np.all(np.diff(n_pg) == 1)
            if have_missing:
                return have_missing
    return have_missing


def setup(dirname, config, gs_path, seed, fname='psls.yaml'):
    if isinstance(config, list):
        _config = None
        for sub_config in config:
            _config = update_config(sub_config, _config)
        config = _config
    else:
        config = update_config(config)

    if gs_path is None:
        hist_num, prof_num, rot = config['Star']['ModelName'].split('_')
        rot = rot[:4]
        gs_path = f'{this_dir}/../MESA/grid/{hist_num}/rot/profile{prof_num}.data.GYRE.{rot}.sgyre_l'
        if not os.path.exists(gs_path):
            raise FileNotFoundError(f'GyreSummary file not found: {gs_path}')

    freq_file_path = f"../../freqs/{config['Star']['ModelName']}"
    os.makedirs(dirname, exist_ok=False)
    os.makedirs(f'{dirname}/../../freqs', exist_ok=True)
    try:
        if gs_path is not None:
            M, R, L, Teff, nu_max, Delta_nu = convert_gyre(config['Star']['Inclination'], gs_path, out_path=f"{dirname}/{freq_file_path}")
            config['Star']['Logg'] = float(np.log10(M/R**2) + LOGG_SUN)
            config['Star']['Teff'] = float(Teff)
            config['Oscillations']['numax'] = nu_max
            config['Oscillations']['deltanu'] = Delta_nu
            config['Star']['ModelName'] = freq_file_path
    except ValueError:
        shutil.rmtree(dirname, ignore_errors=True)
        raise

    if seed == 0:
        pass
    elif seed == 1:  # Generate a MasterSeed using contents of config
        config['Observation']['MasterSeed'] = hash(yaml.dump(config, sort_keys=True)) % 2**32
    else:
        config['Observation']['MasterSeed'] = int(seed)
    with open(f'{dirname}/{fname}', 'w') as handle:
        handle.write(common.make_yaml_str(config))
