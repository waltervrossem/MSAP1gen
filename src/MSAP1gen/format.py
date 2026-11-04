import os.path

import h5py
import numpy as np

from scipy.interpolate import splrep, splev

from py_sas_data_modeler.dm import DataClassFactory
from py_sas_data_modeler.io import HDF5StorageSasWorkflow
from py_sas_data_modeler.workflow import SasPipeline


TEFF_SUN = 5777.0 # Solar effective temperature [K]
LOGG_SUN = 4.43 # Solar surface gravity [dex]
DEFAULT_GAIA_COLOR_ERROR = 0.01 # Default error on Gaia color when it cannot be computed from normal relation


def gaia_color(teff_obj):
    """
    Compute GAIA color from John V magnitude and effective temperature.
    Polynomial fit from Marchiori et al. (2019) are used.
    """
    if isinstance(teff_obj, float):
        teff = teff_obj
    else:
        teff = teff_obj.value
    teff_func = lambda x: -1.184e-12*x**3 + 4.526e-8*x**2 - 5.805e-4*x + 2.449
    gaia_func = lambda x: -0.00292*x**3   + 0.10027*x**2  + 0.37919*x  + 0.00267
    vp = teff_func(teff)
    gaia_arr = np.linspace(-0.227, 4.524, num=1000)
    if vp <= gaia_func(-0.227):
        print('Gaia color falls outside range of validity, defaulting to -0.227')
        return -0.227
    elif gaia_func(4.524) <= vp:
        print('Gaia color falls outside range of validity, defaulting to 4.524')
        return 4.524
    color = splev(vp, splrep(gaia_func(gaia_arr), gaia_arr))
    return color

def error_gaia_color(teff_obj):
    """
    Compute uncertainty on GAIA color from John V magnitude and effective temperature.
    Polynomial fit from Marchiori et al. (2019) are used.
    """
    teff = teff_obj.value
    dTeff = teff_obj.error
    colorInf = gaia_color(teff - dTeff)
    colorSup = gaia_color(teff + dTeff)
    if colorInf != colorSup:
        dColor = np.abs(colorSup - colorInf) / 2.0
    else:
        dColor = DEFAULT_GAIA_COLOR_ERROR
    return dColor


def format_MSAP1_in(input_file, output_file):
    if os.path.exists(output_file):
        raise FileExistsError(f'Output file already exists: {output_file}')
    with h5py.File(input_file, 'r') as f:
            # O: oscillations (ppm)
            # G: granulation (ppm)
            # A: (stochastic) activity (ppm)
            # RSE: residual systematic error (ppm)
            # R: random error (ppm)
            # S: spot (normalized flux)
            # T: transit (normalized flux)
            # s = 10^(-6)
            # INDIVIDUAL_LC = ( ( 1 + ( O + G + A ) * s ) * ( 1 + (RSE + R)*s ) * S * T – 1)/s
            s = 1e-6
            if 'OSCILLATIONS_LC' in f.keys():
                O = f['OSCILLATIONS_LC'][()]
            else:
                O = 0
            if 'GRANULATION_LC' in f.keys():
                G = f['GRANULATION_LC'][()]
            else:
                G = 0
            if 'ACTIVITY_LC' in f.keys():
                A = f['ACTIVITY_LC'][()]
            else:
                A = 0
            if 'EXTERNAL_LC' in f.keys():
                E = 1 + f['EXTERNAL_LC'][()] * s
            else:
                E = 1
            if 'SYSTEMATICS_LC' in f.keys():
                RSE = f['SYSTEMATICS_LC'][()]
            else:
                RSE= np.zeros((1,1,1))  # Can't be scalar 0 due to mean in CAMGROUP_LC calculation
            if 'RANDOM_LC' in f.keys():
                R = f['RANDOM_LC'][()]
            else:
                R = np.zeros((1,1,1))  # Can't be scalar 0 due to mean in CAMGROUP_LC calculation
            if 'SPOT_LC' in f.keys():
                S = f['SPOT_LC'][()]
            else:
                S = 1
            if 'FLARES_LC' in f.keys():
                F = f['FLARES_LC'][()]
            else:
                F = 1
            if 'TRANSIT_LC' in f.keys():
                T = f['TRANSIT_LC']
            else:
                T = 1
            CAMGROUP_LC = ((np.ones(f['TIME'].shape) + (O + G + A) * s) * np.mean(1 + (RSE + R) * s, axis=2) * S * F * T * E - 1)/s  # Within 1e-10 of PSLS output

            time = f['TIME'][()]
            time = time.reshape(np.prod(time.shape))
            dat = {'ONBOARD_TIME': time,
                   'BARYCENTRIC_TIME':  26299 + time/86400,
                   'RFLUX': 1+CAMGROUP_LC.reshape(time.shape)*s,
                   'Cadence': 25.0,
                   'QUARTER': np.floor_divide(time-time[0], 90.0*86400).astype(int) + 1,
                   'STATUS': np.zeros_like(time, dtype=int)}

            # Create external data
            dp1_obj = DataClassFactory.get_model('DP1_LC_MERGED')(**dat)
            Teff_obj = DataClassFactory.get_model("IDP_PFU_TEFF_SAPP")(value=metadata_psls["teff"], error=50)
            logg_obj = DataClassFactory.get_model("IDP_PFU_LOGG_SAPP")(value=metadata_psls["logg"], error=0.1)
            rad_obj = DataClassFactory.get_model("IDP_PFU_RADIUS_CLASSICAL")(value=metadata_psls["radius"], error=0.1*metadata_psls["radius"])
            _bprp_obj = DataClassFactory.get_model("GaiaBpRpDM")(value=float(gaia_color(Teff_obj)), error=float(error_gaia_color(Teff_obj)))
            bprp_obj = DataClassFactory.get_model("LG_PFU_OBS_DATA")(gaia_bp_rp=_bprp_obj, spec_path=[""])

            Teff_SAS_obj = DataClassFactory.get_model("IDP_SAS_TEFF_SAPP")(value=metadata_psls["teff"], error=50)
            logg_SAS_obj = DataClassFactory.get_model("IDP_SAS_LOGG_SAPP")(value=metadata_psls["logg"], error=0.1)
            # Create a SAS storage for the SAS workflow
            hdf5 = HDF5StorageSasWorkflow(output_file)

            # Save the external product into the SAS storage
            sas = SasPipeline(hdf5)
            sas.save_external(dp1_obj)
            sas.save_external(Teff_obj)
            sas.save_external(logg_obj)
            sas.save_external(Teff_SAS_obj)
            sas.save_external(logg_SAS_obj)
            sas.save_external(rad_obj)
            sas.save_external(bprp_obj)
