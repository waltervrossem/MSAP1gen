
import h5py
import numpy as np


def format_MSAP1_in(input_file, output_file):
    with h5py.File(input_file, 'r') as f:
        with h5py.File(output_file, 'w') as o:
            for key, value in f.attrs.items():
                o.attrs[key] = value
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
            O = f['OSCILLATIONS_LC'][()]
            G = f['GRANULATION_LC'][()]
            if 'ACTIVITY_LC' in f.keys():
                A = f['ACTIVITY_LC'][()]
            else:
                A = 1
            E = 1 + f['EXTERNAL_LC'][()] * s
            # OGA = f['STELLAR_LC']
            RSE = f['SYSTEMATICS_LC'][()]
            R = f['RANDOM_LC'][()]
            S = f['SPOT_LC'][()] * f['FLARES_LC']
            if 'TRANSIT_LC' in f.keys():
                T = f['TRANSIT_LC']
            else:
                T = 1
            CAMGROUP_LC = ((1 + (O + G + A) * s) * np.mean(1 + (RSE + R) * s, axis=2) * S * T * E - 1) / s  # Within 1e-10 of PSLS output
            o.create_dataset('TIME', data=f['TIME'])
            o.create_dataset('FLUX', data=1+CAMGROUP_LC*s)
            if 'TRANSIT_LC' in f.keys():
                o.create_dataset('TRANSIT_REL', data=np.mean(f['TRANSIT_LC'], axis=1))
