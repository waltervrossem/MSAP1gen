
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
            CAMGROUP_LC = ((np.ones(f['TIME'].shape) + (O + G + A) * s) * np.mean(1 + (RSE + R) * s, axis=2) * S * F * T * E - 1)  # Within 1e-10 of PSLS output
            o.create_dataset('TIME', data=f['TIME'])
            o.create_dataset('FLUX', data=1+CAMGROUP_LC*s)
            if 'TRANSIT_LC' in f.keys():
                o.create_dataset('TRANSIT_REL', data=np.mean(f['TRANSIT_LC'], axis=1))
