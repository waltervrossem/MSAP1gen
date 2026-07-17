
import h5py
import numpy as np


def format_MSAP1_in(input_file, output_file):
    with h5py.File(input_file, 'r') as f:
        with h5py.File(output_file, 'w') as o:
            for key, value in f.attrs.items():
                o.attrs[key] = value
            o.create_dataset('TIME', data=f['TIME'])
            o.create_dataset('FLUX', data=np.mean(f['INDIVIDUAL_LC'][:, :, :, 0], axis=2))
            if 'TRANSIT_LC' in f.keys():
                o.create_dataset('TRANSIT_REL', data=np.mean(f['TRANSIT_LC'], axis=1))
