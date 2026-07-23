#!/usr/bin/env python

import os
import time
import argparse
import create_config as cg
from common import PSLS_DIR, temp_chdir, read_yaml
from format import format_MSAP1_in


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', '-d', type=str,
                        help="Working directory in which to create config file and run psls.py.")
    parser.add_argument('--config', '-c', type=str, default='config.yaml', nargs='*',
                        help="Path to one or more configuration files with which to update default settings."
                             "They are applied in the order they are specified.")
    parser.add_argument('--gs-path', type=str, default=None,
                        help="Path to Gyre Summary file. This file must contain Mode frequencies in uHz, mode inertia,"
                             "mode degree, model mass, photospheric radius, and luminosity. If not specified, will try"
                             "to find it using ModelName in config.yaml.")
    parser.add_argument('--fname', '-f', type=str, default='psls.yaml',
                        help="Filename of output configuration file.")
    parser.add_argument('--psls-args', type=str, default = '--hdf5',
                        help="Arguments for psls.py. Note that a \\ might be required if starting with a '-'.")
    parser.add_argument('--format', action='store_const', const=True, default=False,
                        help="Put lightcurves into correct format for MSAP1.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random number generator. If 0, use value defined in configuration file. If 1 create"
                             "a seed from hash of config, otherwise use the value passed.")
    parser.add_argument('--setup-only', action='store_const', const=True, default=False,
                        help="Only generate configuration files.")
    parser.add_argument('--run-only', action='store_const', const=True, default=False,
                        help="Run psls on already generated config files.")
    return parser


def run(args):
    dirname = args.dirname
    configname = args.config
    fname = args.fname
    gs_path = args.gs_path
    psls_args = args.psls_args
    seed = args.seed

    if not args.run_only:
        cg.setup(dirname, configname, gs_path, seed, fname)

    if not args.setup_only:
        cmd = f'python {PSLS_DIR}/psls.py {psls_args} {fname}'
        with temp_chdir(dirname):
            print(os.path.abspath(os.path.curdir))
            ierr = os.system(cmd)
            if ierr != 0:
                raise RuntimeError(f'Error running psls.py.')
            if args.format:
                psls_config = read_yaml(args.fname)
                # Filename part from psls.py
                StarName = "%10.10i" % int(psls_config['Star']['ID'])
                input_file = f'{StarName}.hdf5'
                output_file = f'ref{StarName}.hdf5'
                format_MSAP1_in(input_file, output_file)


if __name__ == "__main__":
    args = get_parser().parse_args()
    ts = time.time()
    run(args)
    print('Elapsed time:', time.time() - ts)
