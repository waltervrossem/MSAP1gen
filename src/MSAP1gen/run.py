#!/usr/bin/env python

import os
import argparse
import create_config as cg
from common import PSLS_DIR, temp_chdir


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
    parser.add_argument('--tee', '-t', action='store_const', const=True, default=False,
                        help="Capture output to out.txt.")
    parser.add_argument('--format', action='store_const', const=True, default=False,
                        help="Put lightcurves into correct format for MSAP1.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random number generator. If 0, use default defined in default.yaml. If 1 create"
                             "a seed from hash of config, otherwise use the value passed.")
    return parser


def run(args):
    dirname = args.dirname
    configname = args.config
    fname = args.fname
    gs_path = args.gs_path
    psls_args = args.psls_args
    capture_output = args.tee
    seed = args.seed

    cg.setup(dirname, configname, gs_path, seed, fname)
    
    cmd = f'python {PSLS_DIR}/psls.py {psls_args} {fname}'
    if capture_output:
        cmd += ' | tee out.txt'
    with temp_chdir(dirname):
        print(os.path.abspath(os.path.curdir))
        ierr = os.system(cmd)
        if ierr != 0:
            raise RuntimeError(f'Error running psls.py.')

if __name__ == "__main__":
    args = get_parser().parse_args()
    run(args)
