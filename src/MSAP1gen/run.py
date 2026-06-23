#!/usr/bin/env python

import os
import argparse
import create_config as cg
from common import PSLS_DIR, temp_chdir


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', '-d', type=str,
                        help="Working directory in which to create config file and run psls.py.")
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help="Path to configuration file with which to update default settings.")
    parser.add_argument('--gs-path', type=str, default=None,
                        help="Path to Gyre Summary file. This file must contain Mode frequencies in uHz, mode inertia,"
                             "mode degree, model mass, photospheric radius, and luminosity.")
    parser.add_argument('--fname', '-f', type=str,
                        help="Filename of output configuration file.")
    parser.add_argument('--psls-args', type=str, default = '-V -P',
                        help="Arguments for psls.py")
    parser.add_argument('--tee', '-t', action='store_const', const=True, default=False,
                        help="Capture output to out.txt.")
    args = parser.parse_args()
    run(dirname=args.dirname, configname=args.config, fname=args.fname, psls_config=args.psls_config, capture_output=args.tee)
    return parser


def run(args):
    dirname = args.dirname
    configname = args.config
    fname = args.fname
    gs_path = args.gs_path
    psls_args = args.psls_args
    capture_output = args.tee

    cg.setup(dirname, configname, gs_path, fname)
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
