#!/usr/bin/env python

import os
import argparse
import create_config as cg
from common import PSLS_DIR, temp_chdir



def run(dirname, configname='config.yaml', fname='psls.yaml', psls_config='-V -P', capture_output=False):
    cg.setup(dirname, configname, fname)
    cmd = f'python {PSLS_DIR}/psls.py {psls_config} {fname}'
    if capture_output:
        cmd += ' | tee out.txt'
    with temp_chdir(dirname):
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', '-d', type=str,
                        help="Working directory in which to create config file and run psls.py.")
    parser.add_argument('--config', '-c', type=str,
                        help="Path to configuration file with which to update default settings.")
    parser.add_argument('--fname', '-f', type=str,
                        help="Filename of output configuration file.")
    parser.add_argument('--psls-config', type=str, default = '-V -P',
                        help="Arguments for psls.py")
    parser.add_argument('--tee', '-t', action='store_const', const=True, default=False,
                        help="Capture output to out.txt.")
    args = parser.parse_args()
    run(dirname=args.dirname, configname=args.config, fname=args.fname, psls_config=args.psls_config, capture_output=args.tee)
