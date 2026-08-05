import os
import contextlib

import numpy as np
import yaml
from packaging.version import parse as parse_version

if 'PSLS_DIR' in os.environ:
    PSLS_DIR = os.environ['PSLS_DIR']
else:
    PSLS_DIR = os.path.abspath(os.path.dirname(__file__) + '/../psls/psls-1.9')
if not os.path.exists(PSLS_DIR):
    raise NotADirectoryError(f'PSLS_DIR: {PSLS_DIR}')

# JUPITER_RADIUS = (7.1492e9 + 6.6854e9) / 2
JUPITER_RADIUS = 7.1492e9  # This value is used in psls.py

@contextlib.contextmanager
def temp_chdir(dirname):
    initial_cwd = os.getcwd()
    os.chdir(os.path.abspath(dirname))
    try:
        yield
    finally:
        os.chdir(initial_cwd)


def convert_type(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.float64):
        value = float(value)
    elif isinstance(value, np.int64):
        value = int(value)
    elif isinstance(value, list) or isinstance(value, tuple):
        value = [convert_type(item) for item in value]
    return value


def make_yaml_str(config, indent=0):
    out = ''
    for key, value in config.items():
        out += f"{' '*indent}{key}:\n"
        indent += 4
        if isinstance(value, dict):
            out += make_yaml_str(value, indent)
        else:
            value = convert_type(value)
            out += f"{' '*indent} {value}\n"
        indent -= 4
    return out


def read_yaml(path):
    with open(path, 'r') as handle:
        if parse_version(yaml.__version__) < parse_version("5.0"):
            dat = yaml.load(handle)
        else:
            dat = yaml.load(handle, Loader=yaml.FullLoader)
    return dat
