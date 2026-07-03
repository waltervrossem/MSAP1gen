import os
import contextlib

if 'PSLS_DIR' in os.environ:
    PSLS_DIR = os.environ['PSLS_DIR']
else:
    PSLS_DIR = os.path.abspath(os.path.dirname(__file__) + '/../psls/psls-1.9')
if not os.path.exists(PSLS_DIR):
    raise NotADirectoryError(f'PSLS_DIR: {PSLS_DIR}')


@contextlib.contextmanager
def temp_chdir(dirname):
    initial_cwd = os.getcwd()
    os.chdir(os.path.abspath(dirname))
    try:
        yield
    finally:
        os.chdir(initial_cwd)


def make_yaml_str(config, indent=0):
    out = ''
    for key, value in config.items():
        out += f"{' '*indent}{key}:\n"
        indent += 4
        if isinstance(value, dict):
            out += make_yaml_str(value, indent)
        else:
            out += f"{' '*indent} {value}\n"
        indent -= 4
    return out
