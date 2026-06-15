#!/usr/bin/env python
import os
import yaml
import copy
import contextlib
from packaging.version import parse as parse_version


def read_yaml(path='default.yaml'):
    with open(path, 'r') as stream:
        if(parse_version(yaml.__version__)< parse_version("5.0")):
            dat = yaml.load(stream)
        else:
            dat = yaml.load(stream, Loader=yaml.FullLoader)
    return dat


def new_default_yaml():
    return read_yaml()


def flatten(input_dict, parents=None):
    items = []
    for key, value in input_dict.items():
        if isinstance(value, dict):
            if parents is None:
                items.extend(flatten(value, parents=[key]))
            else:
                items.extend(flatten(value, parents=[*parents, key]))
        else:
            if parents is None:
                items.append([key, value])
            else:
                items.append([*parents, key, value])
    return items


def update_config(new_config):
    config = new_default_yaml()
    if isinstance(new_config, str):
        new_config = read_yaml(new_config)
    to_update = flatten(new_config)
    for option in to_update:
        keys = option[:-1]
        value = option[-1]

        depth = len(keys)

        # Deepest nesting has depth=4
        if depth == 0:
            raise ValueError(f'Zero depth: {option}')
        elif depth == 1:
            config[keys[0]] = value
        elif depth == 2:
            config[keys[0]][keys[1]] = value
        elif depth == 3:
            config[keys[0]][keys[1]][keys[2]] = value
        elif depth == 4:
            config[keys[0]][keys[1]][keys[2]][keys[3]] = value
        else:
            raise ValueError(f'Maximum depth exceeded: {option}')
    return config


def setup(dirname, config, fname='psls.yaml'):
    os.makedirs(dirname, exist_ok=False)
    config = update_config(config)
    with open(f'{dirname}/{fname}', 'w') as stream:
        yaml.dump(config, stream)
