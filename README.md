# MSAP1

This repository contains the code required to generate synthetic lightcurves to test MSAP1.
To generate from scratch, MESA 24.08.1 and gyre 9.1.1 are required then run `make_inputs.sh`.
Otherwise, download `input.tar.gz` and extract in the top directory.

To create the lightcurves, run `psls.py` in `src/psls/psls-1.9/psls.py` and not the one from 
https://sites.lesia.obspm.fr/psls/ as the version in this repository has various changes and additions
which are needed to create these lightcurves.

To install, first clone this repository and then run the following:
```
cd MSAP1gen
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```
Once everything is installed, simply run
```
./make_inputs.sh
```
to run MESA, gyre, and the scripts which generate the input directories.

To create the lightcurves, run as follows: 
```
confignum=00000000
./src/MSAP1gen/run.py -d ./input/general/$confignum -c ./input/general/$confignum/psls.yaml --psls-args '\-V --hdf5 --skip-spot-overlap' --run-only
```
As the input directories have already been created we run with `--run-only`, 
similarly, we pass `--skip-spot-overlap` to `psls.py` as these checks have already been performed
when creating the config files. To see the rest of the available options run
`./src/MSAP1gen/run.py -h` and `./src/psls/psls-1.9/psls.py -h`.

Input files starting with 0 are general cases, i.e. they explore the parameter space defined in `generate_settings.py`.
Inputs starting with 1 are special cases investigating a specific thing.
