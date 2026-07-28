#! /usr/bin/bash
set -e
date "+DATE: %Y-%m-%d%nTIME: %H:%M:%S"
curdir=$(pwd)

#rm -rf configs input src/MESA/gyre_in
#rm -rf src/MESA/grid/*/rot
#rm -rf src/MESA/grid

cd $curdir/src/MESA
./run_mesa.sh
cd $curdir/src/MSAP1gen
./generate_settings.py
cd $curdir/src/MESA
./run_gyre.sh
cd $curdir/src/MSAP1gen
./make_all_config.py

echo "Making tar.gz"
rm -f input.tar.gz
tar -czf input.tar.gz input
date "+DATE: %Y-%m-%d%nTIME: %H:%M:%S"
