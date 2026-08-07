#! /usr/bin/bash
set -e
echo "Creating psls input directories for MSAP1"
date "+DATE: %Y-%m-%d%nTIME: %H:%M:%S"
curdir=$(pwd)

#rm -rf configs input
#rm -rf src/MESA/gyre_in src/MESA/grid/*/rot
#rm -rf src/MESA/grid

if [ ! -d "$curdir/src/MESA/grid" ]; then
  cd "$curdir/src/MESA"
  tar -xzf "$curdir/data/grid.tar.gz"
  cd "$curdir"
fi

cd $curdir/src/MSAP1gen
python generate_settings.py
cd $curdir/src/MESA
./run_gyre.sh
cd $curdir/src/MSAP1gen
python make_all_config.py

cd $curdir
echo "Making input.tar.gz"
rm -f input.tar.gz
tar -czf input.tar.gz input
date "+DATE: %Y-%m-%d%nTIME: %H:%M:%S"
