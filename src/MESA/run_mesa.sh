#! /usr/bin/bash
set -e

echo "Running MESA"
date "+DATE: %Y-%m-%d%nTIME: %H:%M:%S"
# Run with MESA 24081
MESA_version=$(cat "$MESA_DIR/data/version_number")
if [ "$MESA_version" != "r24.08.1" ]; then
  echo "Must run with MESA 24.08.1"
  exit 1
fi

python base_grid.py
cd grid/0000
./mk &> /dev/null
mv star ../
cd ..

export OMP_NUM_THREADS=$(($(nproc --all) / 4))
mesa-go . --cmd-pre-each "cp ../star ./" -n 4 --skip-if-file-exists WORK_DIR/LOGS/profile9.data.GYRE

# Clean up
rm -f */star
rm -f ./star
rm -rf */.mesa_temp_cache
