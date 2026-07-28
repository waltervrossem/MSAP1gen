#! /usr/bin/bash
set -e
date "+DATE: %Y-%m-%d%nTIME: %H:%M:%S"
if grep -wq "9.1.1" "$GYRE_DIR/src/common/version_m.fypp"; then
  true
else
  echo "Run with GYRE 9.1.1"
  exit 1
fi

# Need nproc --all as setting OMP_NUM_THREADS seems to set nproc to that value too
export OMP_NUM_THREADS=2
n_gyre=$(( $(nproc --all) / $OMP_NUM_THREADS ))
find ../../configs/gyre/ -name "*.in" | sort | xargs -n 1 -P$n_gyre ./do_one_gyre.sh
