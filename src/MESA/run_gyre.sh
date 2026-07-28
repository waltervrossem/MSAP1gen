#! /usr/bin/bash
set -e

echo "Running gyre"
date "+DATE: %Y-%m-%d%nTIME: %H:%M:%S"
if grep -wq "9.1.1" "$GYRE_DIR/src/common/version_m.fypp"; then
  true
else
  echo "Run with GYRE 9.1.1"
  exit 1
fi

version_greater_equal(){
    printf '%s\n%s\n' "$2" "$1" | sort --check=quiet --version-sort
}

VERSION="$(gyre-driver --version)"
VERSION=${VERSION/"gyre-driver "/""}  # Only keep version number

if [[ $(version_greater_equal "0.5.0" "$VERSION" ) ]]; then
  true
else
  echo "Need gyre-driver 0.5.0 or above, update wsssss to at least 0.7.8 from PyPI."
  exit 1
fi

# Need nproc --all as setting OMP_NUM_THREADS seems to set nproc to that value too
mkdir -p gyre_in
export OMP_NUM_THREADS=2
n_gyre=$(( $(nproc --all) / $OMP_NUM_THREADS ))
find ../../configs/gyre/ -name "*.in" | sort | xargs -t -n 1 -P$n_gyre ./do_one_gyre.sh
