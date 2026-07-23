#! /usr/bin/bash

base_gyre_in="$1"
fname=$(basename "$1")
dirnum=${fname:0:4}
pnum=${fname:5:1}
rot=${fname:7:4}

gyre-driver 0123 MESA "grid/$dirnum/LOGS/profile$pnum.data.GYRE" --base-in "$base_gyre_in" \
  --summary-item-list 'l,n_pg,n_p,n_g,m,freq,E_norm,M_star,R_star,L_star,E' --out-dir grid/$dirnum/rot \
  --in-dir gyre_in/$dirnum-$pnum-$rot --summary-suffix ."$rot".sgyre_l --rotation --f-nfreq 5 --no-output
