#! /usr/bin/bash

base_gyre_in="$1"
dirnum=${base_gyre_in:17:4}
pnum=${base_gyre_in:22:1}
rot=${base_gyre_in:24:4}
echo base_gyre_in=$base_gyre_in
gyre-driver 0123 MESA "grid/$dirnum/LOGS/profile$pnum.data.GYRE" --base-in "$base_gyre_in" \
  --summary-item-list 'l,n_pg,n_p,n_g,m,freq,E_norm,M_star,R_star,L_star,E' --out-dir grid/$dirnum/rot \
  --summary-suffix "$rot".sgyre_l --rotation
