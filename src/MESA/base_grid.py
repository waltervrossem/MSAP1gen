#!/usr/bin/env python

import os
import numpy as np

from wsssss.inlists import create_grid as cg
from wsssss.inlists import inlists as inl

grid_name = 'grid'
grid = cg.MesaGrid(add_base_workdir=True, inlists_index=4)

mixture_inlist = 'inlist_a09'

grid.add_dir('src')
grid.add_file('history_columns.list')
grid.add_file('profile_columns.list')
grid.add_file('inlist_common')
grid.add_file(mixture_inlist)

grid.inlist['star_job']['read_extra_star_job_inlist(1)'] = True
grid.inlist['star_job']['extra_star_job_inlist_name(1)'] = 'inlist_common'
grid.inlist['star_job']['read_extra_star_job_inlist(2)'] = True
grid.inlist['star_job']['extra_star_job_inlist_name(2)'] = mixture_inlist

grid.inlist['controls']['read_extra_controls_inlist(1)'] = True
grid.inlist['controls']['extra_controls_inlist_name(1)'] = 'inlist_common'
grid.inlist['controls']['read_extra_controls_inlist(2)'] = True
grid.inlist['controls']['extra_controls_inlist_name(2)'] = mixture_inlist

grid.inlist['kap']['read_extra_kap_inlist(1)'] = True
grid.inlist['kap']['extra_kap_inlist_name(1)'] = 'inlist_common'
grid.inlist['kap']['read_extra_kap_inlist(2)'] = True
grid.inlist['kap']['extra_kap_inlist_name(2)'] = mixture_inlist

# Add Imbriani 14N(p,g)15O
grid.add_dir('rates_tables')
grid.add_file('rate_list.txt')

# Add custom options here:
# e.g.
grid.controls['initial_mass'] = np.linspace(0.5, 1.5, 11)
grid.controls['xa_central_lower_limit_species(1)'] = 'h1'
grid.controls['xa_central_lower_limit(1)'] = 0.01
grid.controls['photo_interval'] = 50
grid.controls['max_timestep_factor'] = 5

if __name__ == "__main__":
    if not os.path.exists(grid_name):
        grid.create_grid(grid_name)
        os.system(f'cp {__file__} {grid_name}/')
