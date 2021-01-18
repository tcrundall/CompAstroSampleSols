import copy
import numpy as np
import sys

import sph

if __name__ == '__main__':

    nparts = int(sys.argv[1])       # 100, 500, 1000
    alpha = float(sys.argv[2])      # 1, 0.1, 0.01
    CFL = 0.1
    run_pars = {
        'eta':3,
        'CFL':CFL,
        'tmax':0.3,
        'plt_interv':0.025,
        'output_dir':'output/debug_visc/%i_%.2f_%.2f/'%(nparts, alpha, CFL),
        'use_visc':True,
        'alpha':alpha,
        'setup_only':False,
    }

    x_left = -1
    x_right = 1
    x_front = 0

    dens_left = 1.
    dens_right = 0.25
    # dens_right = 1.
    # dens_right = 0.8
    pres_left = 1.
    pres_right = 0.2
    # pres_right = 1.
    # pres_right = 0.7


    nparts = 2

    parts = {
        'pos':np.linspace(-1,1,nparts),
        'mass':np.ones(nparts),
        'vel':np.linspace(1,-1,nparts),
        'eint':np.ones(nparts),
        'dvdt':np.zeros(nparts),
        'dudt':np.zeros(nparts),
        'neigh':{},
    }

    init_parts = copy.deepcopy(parts)


    sph.run_simulation(parts, **run_pars)
