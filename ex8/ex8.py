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
        'output_dir':'output/final_res/%i_%.2f_%.2f/'%(nparts, alpha, CFL),
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

#     parts = {
#         'pos':np.linspace(-1,2,3*nparts),
#         'mass':np.ones(3*nparts),
#         'vel':np.zeros(3*nparts),
#         'eint':np.ones(3*nparts),
#         'neigh':{},
#     }
#
#     parts['pos'] = np.linspace(-1,2,3*nparts)
#
#     midpoint = 3*nparts//2
#     parts['mass'][:midpoint] = dens_left / nparts
#     parts['mass'][midpoint:] = dens_right / nparts
#
#     parts['eint'][:midpoint] = pres_left/(dens_left*(sph.GAMMA - 1))
#     parts['eint'][midpoint:] = pres_right/(dens_right*(sph.GAMMA - 1))

    # sph.run_simulation(parts, **run_pars)


    mass_left = (x_front - x_left)*dens_left
    mass_right = (x_right - x_front)*dens_right

    mass_total = mass_left + mass_right
    mass_per_part = mass_total / nparts

    nparts_left = round(mass_left / mass_per_part)
    nparts_right = round(mass_right / mass_per_part)

    eint_left = pres_left/(dens_left*(sph.GAMMA -1))
    eint_right = pres_right/(dens_right*(sph.GAMMA -1))

    parts = {
        'pos':np.zeros(nparts),
        'mass':np.ones(nparts) * mass_per_part,
        'vel':np.zeros(nparts),
        'dvdt':np.zeros(nparts),
        'dudt':np.zeros(nparts),
        'neigh':{},
    }

    parts['pos'][:nparts_left] = np.linspace(x_left, x_front, nparts_left, endpoint=False)
    parts['pos'][nparts_left:] = np.linspace(x_front, x_right, nparts_right, endpoint=False)

    parts['eint'] = np.zeros(nparts)
    parts['eint'][:nparts_left] = eint_left
    parts['eint'][nparts_left:] = eint_right

    init_parts = copy.deepcopy(parts)


    sph.run_simulation(parts, **run_pars)
