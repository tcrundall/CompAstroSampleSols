import copy
import matplotlib.pyplot as plt
import numpy as np

import sph
import sys

if __name__ == '__main__':

    nparts = 100

    eta = int(sys.argv[1])
    CFL = float(sys.argv[2])

    run_pars = {
        'eta':eta,
        'CFL':CFL,
        'tmax':0.3,
        'plt_interv':0.1,
        'output_dir':'output/free_expansion/%i_%.1f/'%(eta, CFL),
    }

    min_x = 0.
    max_x = 1.
    dx = max_x - min_x

    dens = 1.
    eint = 1.
    dx = 1.

    mass_tot = dx * dens
    mass_per_part = mass_tot / nparts

    parts = {
        'pos':np.linspace(min_x, max_x, nparts, endpoint=False),
        'mass':np.ones(nparts),
        'vel':np.zeros(nparts),
        'eint':np.ones(nparts) * eint,
        'dvdt': np.zeros(nparts),
        'dudt': np.zeros(nparts),
        'neigh':{},
    }

    sph.run_simulation(parts, **run_pars)
