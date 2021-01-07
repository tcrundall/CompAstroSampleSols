import copy
import matplotlib.pyplot as plt
import numpy as np
import os

import slow_sph


GAMMA = 1.4

def neighbour_search(ii, data, search_radius):
    """
    Find the neighbours of particle ii within a given search radius
    """
    r_i = data['pos'][ii]
    dists = np.abs(r_i - data['pos'])

    return np.where(dists < search_radius)


def calc_smoothing_length(data, eta):
    """
    Calculate the smoothing length.

    This is the average of the minimum distance between each particle
    and its nearest neighbour, scaled by eta.
    """
    min_dists = []
    for ix, part in enumerate(data['pos'][:-1]):
        min_dists.append(np.min(np.abs(part-data['pos'][ix+1:])))
    return eta * np.mean(min_dists)

def kernel(r_ij, h):
    """
    A 1D smoothing kernel, that extends to 2*h.
    Integral from [-2h,2h] equals 1.
    """
    s = np.linalg.norm(r_ij)/h

    # For 1D only!
    N = 2./(3*h)

    if 0 <= s <= 1:
        return N * (1 - 3/2*s**2 + 3/4*s**3)
    elif 1 < s <= 2:
        return N * (1./4 * (2-s)**3)
    if s > 2:
        return 0


def grad_kernel(r_ij, h):
    """
    The gradient of the 1D smoothing kernel.

    The result is returned in the direction of the unit vector
    of r_ij
    """
    s = np.linalg.norm(r_ij)/h

    if np.isclose(np.linalg.norm(r_ij), 0):
        r_hat = 0. * r_ij
    else:
        r_hat = r_ij / np.linalg.norm(r_ij)

    # For 1D only!
    N = 2./(3*h**2)

    if 0 <= s <= 1:
        return N * (-3*s + 9./4*s**2) * r_hat
    elif 1 < s <= 2:
        return N * (-3./4*(2-s)**2) * r_hat
    if s > 2:
        return 0 * r_hat


def calc_density(data, h):
    """
    Calculate the density
    """
    all_dens = []
    for ii, pos in enumerate(data['pos']):
        dens = 0.
        for jj in data['neigh'][ii][0]:
            r_ij = pos - data['pos'][jj]
            dens += data['mass'][jj] * kernel(r_ij, h)
        all_dens.append(dens)
    data['dens'] = np.array(all_dens)


def calc_eos(data, isothermal=False):
    """
    Calculate the equation of state, relating density and internal
    energy to pressure.

    We also calculate the soundspeeds of each particle
    """
    data['pres'] = (GAMMA - 1.) * data['dens'] * data['eint']

    if isothermal:
        raise UserWarning("Not Implemented")
    else:
        data['cs'] = np.sqrt(GAMMA * (GAMMA-1) * data['eint'])
    return


def calc_dt(data, h, CFL=0.5):
    """
    Calculate the maximum safe timestep
    """
    left_dt = 1.e99
    right_dt = 1.e99

    for ii in range(len(data['pos'])):
        div_vel = calc_div_vel(ii, data, h)

        dt = CFL * h / (h * np.abs(div_vel) + data['cs'][ii])
        if dt < left_dt:
            left_dt = dt

        eps = 1e-10
        dt = CFL * np.sqrt(h / ((np.abs(data['dvdt'][ii])) + eps))
        if dt < right_dt:
            right_dt = dt

    min_dt = np.min((left_dt, right_dt))

    return min_dt


def calc_div_vel(ii, data, h):
    """
    Calculate the divergence of the velocity at the position of particle ii
    """
    div_vel = 0.
    r_i = data['pos'][ii]

    for jj in data['neigh'][ii][0]:
        r_j = data['pos'][jj]
        r_ij = r_i - r_j

        div_vel += data['mass'][jj]/data['dens'][jj] \
                   * np.dot(data['vel'][jj], grad_kernel(r_ij, h))

    return div_vel


def calc_forces(data, h):
    """
    Calculate the change in momentum and change in internal energy.

    Since equations utilise similar terms they are evaluated simultaneously
    for efficiency.
    """

    for ii in range(len(data['pos'])):
        dvdt = 0.
        dudt = 0.

        for jj in data['neigh'][ii][0]:
            if ii != jj:
                # Get vector difference in position and velocity
                r_ij = data['pos'][ii] - data['pos'][jj]
                v_ij = data['vel'][ii] - data['vel'][jj]

                # Calculate unit vector in direction r_ij
                r_hat = r_ij / np.linalg.norm(r_ij)


                Press_term = (data['pres'][ii] / (data['dens'][ii]**2) \
                             + data['pres'][jj] / (data['dens'][jj]**2) )

                # Calculate contribution from pressure
                dvdt += -data['mass'][jj]*Press_term*grad_kernel(r_ij,h)
                dudt += data['pres'][ii]*data['mass'][jj]*np.dot(v_ij,grad_kernel(r_ij,h))/(data['dens'][ii]**2)

        data['dvdt'][ii] = dvdt
        data['dudt'][ii] = dudt
    return


def evolve_particles(data, dt):
    """
    Advance particles forward by one timestep
    """
    data['pos'] += data['vel'] * dt
    data['vel'] += data['dvdt'] * dt
    data['eint'] += data['dudt'] * dt
    return

def print_energy(data, e_ref, ctime):
    """
    Print energy values and fractional error
    """
    e_kin = float(np.sum(0.5*data['vel']**2))
    e_int = float(np.sum(data['eint']))
    e_tot = e_kin + e_int
    e_err = 100*(e_tot - e_ref)/e_ref

    print('%.5f | e_kin: %.5f, e_int: %.5f, e_tot: %.5f, err: %.2f%%'%(
        ctime, e_kin, e_int, e_tot, e_err))
    return


def write_checkpoint(data, chk_cnt, output_dir, h):
    """
    Wrtie a checkpoint file in a given directory.
    The output_dir will be created as needed.
    """
    filename = output_dir + '%04i.dat'%chk_cnt

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filename, 'w') as fp:
        header = '%5s%8s%8s%8s%8s%8s%8s%5s\n' % (
            'ix', 'pos', 'vel', 'mass', 'h', 'dens', 'eint', 'nnei')
        fp.write(header)

        for i in range(len(data['pos'])):
            dataline = '%5i%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%5i\n' % (
                i, data['pos'][i], data['vel'][i],
                data['mass'][i], h, data['dens'][i], data['eint'][i],
                len(data['neigh'][i][0]),
            )
            fp.write(dataline)

def generate_plots(data, output_dir, ctime, plt_cnt):
    quants = ['dens', 'vel', 'pres', 'eint']
    for quant in quants:
        plt.clf()
        plt.plot(data['pos'], data[quant], 'b+')
        # plt.xlim(-1, 1)
        plt.xlabel('position')
        plt.ylabel(quant)
#         if quant != 'eint':
#             plt.ylim(-0.1, 1.1)
        plt.title('t = %6.4f' % ctime)
        plt.savefig(output_dir + '%s_%02i.png' % (quant, plt_cnt))


def run_simulation(data, eta, CFL, tmax, plt_interv, output_dir,
                   setup_only=False):
    """
    Run entire simulation

    Parameters
    ----------
    data: dict
        dictionary with arrays for each quantity:
        pos, mass, vel, dvdt, dudt, neigh
    """
    if not os.path.exists(output_dir):
        print('Making directory: %s'%output_dir)
        os.makedirs(output_dir)

    ctime = 0.

    plt_cnt = 0
    plt_time = 0.

    chk_cnt = 0

    nparts = len(data['pos'])
    e_ref = np.sum(0.5*data['vel']**2) + np.sum(data['eint'])

    while ctime < tmax:

        # Find h
        h = calc_smoothing_length(data, eta)

        # Find neighbours
        for ii in range(nparts):
            data['neigh'][ii] = neighbour_search(ii, data, 2*h)

        # Calculate values
        calc_density(data, h)
        calc_eos(data)
        dt = calc_dt(data, h, CFL)

        # Adjust timestep down in order to precisely hit plotting and end times
        if ctime+dt > tmax:
            dt = tmax - ctime
        if ctime+dt > plt_time:
            dt = plt_time - ctime

        # Do output, the first 10, then every 10 thereafter
        if chk_cnt < 10 or chk_cnt%10 == 0:
            write_checkpoint(data, chk_cnt, output_dir, h)
        chk_cnt += 1

        # Generate plots
        if ctime >= plt_time:
            generate_plots(data, output_dir, ctime, plt_cnt)
            plt_cnt += 1
            plt_time += plt_interv
            if setup_only:
                print('Generated only setup plots. Terminating now.')
                break

        # Calculate forces
        calc_forces(data, h)

        # Generate some output
        print_energy(data, e_ref, ctime)
        if False:
            print('Momentum: ', np.sum(data['vel']))
            print('pos:  ', data['pos'])
            print('vel:  ', data['vel'])
            print('dvdt: ', data['dvdt'])

        # Step forward one time step
        evolve_particles(data, dt)

        ctime += dt

    # Generate final state output
    print_energy(data, e_ref, ctime)
    generate_plots(data, output_dir, ctime, plt_cnt)


