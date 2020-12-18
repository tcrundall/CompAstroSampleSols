import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy import integrate

# print('imported!')
print('reimported!')

def calc_smoothing_length_slow(particles, eta):
    """
    Calculate a global SPH smoothing length as defined in tut 6, eq. 10

    Calculate a smoothing length based on the averge distance between
    each particle and their closest neighbour.

    Parameters
    ----------
    particles: [n, ndim] float array
        An array of n particles with positions in `ndim` dimensions
        Units are irrelevant.

    eta: float
        A scaling parameter that should be larger than 1.
        Suggested values are between 2 and 10.

    Returns
    -------
    smoothing_length: float
        The global SPH smoothign length, with same units as input
    """
    min_dists = []
    for i, part1 in enumerate(particles[:-1]):
        all_dists = []
        for part2 in particles[i+1:]:
            dist = np.sqrt(np.sum(np.square(part1 - part2)))
            all_dists.append(dist)

        min_dists.append(np.min(all_dists))

    mean_min_dists = np.mean(min_dists)

    smoothing_length = eta * mean_min_dists
        
    return smoothing_length


def calc_smoothing_length(particles, eta):
    """
    Calculate a global SPH smoothing length as defined in tut 6, eq. 10

    Parameters
    ----------
    particles: [n, ndim] float array
        An array of n particles with positions in `ndim` dimensions
        Units are irrelevant.

    eta: float
        A scaling parameter that should be larger than 1.
        Suggested values are between 2 and 10.

    Returns
    -------
    smoothing_length: float
        The global SPH smoothign length, with same units as input
    """
    min_dists = []
    for i, part1 in enumerate(particles[:-1]):
        all_dists = []
        dists = np.sqrt(np.sum(np.square(particles[i+1:] - part1), axis=1))
        min_dists.append(np.min(dists))

    mean_min_dists = np.mean(min_dists)

    smoothing_length = eta * mean_min_dists
        
    return smoothing_length

def get_neighbours(parts, target_pos, search_radius):
    """
    Find all particles in `parts` that are within `search_radius` of
    `target_pos`

    A particle is a neighbour of itself, so if get_neighbours is called 
    on the position of a particle, this particle will also be included.

    Parameters
    ----------
    parts: [nparts, ndim] float array
        An array of positions of all particles

    target_pos: [ndim] float array
        The position at which the neighbour search is centred on

    search_radius: float
        The radius to search out to. In SPH this is typically 
        2*smoothing_length.

    Returns
    -------
    neighouring_parts: [nneighbours, ndim] float array
    """
    dists = np.sqrt(np.sum(np.square(parts - target_pos), axis=1))

    #!! This actually isn't needed, as the smoothing kernel function is defined
    # # Comparing float equality is problematic due to finite machine precision
    # # So we amplify the search radius by 1e-10.
    # # such that density contributions from dist = 2 are exactly 0.
    # search_radius *= (1+1e-10)

    neighbours = parts[np.where(dists <= search_radius)]
    assert False

    return neighbours
    
def smoothing_kernel(dist, h, ndim=1):
    """
    Calculates the value of a smoothing kernel `dist` units away from
    origin.

    This function is a PDF and hence should have a total area under
    the curve of 1.

    Parameters
    ----------
    dist: float
        Euclidean distance (between two particles)
    h: float
        Smoothing length
    ndim: integer [1, 2, 3]
        Number of dimensions

    Returns
    -------
    res: float
        The evaluation of the kernel at `dist/h` from centre
    """
    norm_const_dict = {
        1: 2./(3*h),
        2: 10./(7*np.pi*h**2),
        3: 1./(np.pi*h**3),
    }

    N = norm_const_dict[ndim]
    dist = abs(dist)
    s = dist/h

    if s < 0:
        raise UserWarning('Distances should be positive')
    elif 0 <= s < 1:
        return N * (1 - 3./2.*s**2 + 3./4.*s**3)
    elif 1 <= s < 2:
        return N * (1./4.*(2-s)**3)
    elif s >= 2:
        return N * 0.

    raise UserWarning('Invalid distance provided?')

def dens_estimator(pos, parts, masses, slength):
    """
    Estimate the density at a given position using SPH particles

    Algorithm:
    - find all the particles within 2*slength from pos
    - Calculate their density contributions via the smoothing kernel
    - Sum densities together

    Parameters
    ----------
    pos: [ndim] float
        position at which we want the density

    parts: [nparts, ndim] float
        Position of all SPH aprticles

    masses: [nparts] float array -or- float
        An array of masses for each SPH particle
            -or-
        A single float, in the case where all particles have identical mass

    slength: float
        The smoothing length.
        Due to how expensive it is to calcuate, this should be pre-calculated.

    Returns
    -------
    res: float
        The estimated density at position `pos`
    """
    # Get neighbours
    neighbours = get_neighbours(parts, pos, 2*slength)

    # Calculate densities
    all_dens = []
    for part in neighbours:
        dist = np.sqrt(np.sum(np.square(part - pos)))
        all_dens.append(smoothing_kernel(dist, slength, ndim=1))

    # Sum up densitites
    return np.sum(all_dens)
    

# def generate_random_particles(nparticles, span, left_boundary=None, ndim=1):
#     """
#     Generate an array of particles with a uniform random distribution.
#     """
#     if left_boundary is None:
#         left_boundary = np.zeros(3)
#     assert len(left_bound) == ndim
# 
#     pos = np.random.rand(nparticles, ndim)      # generate values from 0 to 1
#     pos *= pos_span             # Amplify span (0,.1) to desired range
#     pos += left_bound           # Offset, taking left boundary from 0, to left_bound
# 
#     return pos


def generate_row_of_particles(nparticles, xmin, xmax, ndim=1):
    """
    Generate a uniform row of particles such that each particle
    has an identical separation distance to their nearest neighbour.

    Note, this includes particles on the boundary.

    Parameters
    ----------
    nparticles: int
        The number of particles to be generated

    xmin: float
        Lower bound
    xmax: float
        Upper bound
       
    Returns
    -------
    pos: [nparticles, ndim] float array
        The positions of nparticles distributed evenly along xaxis (y=z=0)
    """
    # Have y=z=0 for all particles in the event that ndim > 1
    pos = np.zeros((nparticles, ndim))

    # But set x based on separation
    pos[:,0] = np.linspace(xmin,xmax,nparticles)
    return pos


def test_functions():
    # --------------------------------------------------
    # --------------------------------------------------
    # ---  TESTING      --------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    # Setting up parameters
    nparticles = 11
    ndim = 1
    av_mass = 1.
    xmin = 0.
    xmax = 1.
    eta = 2.

    # --------------------------------------------------
    # ---  EXERCISE 1.1 --------------------------------
    # --------------------------------------------------
    # Determing smoothing length
    parts = generate_row_of_particles(nparticles, xmin=xmin, xmax=xmax, ndim=ndim)
    span = xmax - xmin
    slength_true = eta * (parts[1] - parts[0])

    ms = np.ones(nparticles) * av_mass
    slength = calc_smoothing_length(parts, eta=eta)
    assert np.isclose(slength_true, slength)

    
    # --------------------------------------------------
    # ---  EXERCISE 1.2 --------------------------------
    # --------------------------------------------------
    # Performing a neighbour search
    target_part = parts[0]
    neighbours = get_neighbours(parts, target_part, 2*slength)

    assert len(neighbours) == 4

    # --------------------------------------------------
    # ---  EXERCISE 1.3 --------------------------------
    # --------------------------------------------------
    # Build a density estimator

    # First check smoothing kernel
    pts = np.linspace(-3,3,601)
    kern_val = [smoothing_kernel(abs(pt), h=1, ndim=1) for pt in pts]
    plt.clf()
    plt.plot(pts, kern_val)
    plt.savefig('M4_kernel.png')

    # Confirm smoothing kernel has area = 1
    area = integrate.quad(smoothing_kernel, -np.inf, np.inf, args=(slength))[0]
    assert np.isclose(1,area, rtol=1e-8)
    
    
    # Now check density
    pts = np.linspace(0,span,101)
    dens_vals = [dens_estimator(pt, parts, ms, slength) for pt in pts]
    plt.clf()
    plt.plot(pts, dens_vals)
    plt.savefig('dens.png')

    assert np.isclose(nparticles, np.max(dens_vals))

# if __name__ == '__main__':
#     my_parts = generate_row_of_particles(11, 0, 10, ndim=1)
#     search_rad = 2.
# 
#     third_part = my_parts[3]
#     print(my_parts)
#     print(third_part)
# 
#     neighbours = get_neighbours(my_parts, third_part, search_rad)
#     print(neighbours)
#     assert len(neighbours) == 5
#     
    
    
if __name__ == '__main__': 
    # --------------------------------------------------
    # --------------------------------------------------
    # ---  ASSIGNMENT   --------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    xmin = 0.
    xmax = 1.
    mass = 1.

    eta_set = [2,5,10]
    nparticles_set = [11,101,1001]

    fig, axes = plt.subplots(1,3)
    fig.set_size_inches(15,5)

    for i, nparticles in enumerate(nparticles_set):
        # Initialise particle positions
        parts = generate_row_of_particles(nparticles, xmin, xmax, ndim=1)

        for eta in eta_set:
            # Calculate smoothing length
            slength = calc_smoothing_length(parts, eta)

            # Calculate density at each particle
            densities = []
            for part in parts:
                densities.append(dens_estimator(part, parts, mass, slength))

            axes[i].plot(parts[:,0], densities, label='eta = %2i'%eta)
            axes[i].set_title('Nparts: %5i, max density: %5.2f'%(nparticles,
                                                            np.max(densities)))
            

        axes[i].set_xlabel('Unit length')
        axes[i].set_ylabel('Unit density')

        axes[-1].legend(loc='best')
    fig.savefig('densities.png')

    plt.close(fig)
    print('end of main')

print('chang')
print('end of file')
