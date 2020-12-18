# coding: utf-8
# iPython magic functions

# Force iPython to reimport files when edited
%load_ext autoreload
%autoreload 2                

%pdb                        # Jump into a debugger whenever an error occurs               
%run my_python_file.py      # Run a file

%timeit 5*3                 # Get execution time of expression

import my_python_file

my_python_file.__dir__()    # show all functions/attributes of this module

my_python_file.some_function? # use '?' to print docstring

from scipy import integrate
integrate.quad(t7.smoothing_kernel, -np.inf, np.inf, (1))

import ipdb; ipdb.set_trace()       # Insert this line into code to trigger debugger

