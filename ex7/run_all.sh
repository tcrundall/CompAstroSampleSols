#!/usr/bin/env bash

# First explore eta dependence
# python ex7.py eta CFL
python ex7.py 3 0.5
python ex7.py 5 0.5
python ex7.py 10 0.5

# Now explore CFl dependence
python ex7.py 3 0.1
python ex7.py 3 0.9
