#!/usr/bin/env python

# This script shows how to run a model on an MPI cluster. It runs this
# example.
#
# To run this on an MPI cluster, use something like:
#
#     mpiexec -n 4 python mpi_example.py 10000

import os
import re
import sys
import time

from pyReef.model import Model

base_path = '.'
xml_name = 'input.xml'

run_years = float(sys.argv[1])

# change into current data directory
os.chdir(base_path)

start_time = time.time()
reef = Model()

#print 'loading %s' % xml_name

reef.load_xml(xml_name)

reef.run_to_time(run_years)
#print 'run to %s years finished in %s seconds' % (run_years, time.time() - start_time)
