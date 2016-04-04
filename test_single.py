
import sys
import unittest

from toast.mpirunner import MPITestRunner

file = sys.argv[1]

loader = unittest.TestLoader()
runner = MPITestRunner(verbosity=2)
suite = loader.discover('tests', pattern='{}'.format(file), top_level_dir='.')
runner.run(suite)
