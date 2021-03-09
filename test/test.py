import unittest
from subprocess import run
import h5py

class TestMatrixSerial(unittest.TestCase):
    def test_serial_rectangle(self):
        run(['random_matrix', '10', '40'])
        with h5py.File('matrix.hdf5', 'r') as fp:
            self.assertEqual(10, fp['matrix'][:, :].shape[0])
            self.assertEqual(40, fp['matrix'][:, :].shape[1])

    def test_square(self):
        run(['random_matrix', '40'])
        with h5py.File('matrix.hdf5', 'r') as fp:
            self.assertEqual(40, fp['matrix'][:, :].shape[0])
            self.assertEqual(40, fp['matrix'][:, :].shape[1])

class TestMatrixParallel(unittest.TestCase):
    def test_rectangle(self):
        run(['mpirun', '-np', '4', 'random_matrix', '10', '40'])
        with h5py.File('matrix.hdf5', 'r') as fp:
            self.assertEqual(10, fp['matrix'][:, :].shape[0])
            self.assertEqual(40, fp['matrix'][:, :].shape[1])

    def test_square(self):
        run(['mpirun', '-np', '4', 'random_matrix', '40', '40'])
        with h5py.File('matrix.hdf5', 'r') as fp:
            self.assertEqual(40, fp['matrix'][:, :].shape[0])
            self.assertEqual(40, fp['matrix'][:, :].shape[1])
