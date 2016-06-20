#!/usr/bin/env python

import unittest
import numpy
import numpy.testing as nt
from multitool import MultiTool


class MultiProcessTest(unittest.TestCase):
    """ Test MultiTool class """

    nchunks = 100
    small_shape = (10, 100, 100)
    large_shape = (10, 4000, 4000)

    @classmethod
    def _test_passthrough(cls, arrayshape, nproc=2):
        """ Test passthrough function """
        arrin = numpy.zeros(arrayshape)
        mp = MultiTool(lambda x: x, nbandsout=arrin.shape[0], nchunks=cls.nchunks, nproc=nproc)
        arrout = mp.process_array(arrin)
        nt.assert_array_equal(arrin, arrout)        

    @classmethod
    def _test_process(cls, arrayshape, nproc=2):
        arrin = numpy.zeros(arrayshape)
        # test adding one
        mp = MultiTool(lambda x: numpy.sqrt(x) + 1, nbandsout=arrin.shape[0], nchunks=cls.nchunks, nproc=nproc)
        arrout = mp.process_array(arrin)
        nt.assert_array_equal(numpy.sqrt(arrin) + 1, arrout)

    def test1_simple_process_nomulti(self):
        """ Test simple processes work without multiprocessing """
        self._test_passthrough(self.small_shape, nproc=1)
        self._test_process(self.small_shape, nproc=1)

    def test2_simple_process(self):
        """ Test simple processes work with multiprocessing """
        #self._test_passthrough(self.small_shape)
        self._test_process(self.small_shape)

    def test3_process_nomulti(self):
        """ Test process on large array without multiprocessing """
        self._test_process(self.large_shape, nproc=1)

    def test4_process(self):
        """ Test process on large array with multiprocessing """
        self._test_process(self.large_shape)
