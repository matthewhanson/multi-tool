#!/usr/bin/env python
################################################################################
#    multi-tool
#
#    Utilities for multiprocessing on 3d arrays such multi-band imagery
#
#    AUTHOR: Matthew Hanson
#    EMAIL:  matt.a.hanson@gmail.com
#
#    Copyright (C) 2015 Matthew Hanson
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
################################################################################

import SharedArray as sa
import numpy as np
import multiprocessing
import traceback
import uuid


class MultiTool(object):
    """ Class for processing arrays (on disk or in memory) using Python multiprocessing """

    def __init__(self, pfunc, nbandsout=1, nchunks=100, nproc=2, dropnodata=False):
        """ Create multiprocessing pool
            pfunc - processing function
            outbands - # of output bands, output will be (outbands x Rows x Columns)
            nchunks - # of chunks to use
            nproc - the # of processors to use
            dropnodata - if set to True drop nodata values will be dropped
        """
        self.pfunc = pfunc
        self.nbandsout = nbandsout
        self.nchunks = nchunks
        self.nproc = nproc
        self.dropnodata = dropnodata
        self.wfunc = None

    @staticmethod
    def cpu_count():
        """ Get count of CPUs on system """
        return multiprocessing.cpu_count()

    def process_array(self, arrin):
        """ Process an in memory array with pfunc """
        self._setup_arrays(arrin.shape)

        self.rfunc = lambda chunk: arrin[:, chunk[1]:chunk[1] + chunk[3], chunk[0]:chunk[0] + chunk[2]]

        if self.nproc == 1:
            self._process_nomulti()
        else:
            self._process()
        self._cleanup()
        return self.arrout

    def read_and_process(self, inshape, rfunc):
        """ Provide a read function in place of an array
            inshape - shape of the input array (NumInBands x Rows x Columns)
            rfunc - function used to read a chunk of data
        """
        self._setup_arrays(inshape)

        if self.nproc == 1:
            self._process_nomulti()
        else:
            self._process()
        self._cleanup()
        return self.arrout

    def _setup_arrays(self, inshape):
        """ Setup instance variabels and arrays for processing """
        self.inshape = inshape

        # use shared array
        if self.wfunc is None:
            self.id = str(uuid.uuid4())
            self.arrout = sa.create(self.id, self.inshape)
            self.arrout[:] = np.nan
            self.chunks = self.chunk(self.inshape)

            # TODO - allow custom write function
            #self.arrout = np.empty((self.nbandsout, self.inshape[1], self.inshape[2]))

            def wfunc(output, chunk):
                self.arrout[:, chunk[1]:chunk[1] + chunk[3], chunk[0]:chunk[0] + chunk[2]] = output

            self.wfunc = wfunc

    def _cleanup(self):
        if self.wfunc is None:
            sa.delete(self.id)

    def _process_nomulti(self):
        """ Run processing function on chunks sequentially (no multiprocessing) """
        self._setup_globals(self.nbandsout, self.rfunc, self.pfunc, self.wfunc, self.dropnodata)
        # sequentially process chunks
        for ch in self.chunks:
            mt_worker(ch)
        return self.arrout

    def _process(self):
        """ Create multiprocessing pool and process """
        initargs = (self.nbandsout, self.rfunc, self.pfunc, self.wfunc, self.dropnodata)
        pool = multiprocessing.Pool(self.nproc, initializer=self._setup_globals, initargs=initargs)
        pool.map(mt_worker, self.chunks)
        return self.arrout

    @staticmethod
    def chunk(shape, nchunks=100):
        """ Divide input array into chunks """
        chunksz = int(shape[1] / nchunks)
        remainder = shape[1] - chunksz * nchunks
        chszs = [chunksz] * (nchunks - remainder) + [chunksz + 1] * remainder
        chunks = []
        for ichunk in range(nchunks):
            chunks.append([0, sum(chszs[:ichunk]), shape[2], chszs[ichunk]])
        return chunks

    @staticmethod
    def _setup_globals(_outbands, _rfunc, _pfunc, _wfunc, _dropnodata):
        """ Initializer sets globals for processes """
        global MT_OUTBANDS, MT_RFUNC, MT_PFUNC, MT_WFUNC, MT_DROPNODATA
        MT_OUTBANDS = _outbands
        MT_RFUNC = _rfunc
        MT_PFUNC = _pfunc
        MT_WFUNC = _wfunc
        MT_DROPNODATA = _dropnodata


def mt_worker(chunk):
    """ Worker function that operates on single chunk (has access to global variables set in _setup """

    # read chunk of data and make sure it is 3-D: BxYxX
    data = MT_RFUNC(chunk)
    shape = data.shape
    if len(shape) == 2:
        data = data.reshape((1, shape[0], shape[1]))
        shape = data.shape

    # make output array for this chunk
    output = np.empty((MT_OUTBANDS, shape[1], shape[2]))
    output[:] = np.nan

    # only run on valid pixel signatures unless nodata set
    if MT_DROPNODATA:
        valid = np.all(~np.isnan(data), axis=0)
    else:
        valid = np.ones((shape[1], shape[2])).astype('bool')

    # run processing function
    try:
        output[:, valid] = MT_PFUNC(data[:, valid])
    except Exception, e:
        print "Error processing chunk %s: %s" % (' '.join(chunk), e)
        print traceback.format_exc()
        from nose.tools import set_trace
        set_trace()

    # write using write function
    MT_WFUNC(output, chunk)
