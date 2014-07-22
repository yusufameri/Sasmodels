#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from random import randrange
from time import time

count = 50000 #total times done

#sets up 500 random floats
input = np.random.rand(50000).astype(np.float32)

for platform in cl.get_platforms():
    for device in platform.get_devices():

        ti=time()

        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags
        #buffers for input/result
        in_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input)
        result_b = cl.Buffer(ctx, mf.WRITE_ONLY, input.nbytes)

        prg = cl.Program(ctx, """
        __kernel void Square(__global const float *in_g, __global float *result_g, const int count){
            int gid = get_global_id(0);
            if(gid < count)
                result_g[gid] = in_g[gid]*in_g[gid];
        }
        """).build()

        #.shape means same size as input data
        prg.Square(queue, input.shape, None, in_b, result_b, np.uint32(count))

        #creates an output result, and fills with values calculated
        result = np.empty_like(input)
        cl.enqueue_copy(queue, result, result_b)

        ti2=time()

        #print the results
        correct = 0
        for x in xrange(50000):
            if input[x]*input[x] == result[x]:
                correct+=1

        print correct
        print("Time taken: ", ti2-ti)
