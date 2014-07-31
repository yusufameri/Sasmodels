#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from weights import GaussianDispersion
from sasmodel import card
from time import time

def set_precision(src, qx, qy, dtype):
    qx = np.ascontiguousarray(qx, dtype=dtype)
    qy = np.ascontiguousarray(qy, dtype=dtype)
    if np.dtype(dtype) == np.dtype('float32'):
        header = """\
#define real float
"""
    else:
        header = """\
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define real double
"""
    return header+src, qx, qy

class GpuCylinder(object):
    PARS = {
        'scale':1,'radius':1,'length':1,'sldCyl':1e-6,'sldSolv':0,'background':0,
        'cyl_theta':0,'cyl_phi':0,
    }
    PD_PARS = ['radius', 'length', 'cyl_theta', 'cyl_phi']

    def __init__(self, qx, qy, dtype='float32'):

        #create context, queue, and build program
        ctx,_queue = card()
        src, qx, qy = set_precision(open('NR_BessJ1.cpp').read()+"\n"+open('Kernel-Cylinder.cpp').read(), qx, qy, dtype=dtype)
        self.alignment = ctx.devices[0].min_data_type_align_size/np.dtype(dtype).itemsize
        #self.workgroup = ctx.devices[0].max_work_group_size
        self.workgroup_size = 64
        print "workgroup size",self.workgroup_size,dtype
        max_nloc = "#define MAX_NLOC %d\n"%self.workgroup_size
        src = max_nloc + src

        self.prg = cl.Program(ctx, src).build()
        self.qx, self.qy = qx, qy

        #buffers
        mf = cl.mem_flags
        self.qx_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.qx)
        self.qy_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.qy)
        self.res_b = cl.Buffer(ctx, mf.WRITE_ONLY, qx.nbytes)
        self.res = np.empty_like(self.qx)

    def eval(self, pars):

        ctx,queue = card()
        radius, length, cyl_theta, cyl_phi = \
            [GaussianDispersion(int(pars[base+'_pd_n']), pars[base+'_pd'], pars[base+'_pd_nsigma'])
             for base in GpuCylinder.PD_PARS]

        #Get the weights for each
        radius.value, radius.weight = radius.get_weights(pars['radius'], 0, 10000, True)
        length.value, length.weight = length.get_weights(pars['length'], 0, 10000, True)
        cyl_theta.value, cyl_theta.weight = cyl_theta.get_weights(pars['cyl_theta'], -90, 180, False)
        cyl_phi.value, cyl_phi.weight = cyl_phi.get_weights(pars['cyl_phi'], -90, 180, False)

        #Perform the computation, with all weight points
        sum, norm, norm_vol, vol = 0.0, 0.0, 0.0, 0.0
        size = len(cyl_theta.weight)
        sub = pars['sldCyl'] - pars['sldSolv']

        start_time = time()
        kernel_runs_count = 0

        N = np.prod([len(V) for V in radius.weight, length.weight, cyl_theta.weight, cyl_phi.weight])
        nloc=self.alignment
        if N%nloc != 0:
            N+= nloc - N%nloc

        w = np.zeros(N, dtype="float")

        uRadius, uLength, uCyl_Theta, uCyl_Phi = [np.empty(N, dtype="float") for _ in range(4)]

        print "N is the product of the length of each of the following\nIn other words radius.weight*length.weight*cyl_theta.weight*cyl_phi.weight= ",N

        j = 0
        for a,wa in zip(radius.value, radius.weight):
            for b,wb in zip(length.value, length.weight):
                for c,wc in zip(cyl_theta.value, cyl_theta.weight):
                    for d,wd in zip(cyl_phi.value, cyl_phi.weight):
                        uRadius[j] = a
                        uLength[j] = b
                        uCyl_Theta[j] = c
                        uCyl_Phi[j] = d
                        w[j] = wa*wb*wc*wd
                        vol += wa*wb*pow(a, 2)*b
                        norm_vol += wa*wb
                        norm += w[j]
                        j+=1

        ZIP = np.hstack([w, uRadius, uLength, uCyl_Theta, uCyl_Phi])
        d_zip = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ZIP)

        real = np.float32 if self.qx.dtype == np.dtype('float32') else np.float64

        self.prg.CylinderKernel(queue, self.qx.shape, [self.workgroup_size],
                                self.qx_b, self.qy_b, self.res_b,
                                #np.uint32(self.qx.shape[0]),
                                d_zip, real(sub),
                                real(pars['scale']), np.uint32(size), np.uint32(N))
       


        queue.finish()
        cl.enqueue_copy(queue, self.res, self.res_b)
        kernel_runs_count +=1

        run_time = time() - start_time
        print run_time, "seconds it TOOK!!!!!!!!!!!! and the kernel ran", kernel_runs_count,"times!"

        if vol != 0.0 and norm_vol != 0.0:
            self.res *= norm_vol/vol

        return self.res/norm+pars['background']
