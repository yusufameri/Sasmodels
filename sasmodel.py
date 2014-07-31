#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from bumps.names import Parameter
from sans.dataloader.loader import Loader
from sans.dataloader.manipulations import Ringcut


def load_data(filename):
    loader = Loader()
    data = loader.load(filename)
    if data is None:
        raise IOError("Data %r could not be loaded"%filename)
    return data


def set_beam_stop(data, radius):
    data.mask = Ringcut(0, radius)(data)


def plot_data(data, iq):
    from numpy.ma import masked_array
    import matplotlib.pyplot as plt
    img = masked_array(iq, data.mask)
    xmin, xmax = min(data.qx_data), max(data.qx_data)
    ymin, ymax = min(data.qy_data), max(data.qy_data)
    plt.imshow(img.reshape(128,128),
               interpolation='nearest', aspect=1, origin='upper',
               extent=[xmin, xmax, ymin, ymax])


def plot_result(data, theory):
    import matplotlib.pyplot as plt
    plt.subplot(1,3,1)
    plot_data(data, data.data)
    plt.subplot(1,3,2)
    plot_data(data, theory)
    plt.subplot(1,3,3)
    plot_data(data, (theory-data.data)/data.err_data)
    plt.colorbar()


def demo():
    data = load_data('JUN03289.DAT')
    set_beam_stop(data, 0.004)
    plot_data(data)
    import matplotlib.pyplot as plt; plt.show()


GPU_CONTEXT = None
GPU_QUEUE = None
def card():
    global GPU_CONTEXT, GPU_QUEUE
    if GPU_CONTEXT is None:
        GPU_CONTEXT = cl.create_some_context()
        GPU_QUEUE = cl.CommandQueue(GPU_CONTEXT)
    return GPU_CONTEXT, GPU_QUEUE


class SasModel(object):
    def __init__(self, data, model, dtype='float32', **kw):
        self.__dict__['_parameters'] = {}
        self.index = data.mask==0
        self.iq = data.data[self.index]
        self.diq = data.err_data[self.index]
        self.data = data
        self.qx = data.qx_data
        self.qy = data.qy_data
        self.gpu = model(self.qx, self.qy, dtype=dtype)
        pd_pars = set(base+attr for base in model.PD_PARS for attr in ('_pd','_pd_n','_pd_nsigma'))
        total_pars = set(model.PARS.keys()) | pd_pars
        extra_pars = set(kw.keys()) - total_pars
        if extra_pars:
            raise TypeError("unexpected parameters %s"%(str(extra_pars,)))
        pars = model.PARS.copy()
        pars.update((base+'_pd', 0) for base in model.PD_PARS)
        pars.update((base+'_pd_n', 35) for base in model.PD_PARS)
        pars.update((base+'_pd_nsigma', 3) for base in model.PD_PARS)
        pars.update(kw)
        self._parameters = dict((k, Parameter.default(v, name=k)) for k, v in pars.items())
        self._cache = {}

    def update(self):
        self._cache = {}

    def numpoints(self):
        return len(self.iq)

    def parameters(self):
        return self._parameters

    def __getattr__(self, par):
        try:
            return self._parameters[par]
        except:
            raise AttributeError("%r is not a parameter or attribute"%par) 

    def __setattr__(self, par, val):
        if par in self._parameters:
            self._parameters[par] = val
        else:
            self.__dict__[par] = val

    def theory(self):
        if 'theory' not in self._cache:
            pars = dict((k,v.value) for k,v in self._parameters.items())
            self._cache['theory'] = self.gpu.eval(pars)
        return self._cache['theory']

    def residuals(self):
        #if np.any(self.err ==0): print "zeros in err"
        return (self.theory()[self.index]-self.iq)/self.diq

    def nllf(self):
        R = self.residuals()
        #if np.any(np.isnan(R)): print "NaN in residuals"
        return 0.5*np.sum(R**2)

    def __call__(self):
        return 2*self.nllf()/self.dof

    def plot(self, view='linear'):
        plot_result(self.data, self.theory())

    def save(self, basename):
        pass

    def update(self):
        pass
