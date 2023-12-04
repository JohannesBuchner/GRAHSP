# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN powerlaw module
==================================================

This module uses a simple powerlaw for the BBB,
with a turn-over.

"""
from collections import OrderedDict
import numpy as np
from numpy import exp, log
from . import CreationModule

from numba import jit

@jit(nopython=True)
def sbpl_jitted(result, x, norm, lam1, lam2, x0, xbrk, Lambda):
    """Smoothly bending powerlaw

    Parameterization from Ryde98

    Parameters
    ----------
    result: array for storing the result, assumed dense
    x: array of independent variable
    xmax: only consider x values up to this value
    norm: normalisation at x0
    lam1: powerlaw slope below xbrk
    lam2: powerlaw slope above xbrk
    xbrk: x value where powerlaw break occurs
    Lambda: width of bend at xbrk
    """
    expo = 1. / Lambda
    lamaddexpo = (lam1 + lam2 + 2) / 2.
    lamsubexpo = (lam2 - lam1) / 2. * Lambda
    xpivratio = x0 / xbrk
    divisor = 1.0 / (xpivratio**expo + xpivratio**-expo)
    for i in range(len(x)):
        xratio = x[i] / xbrk
        result[i] = norm * (x[i] / x0)**lamaddexpo * \
            ((xratio**expo + xratio**-expo) * divisor)**lamsubexpo  * (x0 / x[i])


def sbpl(x, norm, lam1, lam2, x0, xbrk, Lambda):
    """Smoothly bending powerlaw

    Parameterization from Ryde98

    Parameters
    ----------
    x: array of independent variable
    xmax: only consider x values up to this value
    norm: normalisation at x0
    lam1: powerlaw slope below xbrk
    lam2: powerlaw slope above xbrk
    xbrk: x value where powerlaw break occurs
    Lambda: width of bend at xbrk
    """
    with np.errstate(over='ignore'):
        q = log(x/xbrk) / Lambda
        qpiv = log(x0/xbrk) / Lambda
        return norm * (x/x0)**((lam1 + lam2 + 2)/2.) * \
            ((exp(q) + exp(-q))/(exp(qpiv) + exp(-qpiv))) ** ((lam2 - lam1)/2. * Lambda) * (x0 / x)


class ActivatePL(CreationModule):
    """Activate AGN powerlaw emission

    Use ActivateLines to add emission lines
    """

    parameter_list = OrderedDict([
        ('plslope', (
            'float',
            "Powerlaw slope in the optical"
            "values between -2 and 2 are reasonable",
            6.0
        )),
        ('plbendloc', (
            'float',
            "Wavelength at which the powerlaw bends, in nm"
            "90 to 200 nm are reasonable",
            100.0
        )),
        ('plbendwidth', (
            'float',
            "Width of the powerlaw bend, in nm"
            "10 to 1000 nm are reasonable",
            10.0
        )),
        ('uvslope', (
            'float',
            "Powerlaw slope in the UV"
            "values near 0 are reasonable",
            0.0
        )),
        ('cutoff', (
            'float',
            "Powerlaw cutoff in the IR in nm. Set to -1 to not apply a cutoff.",
            10000.0
        )),
    ])


    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.plslope', self.parameters["plslope"])
        sed.add_info('agn.plbendloc', self.parameters["plbendloc"])
        sed.add_info('agn.plbendwidth', self.parameters["plbendwidth"])
        sed.add_info('agn.uvslope', self.parameters["uvslope"])
        sed.add_info('agn.cutoff', self.parameters["cutoff"])

        # convert lamLlam into Llam
        l_agn = sed.info["agn.lum5100A"] / 510
        
        assert (self.parameters["uvslope"] > self.parameters["plslope"]), (self.parameters["uvslope"], self.parameters["plslope"])
        bbb = np.empty_like(sed.wavelength_grid)
        sbpl_jitted(bbb, sed.wavelength_grid, l_agn, 
            self.parameters["uvslope"], self.parameters["plslope"],
            510.0, self.parameters["plbendloc"], self.parameters["plbendwidth"])
        assert np.isfinite(bbb).all()
        if self.parameters["cutoff"] > 0:
            bbb *= -np.expm1(-(self.parameters["cutoff"]/sed.wavelength_grid))

        sed.add_contribution('agn.activate_Disk', sed.wavelength_grid, bbb)
        sed.add_info('agn.lum2500A_disk', np.interp(250., sed.wavelength_grid, bbb), True)

# CreationModule to be returned by get_module
Module = ActivatePL
