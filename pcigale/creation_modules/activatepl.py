# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate Powerlaw AGN module
==================================================

This module uses a simple powerlaw for the BBB,
with a turn-over.

"""
from collections import OrderedDict
import numpy as np
from numpy import exp, log
from . import CreationModule

def sbpl(x, norm, lam1, lam2, x0, xbrk, Lambda):
    """Smoothly bending powerlaw

    Parameterization from Ryde98
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

        l_agn = sed.info["agn.lum5100A"]
        
        assert (self.parameters["uvslope"] > self.parameters["plslope"]), (self.parameters["uvslope"], self.parameters["plslope"])
        bbb = sbpl(sed.wavelength_grid, l_agn, 
            self.parameters["uvslope"], self.parameters["plslope"],
            510.0, self.parameters["plbendloc"], self.parameters["plbendwidth"])
        assert np.isfinite(bbb).all()
        
        sed.add_contribution('agn.activate_Disk', sed.wavelength_grid, bbb)

# CreationModule to be returned by get_module
Module = ActivatePL
