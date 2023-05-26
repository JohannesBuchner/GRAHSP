# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Wavelength cropper module
==================================================

This module cuts the wavelength range over which SEDs are computed.

"""
from collections import OrderedDict
import numpy as np
from . import CreationModule


class Crop(CreationModule):
    """Activate AGN powerlaw emission

    Use ActivateLines to add emission lines
    """

    parameter_list = OrderedDict([
        ('minlam', (
            'float',
            "Lowest rest-frame wavelength to consider, in nm.",
            90.0
        )),
        ('maxlam', (
            'float',
            "Highest rest-frame wavelength to consider, in nm.",
            50.0
        )),
    ])


    def process(self, sed):
        """Cut the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object
        """

        mask = np.logical_and(sed.wavelength_grid >= self.parameters['minlam'], sed.wavelength_grid >= self.parameters['maxlam'])
        sed.wavelength_grid = sed.wavelength_grid[mask]
        sed.luminosity = sed.luminosity[mask]
        sed.luminosities = sed.luminosities[:,mask]

# CreationModule to be returned by get_module
Module = Crop
