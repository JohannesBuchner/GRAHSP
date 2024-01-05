# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN module
==================================================

This module prepares the use of activate AGN components.

"""
from collections import OrderedDict
import numpy as np
from . import CreationModule

class Activate(CreationModule):
    """Activate AGN dust torus emission

    Combination of
    
    * Disk emission (Netzer)
    * Torus emission (Mor & Netzer 2012)
    
    Use ActivateLines to add emission lines
    """

    parameter_list = OrderedDict([
        ('fracAGN', (
            'float',
            "AGN fraction at 510nm.",
            0.1
        )),
    ])

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        fracAGN = self.parameters["fracAGN"]
        # Compute the AGN luminosity
        if fracAGN < 0.:
            l_agn = 1.
            scales_with_mass = False
        elif fracAGN < 1.:
            luminosity = np.interp(510.0, sed.wavelength_grid, sed.luminosity)
            assert luminosity >= 0, luminosity
            l_agn = luminosity * (1./(1.-fracAGN) - 1.) * 510
            scales_with_mass = True
        else:
            raise Exception("AGN fraction is exactly 1. Behaviour "
                            "undefined.")
        assert l_agn >= 0, l_agn
        
        sed.add_info('agn.lum5100A', l_agn, scales_with_mass)

# CreationModule to be returned by get_module
Module = Activate
