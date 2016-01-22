# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Pacifici 2012 Galaxy model
==================================================

This module uses a self-consistent simulation of galaxy stellar population
 and gas.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule

class Pacifici2012(CreationModule):
    """Pacifici 2012 Galaxy model (gas+stars)"""

    parameter_list = OrderedDict([
        ('template', (
            'str',
            "template name"
            "Possible values are: {sf,qui}[1-9]",
            'sf1'
        )),
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        name = self.parameters["template"]

        with Database() as base:
            self.gal = base.get_ActivatePacifici2012Gal(name)
            

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        sed.add_module(self.name, self.parameters)
        #sed.add_info('gal.SFR', self.parameters["M"])
        sed.add_contribution('gal.Pacifici2012', self.gal.wave, self.gal.lumin)

# CreationModule to be returned by get_module
Module = Pacifici2012
