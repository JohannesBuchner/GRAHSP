# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN module
==================================================

This module combines disk and torus.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule

class ActivateDisk(CreationModule):
    """Activate AGN dust torus emission

    Disk emission (Netzer)
    """

    parameter_list = OrderedDict([
        ('M', (
            'float',
            "BH mass"
            "Possible values are: 6.0, 8.0, 9.0",
            6.0
        )),
        ('a', (
            'float',
            "Spin parameter. "
            "Possible values are: 0, 0.998",
            0.0
        )),
        ('Mdot', (
            'float',
            "Eddington accretion rate. "
            "Possible values are: 0.03, 0.3",
            0.3
        )),
        ('inc', (
            'float',
            "Inclination. "
            "Possible values are: 0",
            0.
        )),
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        M = self.parameters["M"]
        a = self.parameters["a"]
        Mdot = self.parameters["Mdot"]
        inc = self.parameters["inc"]
        # agnType = self.parameters["AGNtype"]
        # for LINERs, we still use a type-2 template

        with Database() as base:
            self.disk = base.get_ActivateNetzerDisk(M, a, Mdot, inc)
            assert (self.disk.lumin >= 0).all()

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        l_agn = sed.info["agn.lum5100A"]
        
        #print("activate.processing with parameters:", self.parameters)
        # get existing normalisation at 5100A
        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.M', self.parameters["M"])
        sed.add_info('agn.a', self.parameters["a"])
        sed.add_info('agn.Mdot', self.parameters["Mdot"])
        sed.add_info('agn.inc', self.parameters["inc"])

        sed.add_contribution('agn.activate_Disk', self.disk.wave,
                             l_agn * self.disk.lumin)

# CreationModule to be returned by get_module
Module = ActivateDisk
