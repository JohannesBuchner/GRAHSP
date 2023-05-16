# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN torus module
==================================================

This module adds the "torus" mid-infrared emission.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule
        
class ActivateTorus(CreationModule):
    """Activate torus."""

    parameter_list = OrderedDict([
        ('fcov', (
            'float',
            "Torus Covering fraction. 0.1 to 0.7 is recommended.",
            0.1
        )),
        ('Si', (
            'float',
            "Strength of the 12um Silicate feature (relative to the difference in Mullaney+11). -3 to 3 is reasonable.",
            0.
        )),
        ('TORtemp', (
            'float',
            "Steepness of the torus spectrum (relative to the spread in Mor&Netzer+09). -3 to 3 is reasonable.",
            0.
        )),
        ('TORcutoff', (
            'float',
            "Wavelength cutoff in um.",
            1.2
        ))
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        with Database() as base:
            # load average torus model spectrum
            self.torus_avg = base.get_ActivateMorNetzer2012Torus('mor-avg')
            self.torus_lo = base.get_ActivateMorNetzer2012Torus('mor-lo')
            self.torus_hi = base.get_ActivateMorNetzer2012Torus('mor-hi')
            assert (self.torus_avg.lumin >= 0).all()
            # load silicate feature model spectrum
            self.si = base.get_ActivateMorNetzer2012Torus('mullaney-silicate')

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        """

        fcov = self.parameters["fcov"]
        # agnType = self.parameters["AGNtype"]
        Si = self.parameters["Si"]
        TORtemp = self.parameters["TORtemp"]
        TORcut = self.parameters["TORcutoff"]

        l_agn = sed.info['agn.lum5100A']
        
        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.fcov', self.parameters["fcov"])
        sed.add_info('agn.Si', self.parameters["Si"])
        sed.add_info('agn.TORtemp', self.parameters["TORtemp"])
        sed.add_info('agn.TORcutoff', self.parameters["TORcutoff"])

        
        # Add torus for NIR-MIR continuum
        # formula of Netzer (readme)
        # l_torus * 12um = 2.5 * l_agn * 510nm * fcov
        # l_agn is defined at 510nm, l_torus at 12um
        # because both are nu*L_nu = lam*L_lam normalisations, we need a
        l_torus = 2.5 * l_agn * fcov / 12.0 * 0.510
        sed.add_info('agn.lum12um', l_torus, True)
        if TORtemp > 0:
            torus_deviation = (self.torus_hi.lumin - self.torus_avg.lumin) * TORtemp
        else:
            torus_deviation = (self.torus_lo.lumin - self.torus_avg.lumin) * -TORtemp
        
        # gaussian-like cut-off at low wavelengths
        # approximates Lyu&Rieke (TORcut=1.7) templates and Mor&Netzer (TORcut=1.2) templates
        cutoff = 1 - np.exp( - (self.torus_avg.wave / 1000 / TORcut)**2)

        torus_spectrum = l_torus * (self.torus_avg.lumin + torus_deviation) * cutoff
        sed.add_contribution('agn.activate_Torus', self.torus_avg.wave, torus_spectrum)
        si_spectrum = l_torus * self.si.lumin * Si
        sed.add_contribution('agn.activate_TorusSi', self.si.wave, si_spectrum)
        l_torus_6um = np.interp(6000., self.torus_avg.wave, torus_spectrum)
        l_si_6um = np.interp(6000., self.si.wave, si_spectrum)
        sed.add_info('agn.lum6um', (l_torus_6um + l_si_6um) * 6 / 0.510, True)
    
# CreationModule to be returned by get_module
Module = ActivateTorus
