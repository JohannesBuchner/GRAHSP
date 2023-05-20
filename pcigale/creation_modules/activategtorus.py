# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN gtorus module
==================================================

This module adds a empirical "torus" mid-infrared emission based
on two gaussian and a Si feature.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule
from numpy import exp


class ActivateGTorus(CreationModule):
    """Activate Gaussian mixture torus."""

    parameter_list = OrderedDict([
        ('fcov', (
            'float',
            "Covering fraction of total at 12um, relative to disk at 510nm. 0.1 to 0.7 is recommended.",
            0.1
        )),
        ('Si', (
            'float',
            "Strength of the 12um Silicate feature (relative to the difference in Mullaney+11). -3 to 3 is reasonable.",
            0.
        )),
        ('COOLlam', (
            'float',
            "Wavelength of peak of cold dust component in um. 15-20 is reasonable.",
            17.0
        )),
        ('COOLwidth', (
            'float',
            "Standard deviation of Log-Gaussian cold dust component, in dex. 0.3-0.6 is reasonable.",
            0.45
        )),
        ('HOTlam', (
            'float',
            "Wavelength of peak of hot dust component in um. 1-4 is reasonable.",
            2.0
        )),
        ('HOTwidth', (
            'float',
            "Standard deviation of Log-Gaussian hot dust component, in dex. 0.3-0.6 is reasonable.",
            0.5
        )),
        ('HOTfcov', (
            'float',
            "Covering factor of the hot dust component. Ratio of peak to peak of cold component.",
            0.0
        ))
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        with Database() as base:
            # load average torus model spectrum
            self.log_wave = np.linspace(-0.5, 2.5, 1000)
            self.norm_index = np.argmin(np.abs(10**self.log_wave - 12))
            self.wave = 10**self.log_wave * 1000 # in nm
            # load silicate feature model spectrum
            self.si = base.get_ActivateMorNetzer2012Torus('mullaney-silicate')

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        """

        fcov = self.parameters["fcov"]
        Si = self.parameters["Si"]
        logCOOLlam = np.log10(self.parameters["COOLlam"])
        COOLwidth = self.parameters["COOLwidth"]
        HOTfcov = self.parameters["HOTfcov"]
        logHOTlam = np.log10(self.parameters["HOTlam"])
        HOTwidth = self.parameters["HOTwidth"]

        l_agn = sed.info['agn.lum5100A']

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.fcov', self.parameters["fcov"])
        sed.add_info('agn.Si', self.parameters["Si"])
        sed.add_info('agn.COOLlam', self.parameters["COOLlam"])
        sed.add_info('agn.COOLwidth', self.parameters["COOLwidth"])
        sed.add_info('agn.HOTfcov', self.parameters["HOTfcov"])
        sed.add_info('agn.HOTwidth', self.parameters["HOTwidth"])
        sed.add_info('agn.HOTlam', self.parameters["HOTlam"])

        # Add torus for NIR-MIR continuum
        # formula of Netzer (readme)
        # l_torus * 12um = 2.5 * l_agn * 510nm * fcov
        # l_agn is defined at 510nm, l_torus at 12um
        # because both are nu*L_nu = lam*L_lam normalisations, we need a
        l_torus = 2.5 * l_agn * fcov / 12.0 * 0.510
        sed.add_info('agn.lum12um', l_torus, True)

        cool_spectrum = exp(-((self.log_wave - logCOOLlam) / COOLwidth)**2)
        
        hot_spectrum = HOTfcov * 10**(logCOOLlam - logHOTlam) * exp(-((self.log_wave - logHOTlam) / HOTwidth)**2)
        total_spectrum = cool_spectrum + hot_spectrum
        # apply normalisation at 12 um:
        torus_spectrum = l_torus * total_spectrum / total_spectrum[self.norm_index]

        sed.add_contribution('agn.activate_Torus', self.wave, torus_spectrum)
        si_spectrum = l_torus * self.si.lumin * Si
        sed.add_contribution('agn.activate_TorusSi', self.si.wave, si_spectrum)
        l_torus_6um = np.interp(6000., self.wave, torus_spectrum)
        l_si_6um = np.interp(6000., self.si.wave, si_spectrum)
        sed.add_info('agn.lum6um', (l_torus_6um + l_si_6um) * 6 / 0.510, True)


# CreationModule to be returned by get_module
Module = ActivateGTorus
