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
            "Covering factor of the hot dust component. Ratio of peak to peak of cold component in lambdaLlambda.",
            0.0
        )),
        ('SiRatio', (
            'float',
            "Si absorption to emission ratio.",
            0.29
        )),
        ('SiEmlam', (
            'float',
            "Wavelength of Si emission feature in nm.",
            9841
        )),
        ('SiAbslam', (
            'float',
            "Wavelength of Si absorption feature in nm.",
            14224
        )),
        ('SiEmwidth', (
            'float',
            "Width of Si absorption feature in nm.",
            1025.3
        )),
        ('SiAbswidth', (
            'float',
            "Width of Si absorption feature in nm.",
            1163.5
        )),
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        self.log_wave = np.linspace(-0.5, 2.5, 1000)
        self.norm_index = np.argmin(np.abs(10**self.log_wave - 12))
        self.wave = 10**self.log_wave * 1000 # in nm
        self.Siwave = self.wave[np.logical_and(self.wave > 8015, self.wave < 19000)]
        
        self.fcov = self.parameters["fcov"]
        self.logCOOLlam = np.log10(self.parameters["COOLlam"])
        self.COOLlam = self.parameters["COOLlam"]
        self.COOLwidth = self.parameters["COOLwidth"]
        self.HOTfcov = self.parameters["HOTfcov"]
        self.logHOTlam = np.log10(self.parameters["HOTlam"])
        self.HOTlam = self.parameters["HOTlam"]
        self.HOTwidth = self.parameters["HOTwidth"]

        self.Si = self.parameters["Si"]
        self.SiEmAmpl = 0.4
        self.SiEmlam = self.parameters["SiEmlam"]
        self.SiEmWidth = self.parameters["SiEmWidth"]
        self.SiAbsAmpl = self.SiEmAmpl * self.parameters["SiRatio"]
        self.SiAbslam = self.parameters["SiAbslam"]
        self.SiAbsWidth = self.parameters["SiAbsWidth"]
        

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        """


        l_agn = sed.info['agn.lum5100A']

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.fcov', self.fcov)
        sed.add_info('agn.Si', self.Si)
        sed.add_info('agn.COOLlam', self.COOLlam)
        sed.add_info('agn.COOLwidth', self.COOLwidth)
        sed.add_info('agn.HOTfcov', self.HOTfcov)
        sed.add_info('agn.HOTwidth', self.HOTwidth)
        sed.add_info('agn.HOTlam', self.HOTlam)

        # Add torus for NIR-MIR continuum
        # formula of Netzer (readme)
        # l_torus * 12um = 2.5 * l_agn * 510nm * fcov
        # l_agn is defined at 510nm, l_torus at 12um
        # because both are nu*L_nu = lam*L_lam normalisations, we need a
        l_torus = 2.5 * l_agn * self.fcov / 12.0 * 0.510
        sed.add_info('agn.lum12um', l_torus, True)

        cool_spectrum = exp(-((self.log_wave - self.logCOOLlam) / self.COOLwidth)**2)
        
        hot_spectrum = self.HOTfcov * 10**(self.logCOOLlam - self.logHOTlam) * exp(-((self.log_wave - self.logHOTlam) / self.HOTwidth)**2)
        total_spectrum = cool_spectrum + hot_spectrum
        # apply normalisation at 12 um:
        torus_spectrum = l_torus * total_spectrum / total_spectrum[self.norm_index]
        si_spectrum = l_torus * self.Si * (
            self.SiEmAmpl * exp(-0.5 * ((self.Siwave - self.SiEmAmpl) / self.SiEmWidth)**2) - 
            self.SiAbsAmpl * exp(-0.5 * ((self.Siwave - self.SiAbsAmpl) / self.SiAbsWidth)**2))

        sed.add_contribution('agn.activate_Torus', self.wave, torus_spectrum)
        sed.add_contribution('agn.activate_Torus_Si', self.Siwave, si_spectrum)
        l_torus_6um = np.interp(6000., self.wave, torus_spectrum)
        sed.add_info('agn.lum6um', l_torus_6um * 6 / 0.510, True)


# CreationModule to be returned by get_module
Module = ActivateGTorus
