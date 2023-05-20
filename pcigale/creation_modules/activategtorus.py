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
        ('SiEmWidth', (
            'float',
            "Width of Si absorption feature in nm.",
            1025.3
        )),
        ('SiAbsWidth', (
            'float',
            "Width of Si absorption feature in nm.",
            1163.5
        )),
    ])

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        """

        fcov = self.parameters["fcov"]
        logCOOLlam = np.log10(self.parameters["COOLlam"])
        COOLlam = self.parameters["COOLlam"]
        COOLwidth = self.parameters["COOLwidth"]
        HOTfcov = self.parameters["HOTfcov"]
        logHOTlam = np.log10(self.parameters["HOTlam"])
        HOTlam = self.parameters["HOTlam"]
        HOTwidth = self.parameters["HOTwidth"]

        Si = self.parameters["Si"]
        SiEmAmpl = 0.4
        SiEmlam = self.parameters["SiEmlam"]
        SiEmWidth = self.parameters["SiEmWidth"]
        SiAbsAmpl = SiEmAmpl * self.parameters["SiRatio"]
        SiAbslam = self.parameters["SiAbslam"]
        SiAbsWidth = self.parameters["SiAbsWidth"]

        l_agn = sed.info['agn.lum5100A']

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.fcov', fcov)
        sed.add_info('agn.Si', Si)
        sed.add_info('agn.COOLlam', COOLlam)
        sed.add_info('agn.COOLwidth', COOLwidth)
        sed.add_info('agn.HOTfcov', HOTfcov)
        sed.add_info('agn.HOTwidth', HOTwidth)
        sed.add_info('agn.HOTlam', HOTlam)

        # Add torus for NIR-MIR continuum
        # formula of Netzer (readme)
        # l_torus * 12um = 2.5 * l_agn * 510nm * fcov
        # l_agn is defined at 510nm, l_torus at 12um
        # because both are nu*L_nu = lam*L_lam normalisations, we need a
        l_torus = 2.5 * l_agn * fcov / 12.0 * 0.510
        sed.add_info('agn.lum12um', l_torus, True)

        wave = sed.wavelength_grid
        log_wave = np.log10(sed.wavelength_grid / 1000)
        cool_spectrum = exp(-((log_wave - logCOOLlam) / COOLwidth)**2)
        
        hot_spectrum = HOTfcov * 10**(logCOOLlam - logHOTlam) * exp(-((log_wave - logHOTlam) / HOTwidth)**2)
        total_spectrum = cool_spectrum + hot_spectrum
        norm12 = np.interp(12000, sed.wavelength_grid, total_spectrum)
        # apply normalisation at 12 um:
        torus_spectrum = l_torus * total_spectrum / norm12
        sed.add_contribution('agn.activate_Torus', wave, torus_spectrum)

        si_spectrum = l_torus * Si * (
            SiEmAmpl * exp(-0.5 * ((sed.wavelength_grid - SiEmlam) / SiEmWidth)**2) - 
            SiAbsAmpl * exp(-0.5 * ((sed.wavelength_grid - SiAbslam) / SiAbsWidth)**2))
        sed.add_contribution('agn.activate_Torus_Si', wave, si_spectrum)
        l_torus_6um = np.interp(6000., sed.wavelength_grid, torus_spectrum)
        sed.add_info('agn.lum6um', l_torus_6um * 6 / 0.510, True)


# CreationModule to be returned by get_module
Module = ActivateGTorus
