# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN module
==================================================

This module adds line emissions.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule
import scipy.constants as cst

fwhm_to_sigma_conversion = 1 / (2 * np.sqrt(2 * np.log(2)))
        
class ActivateLines(CreationModule):
    """Activate AGN Emission lines (BL, Sy2 or LINER), and FeII forest
    """

    parameter_list = OrderedDict([
        ('AFeII', (
            'float',
            "Strength of FeII lines compared to Hb. Reasonable values are within 2-10",
            5
        )),
        ('AGNtype', (
            'int',
            "AGN classification: 1 (Broad lines), 2 (Sy2, narrow lines), 3 (LINER).",
            1
        )),
        ('linewidth', (
            'float',
            "Line width in km/s. Reasonable values are 100-10000."
            "Use 100 if you do not attempt to resolve the lines.",
            100
        )),
        ('linestrength_boost_factor', (
            'float',
            "factor to multiply Netzer's typical equivalent widths",
            1
        )),
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        with Database() as base:
            self.fe2 = base.get_ActivateFeIIferland()
            assert self.fe2.wave.shape == self.fe2.lumin.shape, (self.fe2.wave.shape, self.fe2.lumin.shape)
            assert (self.fe2.lumin >= 0).all()
            self.emLines = base.get_ActivateMorNetzerEmLines()
            assert (self.emLines.lumin_BLAGN >= 0).all()
            assert (self.emLines.lumin_Sy2 >= 0).all()
            assert (self.emLines.lumin_LINER >= 0).all()


        # all widths are FWHM, so
        # sigma = width / (2 * sqrt(2 * log(2)))
        
        # we do not attempt to resolve the lines
        # so choose something very small here
        self.lines_width = self.parameters["linewidth"]  # km / s
        print("line width: FWHM=", self.lines_width, "km/s")
        new_wave = np.array([])
        for line_wave in self.emLines.wave:
            # get line width in nm
            width = line_wave * (self.lines_width * 1000) / cst.c
            print("line width in nm:", width, "at", line_wave, "R:", width / line_wave)
            new_wave = np.concatenate(
                (new_wave,
                 np.linspace(line_wave - 3. * width,
                             line_wave + 3. * width,
                             9)))
        new_wave.sort()
        self.new_wave = new_wave

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        l_agn = sed.info['agn.lum5100A']
        agnType = self.parameters["AGNtype"]
        AFeII = self.parameters["AFeII"]
        linestrength_boost_factor = self.parameters["linestrength_boost_factor"]
        
        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.AFeII', AFeII)
        sed.add_info('agn.type', agnType)

        l_broadlines = 0.02 * l_agn * linestrength_boost_factor
        l_narrowlines = 0.002 * l_agn * linestrength_boost_factor
        if agnType == 1: # BLAGN
            self.add_lines(sed, 'agn.activate_EmLines_BL', self.emLines.wave,
                                 l_broadlines * self.emLines.lumin_BLAGN)
            self.add_lines(sed, 'agn.activate_EmLines_NL', self.emLines.wave,
                                 l_narrowlines * self.emLines.lumin_Sy2)
            # use FeII as well
            l_fe2 = AFeII * l_broadlines
            sed.add_contribution('agn.activate_FeLines', self.fe2.wave,
                                 l_fe2 * self.fe2.lumin)
        elif agnType == 2: # Sy2
            self.add_lines(sed, 'agn.activate_EmLines_NL', self.emLines.wave,
                                 l_narrowlines * self.emLines.lumin_Sy2)
            sed.add_contribution('agn.activate_FeLines', self.fe2.wave, 0 * self.fe2.lumin)
        elif agnType == 3: # LINER
            self.add_lines(sed, 'agn.activate_EmLines_LINER', self.emLines.wave,
                                 l_narrowlines * self.emLines.lumin_LINER)
            sed.add_contribution('agn.activate_FeLines', self.fe2.wave, 0 * self.fe2.lumin)
    
    def add_lines(self, sed, name, wave, lumin):
        """ make small gaussian lines out of the delta function information """
        # prev:
        # sed.add_contribution(name, wave, lumin)
        
        # all widths are FWHM, so
        # sigma = width / (2 * sqrt(2 * log(2)))
        
        # we do not attempt to resolve the lines
        # so choose something very small here
        new_wave = self.new_wave
        new_lumin = np.zeros_like(new_wave)
        for line_flux, line_wave in zip(lumin, wave):
            width = line_wave * (self.lines_width * 1000) / cst.c
            sigma = width * fwhm_to_sigma_conversion
            norm = 510 / np.sqrt(np.pi * sigma**2)
            shape = np.exp(-0.5 * (new_wave - line_wave) ** 2. / sigma**2)
            new_lumin += line_flux * shape * norm
        
        sed.add_contribution(name, new_wave, new_lumin)

# CreationModule to be returned by get_module
Module = ActivateLines
