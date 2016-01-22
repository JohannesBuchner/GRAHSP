# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN module
==================================================

This module combines disk, torus and line emissions.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule
import scipy.constants as cst

class Activate(CreationModule):
    """Activate AGN dust torus emission

    Combination of
    
    * Disk emission (Netzer)
    * Torus emission (Mor & Netzer 2012)
    * Emission lines (BL, Sy2 or LINER), and FeII forest
    

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
        ('fcov', (
            'float',
            "Torus Covering fraction.",
            0.1
        )),
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
        ('fracAGN', (
            'float',
            "AGN fraction at 510nm.",
            0.1
        ))
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        M = self.parameters["M"]
        a = self.parameters["a"]
        Mdot = self.parameters["Mdot"]
        inc = self.parameters["inc"]
        agnType = self.parameters["AGNtype"]

        with Database() as base:
            self.disk = base.get_ActivateNetzerDisk(M, a, Mdot, inc)
            assert (self.disk.lumin >= 0).all()
            self.torus = base.get_ActivateMorNetzer2012Torus()
            assert (self.torus.lumin >= 0).all()
            self.fe2 = base.get_ActivateFeIIferland()
            assert self.fe2.wave.shape == self.fe2.lumin.shape, (self.fe2.wave.shape, self.fe2.lumin.shape)
            assert (self.fe2.lumin >= 0).all()
            self.emLines = base.get_ActivateMorNetzerEmLines()
            assert (self.emLines.lumin_BLAGN >= 0).all()
            assert (self.emLines.lumin_Sy2 >= 0).all()
            assert (self.emLines.lumin_LINER >= 0).all()
            

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        fracAGN = self.parameters["fracAGN"]
        fcov = self.parameters["fcov"]
        agnType = self.parameters["AGNtype"]
        AFeII = self.parameters["AFeII"]
        
        print("activate.processing with parameters:", self.parameters)
        # get existing normalisation at 5100A
        luminosity = np.interp(510.0, sed.wavelength_grid, sed.luminosity)
        assert luminosity >= 0, luminosity
        
        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.M', self.parameters["M"])
        sed.add_info('agn.a', self.parameters["a"])
        sed.add_info('agn.Mdot', self.parameters["Mdot"])
        sed.add_info('agn.inc', self.parameters["inc"])
        sed.add_info('agn.AFeII', self.parameters["AFeII"])
        sed.add_info('agn.fcov', self.parameters["fcov"])
        sed.add_info('agn.type', self.parameters["AGNtype"])
        sed.add_info('agn.fracAGN', self.parameters["fracAGN"])

        
        # Compute the AGN luminosity
        if fracAGN < 1.:
            l_agn = luminosity * (1./(1.-fracAGN) - 1.)
        else:
            raise Exception("AGN fraction is exactly 1. Behaviour "
                            "undefined.")
        assert l_agn >= 0, l_agn
        
        # Add disk
        sed.add_contribution('agn.activate_Disk', self.disk.wave,
                             l_agn * self.disk.lumin)
        #print(' disk', self.disk.wave, l_agn, self.disk.lumin)

        # Add torus for NIR-MIR continuum
        # formula of Netzer (readme)
        l_torus = 1.25 * l_agn * fcov
        sed.add_contribution('agn.activate_Torus', self.torus.wave,
                             l_torus * self.torus.lumin)
        #print(' torus', self.torus.wave, l_torus, self.torus.lumin)

        l_broadlines = 0.02 * l_agn
        l_narrowlines = 0.002 * l_agn
        if agnType == 1: # BLAGN
            linecol = 0
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
        elif agnType == 3: # LINER
            self.add_lines(sed, 'agn.activate_EmLines_LINER', self.emLines.wave,
                                 l_narrowlines * self.emLines.lumin_LINER)
    
    def add_lines(self, sed, name, wave, lumin):
        """ make small gaussian lines out of the delta function information """
        # prev:
        # sed.add_contribution(name, wave, lumin)
        
        # we do not attempt to resolve the lines
        # so choose something very small here
        lines_width = 100 # km / s
        new_wave = np.array([])
        for line_wave, line_lumin in zip(wave, lumin):
            # get line width in nm
            width = line_wave * (lines_width * 1000) / cst.c
            new_wave = np.concatenate((new_wave,
                                           np.linspace(line_wave - 3. * width,
                                                       line_wave + 3. * width,
                                                       9)))
        new_wave.sort()
        new_lumin = np.zeros_like(new_wave)
        for line_flux, line_wave in zip(lumin, wave):
            width = line_wave * (lines_width * 1000) / cst.c
            new_lumin += (line_flux * np.exp(- 4. * np.log(2.) *
                        (new_wave - line_wave) ** 2. / (width * width)) /
                        (width * np.sqrt(np.pi / np.log(2.)) / 2.))
        
        sed.add_contribution(name, new_wave, new_lumin)
        

# CreationModule to be returned by get_module
Module = Activate
