# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN lines module
==================================================

This module adds line emissions.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule
import scipy.constants as cst
from scipy.special import erf

fwhm_to_sigma_conversion = 1 / (2 * np.sqrt(2 * np.log(2)))

def blackbody_flux_frequency(T_K, wavelength_nm):
    """Planck Law in flux per unit frequency.

    Parameters
    ----------
    T_K
        Temperature in Kelvin.
    wavelength_nm: float
        Wavelength in nm.

    Returns
    -------
    flux: float
        spectral flux density per unit frequency.
    """
    # (Planck constant h) * (speed of light) / (Boltzmann constant k_B) in nm * K
    h_c_per_k_B = 1.439e7
    return (wavelength_nm**-3) / (np.expm1(h_c_per_k_B / (T_K * wavelength_nm)))

def blackbody_flux_wavelength(T_K, wavelength_nm):
    """Planck Law in flux per unit wavelength.

    Parameters
    ----------
    T_K
        Temperature in Kelvin.
    wavelength_nm: float
        Wavelength in nm.

    Returns
    -------
    flux: float
        spectral flux density per unit frequency.
    """
    # (Planck constant h) * (speed of light) / (Boltzmann constant k_B) in nm * K
    h_c_per_k_B = 1.439e7
    return (wavelength_nm**-5) / (np.expm1(h_c_per_k_B / (T_K * wavelength_nm)))


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
            "Use 1000 if you do not attempt to resolve the lines.",
            5000
        )),
        ('Alines', (
            'float',
            "Factor to multiply Netzer's typical equivalent widths.",
            1
        )),
        ('ABC', (
            'float',
            "Strength of the Balmer continuum relative to the powerlaw at 3000nm.",
            0.0
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
        self.narrow_lines_width = self.lines_width
        # sensible change for future versions:
        # self.narrow_lines_width = 500
        new_wave = np.array([])
        for line_wave in self.emLines.wave:
            # get line width in nm
            width = line_wave * (self.lines_width * 1000) / cst.c
            new_wave = np.concatenate(
                (new_wave,
                 np.linspace(line_wave - 3. * width,
                             line_wave + 3. * width,
                             9)))
        new_wave.sort()
        self.new_wave = new_wave

        self.agnType = self.parameters["AGNtype"]
        self.AFeII = self.parameters["AFeII"]
        self.Alines = self.parameters["Alines"]
        self.ABC = self.parameters["ABC"]
        
        # compute Balmer continuum following Grandi (1982)
        # Balmer edge wavelength in nm
        self.BE_wave = 364.6
        # Balmer continuum optical depth
        self.BC_tau = 1.0
        # Black body temperature in Kelvin
        self.BC_T = 15000
        self.BC_wave = self.fe2.wave[self.fe2.wave <= self.BE_wave]
        self.BC_wave_ratio = self.BC_wave / self.BE_wave
        # compute Balmer black-body
        # (Planck constant h) * (speed of light) / (Boltzmann constant k_B) in nm * K
        h_c_per_k_B = 1.439e7
        black_body_BC = (self.BC_wave**-5) / np.expm1(h_c_per_k_B / (self.BC_T * self.BC_wave))
        truncation = -np.expm1(-self.BC_tau * self.BC_wave_ratio**3)
        black_body_BC0 = (self.BE_wave**-5) / np.expm1(h_c_per_k_B / (self.BC_T * self.BE_wave))
        truncation0 = -np.expm1(-self.BC_tau)

        # the following is based on a derivation of 
        # convolving a gaussian with a linear approximation
        #    truncation ~= alpha * x + beta
        # with the values:
        alpha = 1.8
        beta = -0.8
        # this leads to a Gaussian CDF term (termB) and a
        #   more complicated result from the x * Gaussian(x) integration (termA1-3)
        sigma = (self.lines_width * 1000) / cst.c 
        x = self.BC_wave_ratio
        z = (x - 1) * 2**-0.5 / sigma
        termB = 0.5 * (1 - erf(z))
        termA1 = 0.5 * x
        termA2 = -0.5 * x * erf(z)
        termA3 = -sigma * (2 * np.pi)**-0.5 * np.exp(-z**2)
        convolved = (beta * termB + alpha * (termA1 + termA2 + termA3)) * (1 - np.exp(-1))
        # 250 nm is >3 sigma away from the balmer edge up to 30000km/s
        #   below we can use the original truncation formula
        truncation_convolved = np.where(self.BC_wave > 250, convolved, truncation)

        self.BC = black_body_BC / black_body_BC0 * truncation_convolved / truncation0




    def process(self, sed):
        """Add the line contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """

        self.l_agn = sed.info['agn.lum5100A'] / 510
        l_broadlines = 0.02 * self.l_agn * self.Alines
        l_narrowlines = 0.002 * self.l_agn * self.Alines
        
        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.AFeII', self.AFeII)
        sed.add_info('agn.Alines', self.Alines)
        sed.add_info('agn.type', self.agnType)
        sed.add_info('agn.ABC', self.ABC)

        if self.agnType == 1: # BLAGN
            self.add_lines(sed, 'agn.activate_EmLines_BL', self.emLines.wave,
                                 l_broadlines * self.emLines.lumin_BLAGN, self.lines_width)
            self.add_lines(sed, 'agn.activate_EmLines_NL', self.emLines.wave,
                                 l_narrowlines * self.emLines.lumin_Sy2, self.narrow_lines_width)
            # use FeII as well
            l_fe2 = self.AFeII * l_broadlines
            sed.add_contribution('agn.activate_FeLines', self.fe2.wave,
                                 l_fe2 * self.fe2.lumin)
            if self.ABC > 0:
                l_BC = self.l_agn * self.ABC
                sed.add_contribution('agn.activate_BC', self.BC_wave, l_BC * self.BC)
        elif self.agnType == 2: # Sy2
            self.add_lines(sed, 'agn.activate_EmLines_NL', self.emLines.wave,
                                 l_narrowlines * self.emLines.lumin_Sy2, self.narrow_lines_width)
            sed.add_contribution('agn.activate_FeLines', self.fe2.wave, 0 * self.fe2.lumin)
        elif self.agnType == 3: # LINER
            self.add_lines(sed, 'agn.activate_EmLines_LINER', self.emLines.wave,
                                 l_narrowlines * self.emLines.lumin_LINER, self.narrow_lines_width)
            sed.add_contribution('agn.activate_FeLines', self.fe2.wave, 0 * self.fe2.lumin)
    
    def add_lines(self, sed, name, wave, lumin, lines_width):
        """add Gaussian lines to SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object
        name: name of the contribution
        wave: array of wavelengths of the lines
        lumin: array of equivalent widths of the lines
        lines_width: line width in km/s
        """
        # all widths are FWHM, so
        # sigma = width / (2 * sqrt(2 * log(2)))
        # the amplitude is computed from width, EW and luminosity.
        
        new_wave = self.new_wave
        new_lumin = np.zeros_like(new_wave)
        for line_flux, line_wave in zip(lumin, wave):
            width = line_wave * (lines_width * 1000) / cst.c
            sigma = width * fwhm_to_sigma_conversion
            norm = 510 / np.sqrt(np.pi * sigma**2)
            shape = np.exp(-0.5 * (new_wave - line_wave) ** 2. / sigma**2)
            new_lumin += line_flux * shape * norm
        
        sed.add_contribution(name, new_wave, new_lumin)

# CreationModule to be returned by get_module
Module = ActivateLines
