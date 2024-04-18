# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN bol module
==================================================

This module computes bolometric luminosities.

"""
import numpy as np
from . import CreationModule

class ActivateBol(CreationModule):
    """Activate AGN bolometric luminosities"""

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        """

        wavelength = sed.wavelength_grid

        # Compute bolometric AGN luminosity.
        # Using all AGN components except for the torus
        # integrate from 91.2nm upwards.
        wave_mask = wavelength > 91.1753
        agn_noTOR_mask = np.array(['activate' in name and 'Torus' not in name for name in sed.contribution_names])
        BBB_luminosity = sed.luminosities[agn_noTOR_mask,:].sum(axis=0)
        LbolBBB = np.trapz(y=BBB_luminosity[wave_mask], x=wavelength[wave_mask])
        sed.add_info('agn.lumBolBBB', LbolBBB, True)

        # Compute bolometric torus luminosity
        agn_TOR_mask = np.array(['activate' in name and 'Torus' in name for name in sed.contribution_names])
        TOR_luminosity = sed.luminosities[agn_TOR_mask,:].sum(axis=0)
        LbolTOR = np.trapz(y=TOR_luminosity, x=wavelength)
        sed.add_info('agn.lumBolTOR', LbolTOR, True)

        # compute ratio of TOR to BBB
        sed.add_info('agn.ratioTORBBB', LbolTOR / LbolBBB if LbolBBB>0 else np.nan, False)

        # Compute AGN fraction with this luminosity
        gal_mask = np.array(['activate' not in name for name in sed.contribution_names])
        LbolGAL = np.trapz(sed.luminosities[gal_mask,:].sum(axis=0), x=wavelength)
        sed.add_info('agn.fracAGNTOR', LbolTOR / (LbolTOR + LbolGAL), False)

        # AGN fraction as defined in Dale+2014:
        # the luminosity fraction over 5-20 micron
        mask_wave_range_Dale = np.logical_and(wavelength >= 5000, wavelength <= 20000)
        LDaleAGN = np.trapz(
            sed.luminosities[~gal_mask,:][:,mask_wave_range_Dale].sum(axis=0),
            x=wavelength[mask_wave_range_Dale])
        LDaleGal = np.trapz(
            sed.luminosities[gal_mask,:][:,mask_wave_range_Dale].sum(axis=0),
            x=wavelength[mask_wave_range_Dale])
        sed.add_info('agn.fracAGNDale', LDaleAGN / (LDaleAGN + LDaleGal), False)

        # Compute expected normalised excess variance.
        # from Simm+16 Table 3 empirical relation.
        # from Simm+16 Table 3: normalised excess variance as a function of Lbol
        # NEV = min(0.1, 10**(-1.43 - 0.74 * np.log10(Lbol / 1e45)))
        NEV = min(0.1, 10**(-1.43 - 0.74 * np.log10(LbolBBB * 1e7 / 1e45 + 1e-10)))
        sed.add_info('agn.NEV', NEV)

# CreationModule to be returned by get_module
Module = ActivateBol
