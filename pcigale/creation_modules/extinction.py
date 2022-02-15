# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Denis Burgarella

"""
Prevot attenuation module
====================================================

This module implements attenuation based on the Prevot extinction law.

Also allows burying the AGN component with extra extinction.

"""

from collections import OrderedDict
import numpy as np
from . import CreationModule
from pcigale.data import Database

class ExtinctionLaw(CreationModule):
    """Extinction law module

    This module computes the attenuation using a wavelength-dependent factor.

    The attenuation can be computed on the whole spectrum or on a specific
    contribution and is added to the SED as a negative contribution.
    """

    parameter_list = OrderedDict([
        ("Law", (
            "string",
            "Extinction law to use",
            "Prevot"
        )),
        ("E(B-V)", (
            "float",
            "B-V attenuation applied",
            1.
        )),
        ("E(B-V)-AGN", (
            "float",
            "Additional B-V attenuation applied to activate AGN components",
            0.
        )),
        ("filters", (
            "string",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "V_B90 & FUV"
        ))
    ])
    store_filter_attenuation = True
    store_component_attenuation = True

    def _init_code(self):
        # We cannot compute the attenuation until we know the wavelengths. Yet,
        # we reserve the object.
        law = self.parameters["Law"]
        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]

        with Database() as base:
            self.law = base.get_ExtinctionLaw(law)
        self.sel_attenuation = None

    def process(self, sed):
        """Add the dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        # sed.wavelength_grid
        # sed.luminosity
        
        e_bv = float(self.parameters["E(B-V)"])
        e_bv_agn = float(self.parameters["E(B-V)-AGN"])
        
        wavelength = sed.wavelength_grid

        if self.store_filter_attenuation:
            # Fλ fluxes (only from continuum) in each filter before attenuation.
            flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute attenuation curve
        if self.sel_attenuation is None or self.sel_attenuation.shape != wavelength.shape:
            self.sel_attenuation = np.interp(wavelength, self.law.wave, self.law.k, left=0, right=0)
            self.sel_attenuation[wavelength < 60] = 100

        attenuation_total = 0.
        attenuation_total_total = 0.
        sed.add_module(self.name, self.parameters)
        for contrib in list(sed.contribution_names):
            luminosity = sed.get_lumin_contribution(contrib)
            e_bv_this = e_bv
            if 'activate' in contrib:
                e_bv_this += e_bv_agn 
            attenuated_luminosity = (luminosity * 10 **
                                     (e_bv_this * self.sel_attenuation / -2.5))
            attenuation_spectrum = attenuated_luminosity - luminosity
            attenuation_total_total += attenuation_spectrum
            if self.store_component_attenuation:
                # We integrate the amount of luminosity attenuated (-1 because the
                # spectrum is negative).
                attenuation = -1 * np.trapz(attenuation_spectrum, wavelength)
                attenuation_total += attenuation

                #sed.add_info("attenuation.E_BVs." + contrib, e_bv)
                #sed.add_info("attenuation." + contrib, attenuation, True)
                sed.add_contribution("attenuation." + contrib, wavelength,
                                     attenuation_spectrum)
        if not self.store_component_attenuation:
            sed.add_contribution("attenuation.total", wavelength,
                                 attenuation_total_total)

        if self.store_component_attenuation:
            # Total attenuation
            if 'dust.luminosity' in sed.info:
                sed.add_info("dust.luminosity",
                             sed.info["dust.luminosity"] + attenuation_total, True,
                             True)
            else:
                sed.add_info("dust.luminosity", attenuation_total, True)

        if self.store_filter_attenuation:
            # Fλ fluxes (only from continuum) in each filter after attenuation.
            flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

            # Attenuation in each filter
            with np.errstate(invalid='ignore'):
                for filt in self.filter_list:
                    sed.add_info("attenuation." + filt,
                                 -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]))

        sed.add_info('attenuation.ebv', e_bv)
        sed.add_info('attenuation.ebv_agn', e_bv_agn)

# CreationModule to be returned by get_module
Module = ExtinctionLaw
