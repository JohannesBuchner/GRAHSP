# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Denis Burgarella

"""
Charlot and Fall (2000) power law attenuation module
====================================================

This module implements the attenuation based on a power law as defined
in Charlot and Fall (2000) with a UV bump added.

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
        ("filters", (
            "string",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "V_B90 & FUV"
        ))
    ])

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
        
        wavelength = sed.wavelength_grid

        # Fλ fluxes (only from continuum) in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute attenuation curve
        if self.sel_attenuation is None or self.sel_attenuation.shape != wavelength.shape:
            self.sel_attenuation = np.interp(wavelength, self.law.wave, self.law.k, left=0, right=0)
            self.sel_attenuation[wavelength < 60] = 100

        attenuation_total = 0.
        for contrib in list(sed.contribution_names):
            luminosity = sed.get_lumin_contribution(contrib)
            attenuated_luminosity = (luminosity * 10 **
                                     (e_bv * self.sel_attenuation / -2.5))
            attenuation_spectrum = attenuated_luminosity - luminosity
            # We integrate the amount of luminosity attenuated (-1 because the
            # spectrum is negative).
            attenuation = -1 * np.trapz(attenuation_spectrum, wavelength)
            attenuation_total += attenuation

            sed.add_module(self.name, self.parameters)
            #sed.add_info("attenuation.E_BVs." + contrib, e_bv)
            #sed.add_info("attenuation." + contrib, attenuation, True)
            sed.add_contribution("attenuation." + contrib, wavelength,
                                 attenuation_spectrum)

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"]+attenuation_total, True,
                         True)
        else:
            sed.add_info("dust.luminosity", attenuation_total, True)

        # Fλ fluxes (only from continuum) in each filter after attenuation.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info("attenuation." + filt,
                         -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]))

        sed.add_info('attenuation.ebv', e_bv)

# CreationModule to be returned by get_module
Module = ExtinctionLaw
