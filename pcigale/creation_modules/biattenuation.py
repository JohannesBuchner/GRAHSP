# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Denis Burgarella

"""
Prevot attenuation module
====================================================

This module implements attenuation based on the Prevot-like SMC law.

Also allows burying the AGN component with extra attenuation.

"""

from collections import OrderedDict
import numpy as np
from . import CreationModule


class BiAttenuationLaw(CreationModule):
    """Attenuation law module

    This module computes the attenuation using a wavelength-dependent factor.

    The attenuation can be computed on the whole spectrum or on a specific
    contribution and is added to the SED as a negative contribution.

    The attenuation is applied to all SED components. To activate* components,
    an additional attenuation is applied.
    """

    parameter_list = OrderedDict([
        ("OPT_index", (
            "float",
            "Powerlaw index for attenuation law in the optical. Use -1.2 for Prevot",
            -1.2
        )),
        ("NIR_index", (
            "float",
            "Powerlaw index for attenuation law in the NIR. Use -2.6 for Prevot",
            -3.0
        )),
        ("norm", (
            "float",
            "Attenuation law normalisation at lam_break. Use 1.2 for Prevot",
            1.2
        )),
        ("lam_break", (
            "float",
            "Attenuation law powerlaw break in nm. Use 1100 for Prevot",
            1100
        )),
        ("E(B-V)", (
            "float",
            "B-V attenuation applied",
            1.0
        )),
        ("E(B-V)-AGN", (
            "float",
            "Additional B-V attenuation applied to activate AGN components",
            0.0
        )),
        ("filters", (
            "string",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            ""
        ))
    ])
    store_filter_attenuation = True
    store_component_attenuation = True

    def _init_code(self):
        # We cannot compute the attenuation until we know the wavelengths. Yet,
        # we reserve the object.
        if self.parameters["filters"].strip() != '':
            self.filter_list = [item.strip() for item in
                                self.parameters["filters"].split("&")]
        else:
            self.filter_list = []

    def process(self, sed):
        """Add the dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        wavelength = sed.wavelength_grid

        law_index_OPT = self.parameters["OPT_index"]
        law_index_NIR = self.parameters["NIR_index"]
        law_lam_break = self.parameters["lam_break"]
        law_norm = self.parameters["norm"]
        e_bv = self.parameters["E(B-V)"]
        e_bv_agn = self.parameters["E(B-V)-AGN"]

        sed.add_module(self.name, self.parameters)
        sed.add_info('attenuation.ebv', e_bv)
        sed.add_info('attenuation.ebv_agn', e_bv_agn)

        if self.store_filter_attenuation:
            # Fλ fluxes (only from continuum) in each filter before attenuation.
            flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        mask_agn_contrib = np.empty(len(sed.contribution_names), dtype=bool)
        for i, contrib in enumerate(sed.contribution_names):
            mask_agn_contrib[i] = 'activate' in contrib
            del i, contrib

        # handle only part where attenuation matters
        wavelength_negligible_attenuation = wavelength > 40000
        if np.any(wavelength_negligible_attenuation) and False:
            imax = np.where(wavelength_negligible_attenuation)[0][0]
        else:
            imax = None
        attenuation_curve = (law_norm * (wavelength[:imax] / law_lam_break)**(
            np.where(wavelength[:imax] < law_lam_break, law_index_OPT, law_index_NIR))
        ).reshape((1, -1))

        if self.store_component_attenuation:
            attenuated_luminosities = sed.luminosities.copy()
            attenuated_luminosities[~mask_agn_contrib,:imax] *= 10**(e_bv * attenuation_curve / -2.5)
            attenuated_luminosities[mask_agn_contrib, :imax] *= 10**((e_bv + e_bv_agn) * attenuation_curve / -2.5)
            attenuation_spectra = attenuated_luminosities - sed.luminosities
            # We integrate the amount of luminosity attenuated (-1 because the
            # spectrum is negative).
            attenuation_total_gal = -1 * np.trapz(attenuation_spectra[~mask_agn_contrib,:imax].sum(axis=0), wavelength[:imax])
            for contrib, attenuation_spectrum in zip(sed.contribution_names, attenuation_spectra):
                sed.add_contribution("attenuation." + contrib, wavelength,
                                     attenuation_spectrum)
        else:
            # compute accurately, to avoid negative fluxes
            sed.luminosity = sed.luminosities.sum(axis=0)
            gal_emission = sed.luminosities[~mask_agn_contrib,:imax].sum(axis=0)
            nonagn_luminosities = gal_emission * 10**(e_bv * attenuation_curve[0,:imax] / -2.5)
            agn_luminosities = sed.luminosities[mask_agn_contrib,:imax].sum(axis=0) * 10**((e_bv + e_bv_agn) * attenuation_curve[0,:imax] / -2.5)
            attenuation_total_gal = -np.trapz(nonagn_luminosities - gal_emission, wavelength[:imax])
            attenuation_total_total2 = -sed.luminosity
            attenuation_total_total2[:imax] += nonagn_luminosities + agn_luminosities
            mask_negative = attenuation_total_total2 < -sed.luminosity
            attenuation_total_total2[mask_negative] = -sed.luminosity[mask_negative]
            sed.add_contribution("attenuation.total", wavelength,
                                 attenuation_total_total2)

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"] + attenuation_total_gal, True,
                         True)
        else:
            sed.add_info("dust.luminosity", attenuation_total_gal, True)

        if self.store_filter_attenuation:
            # Fλ fluxes (only from continuum) in each filter after attenuation.
            flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

            # Attenuation in each filter
            with np.errstate(invalid='ignore'):
                for filt in self.filter_list:
                    sed.add_info("attenuation." + filt,
                                 -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]))


# CreationModule to be returned by get_module
Module = BiAttenuationLaw
