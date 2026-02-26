# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

"""
Bruzual and Charlot (2003) stellar emission module
==================================================

This module implements the Bruzual and Charlot (2003) Single Stellar
Populations.

"""

from collections import OrderedDict
import numpy as np
from . import CreationModule
from ..data import Database


class BC03(CreationModule):
    """Bruzual and Charlot (2003) stellar emission module

    This SED creation module convolves the SED star formation history with a
    Bruzual and Charlot (2003) single stellar population to add a stellar
    component to the SED.
    """

    parameter_list = OrderedDict([
        ("imf", (
            "int",
            "Initial mass function: 0 (Salpeter) or 1 (Chabrier).",
            0
        )),
        ("metallicity", (
            "float",
            "Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, "
            "0.02, 0.05.",
            0.02
        )),
        ("separation_age", (
            "int",
            "Age [Myr] of the separation between the young and the old star "
            "populations. The default value in 10^7 years (10 Myr). Set "
            "to 0 not to differentiate ages (only an old population).",
            10
        ))
    ])

    def convolve(self, sfh):
        """Convolve the SSP with a Star Formation History

        Given an SFH, this method convolves the info table and the SSP
        luminosity spectrum.

        Parameters
        ----------
        sfh: array of floats
            Star Formation History in Msun/yr.

        Returns
        -------
        spec_young: array of floats
            Spectrum in W/nm of the young stellar populations.
        spec_old: array of floats
            Same as spec_young but for the old stellar populations.
        info_young: dictionary
            Dictionary containing various information from the *.?color tables
            for the young stellar populations:
            * "m_star": Total mass in stars in Msun
            * "m_gas": Mass returned to the ISM by evolved stars in Msun
            * "n_ly": rate of H-ionizing photons (s-1)
        info_old : dictionary
            Same as info_young but for the old stellar populations.
        info_all: dictionary
            Same as info_young but for the entire stellar population. Also
            contains "age_mass", the stellar mass-weighted age

        """
        # We cut the SSP to the maximum age considered to simplify the
        # computation. We take only the first three elements from the
        # info table as the others do not make sense when convolved with the
        # SFH (break strength).
        info = self.ssp.info[:, :sfh.size]
        spec = self.ssp.spec[:, :sfh.size]

        # The convolution is just a matter of reverting the SFH and computing
        # the sum of the data from the SSP one to one product. This is done
        # using the dot product. The 1e6 factor is because the SFH is in solar
        # mass per year.
        info_young = 1e6 * np.dot(info[:, :self.separation_age],
                                  sfh[-self.separation_age:][::-1])
        spec_young = 1e6 * np.dot(spec[:, :self.separation_age],
                                  sfh[-self.separation_age:][::-1])

        info_old = 1e6 * np.dot(info[:, self.separation_age:],
                                sfh[:-self.separation_age][::-1])
        spec_old = 1e6 * np.dot(spec[:, self.separation_age:],
                                sfh[:-self.separation_age][::-1])

        info_all = info_young + info_old

        info_young = dict(zip(["m_star", "m_gas", "n_ly"], info_young))
        info_old = dict(zip(["m_star", "m_gas", "n_ly"], info_old))
        info_all = dict(zip(["m_star", "m_gas", "n_ly"], info_all))

        info_all['age_mass'] = np.average(self.ssp.t[:sfh.size],
                                          weights=info[0, :] * sfh[::-1])

        return spec_young, spec_old, info_young, info_old, info_all

    def _init_code(self):
        """Read the SSP from the database."""
        self.imf = int(self.parameters["imf"])
        self.Z = float(self.parameters["metallicity"])
        self.separation_age = int(self.parameters["separation_age"])

        with Database() as database:
            if self.imf == 0:
                self.ssp = database.get_bc03(imf='salp', metallicity=self.Z)
            elif self.imf == 1:
                self.ssp = database.get_bc03(imf='chab', metallicity=self.Z)
            else:
                raise Exception(f"IMF #{self.imf} unknown")

    def process(self, sed):
        """Add the convolution of a Bruzual and Charlot SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        out = self.convolve(sed.sfh)
        spec_young, spec_old, info_young, info_old, info_all = out

        # We compute the Lyman continuum luminosity as it is important to
        # compute the energy absorbed by the dust before ionising gas.
        wave = self.ssp.wl
        w = np.where(wave <= 91.1)
        lum_lyc_young, lum_lyc_old = np.trapezoid([spec_young[w], spec_old[w]],
                                              wave[w])

        # We do similarly for the total stellar luminosity
        lum_young, lum_old = np.trapezoid([spec_young, spec_old], wave)

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.Z)
        sed.add_info("stellar.old_young_separation_age", self.separation_age,
                     unit='Myr')

        sed.add_info("stellar.m_star_young", info_young["m_star"], True,
                     unit='solMass')
        sed.add_info("stellar.m_gas_young", info_young["m_gas"], True,
                     unit='solMass')
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True,
                     unit='ph/s')
        sed.add_info("stellar.lum_ly_young", lum_lyc_young, True, unit='W')
        sed.add_info("stellar.lum_young", lum_young, True, unit='W')

        sed.add_info("stellar.m_star_old", info_old["m_star"], True,
                     unit='solMass')
        sed.add_info("stellar.m_gas_old", info_old["m_gas"], True,
                     unit='solMass')
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True, unit='ph/s')
        sed.add_info("stellar.lum_ly_old", lum_lyc_old, True, unit='W')
        sed.add_info("stellar.lum_old", lum_old, True, unit='W')

        sed.add_info("stellar.m_star", info_all["m_star"], True, unit='solMass')
        sed.add_info("stellar.m_gas", info_all["m_gas"], True, unit='solMass')
        sed.add_info("stellar.n_ly", info_all["n_ly"], True, unit='ph/s')
        sed.add_info("stellar.lum_ly", lum_lyc_young + lum_lyc_old, True, unit='W')
        sed.add_info("stellar.lum", lum_young + lum_old, True, unit='W')
        sed.add_info("stellar.age_m_star", info_all["age_mass"], unit='Myr')

        sed.add_contribution("stellar.old", wave, spec_old)
        sed.add_contribution("stellar.young", wave, spec_young)


# CreationModule to be returned by get_module
Module = BC03
