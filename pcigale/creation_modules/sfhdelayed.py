# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Copyright (C) 2014 University of Cambridge
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Véronique Buat & Médéric Boquien

"""
Delayed tau model for star formation history with an optional exponential burst
===============================================================================

This module implements a star formation history (SFH) described as a delayed
rise of the SFR up to a maximum, followed by an exponential decrease. Optionally
a decreasing exponential burst can be added to model a recent episode of star
formation.

"""

from collections import OrderedDict
import numpy as np
from . import CreationModule

class SFHDelayed(CreationModule):
    """Delayed tau model for Star Formation History

    This module sets the SED star formation history (SFH) proportional to time,
    with a declining exponential parametrised with a time-scale (tau_main).

    """

    parameter_list = OrderedDict([
        ("tau_main", (
            "float",
            "e-folding time of the main stellar population model in Myr.",
            2000.
        )),
        ("age_main", (
            "integer",
            "Age of the oldest stars in the galaxy in Myr. The precision "
            "is 1 Myr.",
            5000.
        )),
        ("tau_burst", (
            "float",
            "e-folding time of the late starburst population model in Myr.",
            50.
        )),
        ("age_burst", (
            "integer",
            "Age of the late burst in Myr. The precision is 1 Myr.",
            20.
        )),
        ("f_burst", (
            "float",
            "Mass fraction of the late burst population.",
            0.
        )),
        ("sfr_A", (
            "float",
            "Multiplicative factor controlling the SFR if normalise is False. "
            "For instance without any burst: SFR(t)=sfr_A×t×exp(-t/τ)/τ²",
            1.
        )),
        ("normalise", (
            "boolean",
            "Normalise the SFH to produce one solar mass.",
            "True"
        ))
    ])

    def _init_code(self):
        self.tau_main = float(self.parameters["tau_main"])
        self.age_main = int(self.parameters["age_main"])
        self.tau_burst = float(self.parameters["tau_burst"])
        self.age_burst = int(self.parameters["age_burst"])
        self.f_burst = float(self.parameters["f_burst"])
        sfr_A = float(self.parameters["sfr_A"])
        if isinstance(self.parameters["normalise"], str):
            normalise = self.parameters["normalise"].lower() == 'true'
        else:
            normalise = bool(self.parameters["normalise"])

        # Time grid for each component
        t = np.arange(self.age_main)
        t_burst = np.arange(self.age_burst)

        # SFR for each component
        self.sfr = t * np.exp(-t / self.tau_main) / self.tau_main**2
        sfr_burst = np.exp(-t_burst / self.tau_burst)

        # Height of the late burst to have the desired produced mass fraction
        sfr_burst *= (self.f_burst / (1. - self.f_burst) * np.sum(self.sfr) /
                      np.sum(sfr_burst))

        # We add the age burst exponential for ages superior to age_main -
        # age_burst
        self.sfr[-(t_burst[-1] + 1):] += sfr_burst

        # Compute the integral of the SFH and normalise it to 1 solar mass
        # if asked to.
        self.sfr_integrated = np.sum(self.sfr) * 1e6
        if normalise:
            self.sfr /= self.sfr_integrated
            self.sfr_integrated = 1.
        else:
            self.sfr *= sfr_A
            self.sfr_integrated *= sfr_A

    def process(self, sed):
        """
        Parameters
        ----------
        sed : pcigale.sed.SED object

        """

        sed.add_module(self.name, self.parameters)

        # Add the sfh and the output parameters to the SED.
        sed.sfh = self.sfr
        sed.add_info("sfh.integrated", self.sfr_integrated, True,
                     unit='solMass')
        sed.add_info("sfh.age_main", self.age_main, unit='Myr')
        sed.add_info("sfh.tau_main", self.tau_main, unit='Myr')
        sed.add_info("sfh.age_burst", self.age_burst, unit='Myr')
        sed.add_info("sfh.tau_burst", self.tau_burst, unit='Myr')
        sed.add_info("sfh.f_burst", self.f_burst)

# CreationModule to be returned by get_module
Module = SFHDelayed
