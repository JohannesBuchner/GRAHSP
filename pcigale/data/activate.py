# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

# NetzerDisk, MorNetzer2012Torus, FeIIferland, MorNetzerEmLines

class NetzerDisk(object):
    """AGN disk emission model.
    """

    def __init__(self, M, a, Mdot, inc, wave, lumin):
        """Create a new disk model

        Parameters
        ----------
        M: float
            Mass (log, in units of Msun)
        a: float
            Spin parameter
        Mdot: float
            Eddington accretion rate
        inc: float
            Inclination
        """

        self.M = M
        self.a = a
        self.Mdot = Mdot
        self.inc = inc
        self.wave = wave
        self.lumin = lumin


class Pacifici2012Gal(object):
    """Galaxy Stars+Gas emission model."""

    def __init__(self, name, wave, lumin):
        """Create a new galaxy model"""

        self.name = name
        self.wave = wave
        self.lumin = lumin


class MorNetzer2012Torus(object):
    """AGN Torus emission model."""

    def __init__(self, wave, lumin):
        """Create a new torus model"""

        self.wave = wave
        self.lumin = lumin


class FeIIferland(object):
    """FeII forest model."""

    def __init__(self, wave, lumin):
        """Create a new torus model"""

        self.wave = wave
        self.lumin = lumin


class MorNetzerEmLines(object):
    """Model of the AGN emission line strengths."""

    def __init__(self, wave, lumin_BLAGN, lumin_Sy2, lumin_LINER):
        """Create a new torus model"""

        self.wave = wave
        self.lumin_BLAGN = lumin_BLAGN
        self.lumin_Sy2 = lumin_Sy2
        self.lumin_LINER = lumin_LINER





