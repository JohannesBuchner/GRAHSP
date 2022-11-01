# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

# NetzerDisk, MorNetzer2012Torus, FeIIferland, MorNetzerEmLines

class AttenuationLaw(object):
    """Attenuation law model """

    def __init__(self, name, wave, k):
        """Create a new extinction law"""

        self.name = name
        self.wave = wave
        self.k = k
