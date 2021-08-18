# -*- coding: utf-8 -*-
# Copyright (C) 2012 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

import numpy as np


class Filter(object):
    """A photometric filter with associated transmission data.
    """

    def __init__(self, name, description=None, trans_type=None,
                 trans_table=None, effective_wavelength=None):
        """Create a new filter. If the transmission type, the description
        the transmission table or the effective wavelength are not specified,
        their value is set to None.

        Parameters
        ----------
        name: string
            Name of the filter
        description: string
            Description of the filter
        trans_type: string
            Type of transmission table ('energy' or 'photon')
        trans_table: array
            trans_table[0] is the wavelength in nm,
            trans_table[1] is the transmission)
        effective_wavelength: float
            Effective wavelength of the filter
        """

        self.name = name
        self.description = description
        self.trans_type = trans_type
        self.trans_table = trans_table
        self.effective_wavelength = effective_wavelength

    # Check that the trans_type is correct
    @property
    def trans_type(self):
        return self._trans_type

    @trans_type.setter
    def trans_type(self, value):
        if value in ['energy', 'photon']:
            self._trans_type = value
        else:
            raise ValueError("Filter transmission type can only be "
                             "'energy' or 'photon'.")

    def __str__(self):
        """Pretty print the filter information
        """
        result = ""
        result += ("Filter name: %s\n" % self.name)
        result += ("Description: %s\n" % self.description)
        result += ("Transmission type: %s\n" % self.trans_type)
        result += ("Effective wavelength: %s nm\n" %
                   self.effective_wavelength)
        return result

    def normalise(self):
        """
        Normalise the transmission table to 1 and compute the effective
        wavelength of the filter.
        """
        # normalise transmission to sum to unity
        self.trans_table[1] = self.trans_table[1] / (
            np.trapz(self.trans_table[1], self.trans_table[0]))

        # calculation of the effective wavelength
        # actually, cigale computes the mean wavelength here, and wrongly
        # calls it effective wavelength
        self.effective_wavelength = np.trapz(self.trans_table[1] *
                                             self.trans_table[0],
                                             self.trans_table[0])
        
        # correctly, we need the pivot wavelength for conversion
        # equ. A16 in Bessel & Murphy 2012
        S   = self.trans_table[1]
        lam = self.trans_table[0]
        nominator   = np.trapz(lam*S, x=lam)
        denominator = np.trapz(S/lam, x=lam)
        pivot_wavelength = (nominator / denominator)**0.5
        # dump out information
        import json
        try:
            data = json.load(open('filterinfo.json'))
        except:
            data = {}
        data[str(self.name)] = dict(mean_wavelength=float(self.effective_wavelength),
            pivot_wavelength=float(pivot_wavelength),
            error = float((self.effective_wavelength / pivot_wavelength)**2 ))
        json.dump(data, open('filterinfo.json', 'w'))
        
        # we re-use the same field in cigale, but it is actually the 
        # pivot wavelength
        self.effective_wavelength = pivot_wavelength
        
