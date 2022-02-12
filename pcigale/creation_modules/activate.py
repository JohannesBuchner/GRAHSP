# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN module
==================================================

This module combines disk and torus.

"""
from collections import OrderedDict
import numpy as np
from pcigale.data import Database
from . import CreationModule
import scipy.constants as cst

class Activate(CreationModule):
    """Activate AGN dust torus emission

    Combination of
    
    * Disk emission (Netzer)
    * Torus emission (Mor & Netzer 2012)
    
    Use ActivateLines to add emission lines
    """

    parameter_list = OrderedDict([
        ('M', (
            'float',
            "BH mass"
            "Possible values are: 6.0, 8.0, 9.0",
            6.0
        )),
        ('a', (
            'float',
            "Spin parameter. "
            "Possible values are: 0, 0.998",
            0.0
        )),
        ('Mdot', (
            'float',
            "Eddington accretion rate. "
            "Possible values are: 0.03, 0.3",
            0.3
        )),
        ('inc', (
            'float',
            "Inclination. "
            "Possible values are: 0",
            0.
        )),
        ('fcov', (
            'float',
            "Torus Covering fraction. 0.1 to 0.7 is recommended.",
            0.1
        )),
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
        ('fracAGN', (
            'float',
            "AGN fraction at 510nm.",
            0.1
        )),
        ('Si', (
            'float',
            "Strength of the 12um Silicate feature (relative to the difference in Mullaney+11). -3 to 3 is reasonable.",
            0.
        )),
        ('TORtemp', (
            'float',
            "Steepness of the torus spectrum (relative to the spread in Mor&Netzer+09). -3 to 3 is reasonable.",
            0.
        ))
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        M = self.parameters["M"]
        a = self.parameters["a"]
        Mdot = self.parameters["Mdot"]
        inc = self.parameters["inc"]
        # agnType = self.parameters["AGNtype"]
        # for LINERs, we still use a type-2 template

        with Database() as base:
            self.disk = base.get_ActivateNetzerDisk(M, a, Mdot, inc)
            assert (self.disk.lumin >= 0).all()
            # load average torus model spectrum
            self.torus_avg = base.get_ActivateMorNetzer2012Torus('mor-avg')
            self.torus_lo = base.get_ActivateMorNetzer2012Torus('mor-lo')
            self.torus_hi = base.get_ActivateMorNetzer2012Torus('mor-hi')
            assert (self.torus_avg.lumin >= 0).all()
            # load silicate feature model spectrum
            self.si = base.get_ActivateMorNetzer2012Torus('mullaney-silicate')
            self.fe2 = base.get_ActivateFeIIferland()
            assert self.fe2.wave.shape == self.fe2.lumin.shape, (self.fe2.wave.shape, self.fe2.lumin.shape)
            assert (self.fe2.lumin >= 0).all()
            self.emLines = base.get_ActivateMorNetzerEmLines()
            assert (self.emLines.lumin_BLAGN >= 0).all()
            assert (self.emLines.lumin_Sy2 >= 0).all()
            assert (self.emLines.lumin_LINER >= 0).all()
            

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        fracAGN = self.parameters["fracAGN"]
        fcov = self.parameters["fcov"]
        # agnType = self.parameters["AGNtype"]
        Si = self.parameters["Si"]
        TORtemp = self.parameters["TORtemp"]
        
        #print("activate.processing with parameters:", self.parameters)
        # get existing normalisation at 5100A
        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.M', self.parameters["M"])
        sed.add_info('agn.a', self.parameters["a"])
        sed.add_info('agn.Mdot', self.parameters["Mdot"])
        sed.add_info('agn.inc', self.parameters["inc"])
        sed.add_info('agn.fcov', self.parameters["fcov"])
        sed.add_info('agn.type', self.parameters["AGNtype"])
        sed.add_info('agn.fracAGN', self.parameters["fracAGN"])
        sed.add_info('agn.Si', self.parameters["Si"])
        sed.add_info('agn.TORtemp', self.parameters["TORtemp"])

        
        # Compute the AGN luminosity
        if fracAGN < 0.:
            l_agn = 1.
        elif fracAGN < 1.:
            luminosity = np.interp(510.0, sed.wavelength_grid, sed.luminosity)
            assert luminosity >= 0, luminosity
            l_agn = luminosity * (1./(1.-fracAGN) - 1.)
        else:
            raise Exception("AGN fraction is exactly 1. Behaviour "
                            "undefined.")
        assert l_agn >= 0, l_agn
        
        sed.add_info('agn.lum5100A', l_agn, True)
        #if agnType != 1:
        #	# truncate disk so that it does not produce UV emission
        #	# if type 2
        #	disk = np.where(self.disk.wave < 500, 0, self.disk.lumin)
        #else:
        disk = self.disk.lumin
        
        # Add disk
        sed.add_contribution('agn.activate_Disk', self.disk.wave,
                             l_agn * disk)
        #print(' disk', self.disk.wave, l_agn, self.disk.lumin)
        
        # Add torus for NIR-MIR continuum
        # formula of Netzer (readme)
        # l_torus * 12um = 2.5 * l_agn * 510nm * fcov
        # l_agn is defined at 510nm, l_torus at 12um
        # because both are nu*L_nu = lam*L_lam normalisations, we need a
        l_torus = 2.5 * l_agn * fcov / 12.0 * 0.510
        sed.add_info('agn.lum12um', l_torus, True)
        if TORtemp > 0:
            torus_deviation = (self.torus_hi.lumin - self.torus_avg.lumin) * TORtemp
        else:
            torus_deviation = (self.torus_lo.lumin - self.torus_avg.lumin) * -TORtemp

        sed.add_contribution('agn.activate_Torus', self.torus_avg.wave,
                             l_torus * (self.torus_avg.lumin + torus_deviation))
        sed.add_contribution('agn.activate_TorusSi', self.si.wave,
                             l_torus * self.si.lumin * Si)

# CreationModule to be returned by get_module
Module = Activate
