# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Johannes Buchner

"""
Activate AGN gtorus module
==================================================

This module adds a empirical "torus" mid-infrared emission based
on two gaussian and a Si feature.

"""
from collections import OrderedDict
import numpy as np
from . import CreationModule
from numpy import exp


class ActivateGTorus(CreationModule):
    """Activate Gaussian mixture torus."""

    parameter_list = OrderedDict([
        ('fcov', (
            'float',
            "Covering fraction of total at 12um, relative to disk at 510nm. 0.1 to 0.7 is recommended.",
            0.1
        )),
        ('Si', (
            'float',
            "Strength of the 12um Silicate feature (relative to the difference in Mullaney+11). -3 to 3 is reasonable.",
            0.
        )),
        ('COOLlam', (
            'float',
            "Wavelength of peak of cold dust component in um. 15-20 is reasonable.",
            17.0
        )),
        ('COOLwidth', (
            'float',
            "Standard deviation of Log-Gaussian cold dust component, in dex. 0.3-0.6 is reasonable.",
            0.45
        )),
        ('HOTlam', (
            'float',
            "Wavelength of peak of hot dust component in um. 1-4 is reasonable.",
            2.0
        )),
        ('HOTwidth', (
            'float',
            "Standard deviation of Log-Gaussian hot dust component, in dex. 0.3-0.6 is reasonable.",
            0.5
        )),
        ('HOTfcov', (
            'float',
            "Covering factor of the hot dust component. Ratio of peak to peak of cold component in lambdaLlambda.",
            0.0
        )),
        ('SiRatio', (
            'float',
            "Si absorption to emission ratio.",
            0.29
        )),
        ('SiEmlam', (
            'float',
            "Wavelength of Si emission feature in nm.",
            9841
        )),
        ('SiAbslam', (
            'float',
            "Wavelength of Si absorption feature in nm.",
            14224
        )),
        ('SiEmWidth', (
            'float',
            "Width of Si absorption feature in nm.",
            1025.3
        )),
        ('SiAbsWidth', (
            'float',
            "Width of Si absorption feature in nm.",
            1163.5
        )),
    ])
    
    def _init_code(self):
        # compute only from 0.3 um to 100 um for efficiency
        #self.wave = 10**self.log_wave * 1000 # in nm
        # use same grid as Dale models, this makes add_contribution more efficient
        # (but with 12um injected)
        self.wave = 1000 * np.array([0.360, 0.450, 0.580, 0.750, 1.000, 1.009, 1.019, 1.028, 1.038, 1.047, 1.057, 1.067, 1.076, 1.086, 1.096, 1.107, 1.117, 1.127, 1.138, 1.148, 1.159, 1.169, 1.180, 1.191, 1.202, 1.213, 1.225, 1.236, 1.247, 1.259, 1.271, 1.282, 1.294, 1.306, 1.318, 1.330, 1.343, 1.355, 1.368, 1.380, 1.393, 1.406, 1.419, 1.432, 1.445, 1.459, 1.472, 1.486, 1.500, 1.514, 1.528, 1.542, 1.556, 1.570, 1.585, 1.600, 1.614, 1.629, 1.644, 1.660, 1.675, 1.690, 1.706, 1.722, 1.738, 1.754, 1.770, 1.786, 1.803, 1.820, 1.837, 1.854, 1.871, 1.888, 1.905, 1.923, 1.941, 1.959, 1.977, 1.995, 2.014, 2.032, 2.051, 2.070, 2.089, 2.109, 2.128, 2.148, 2.168, 2.188, 2.208, 2.228, 2.249, 2.270, 2.291, 2.312, 2.333, 2.355, 2.377, 2.399, 2.421, 2.443, 2.466, 2.489, 2.512, 2.535, 2.559, 2.582, 2.606, 2.630, 2.655, 2.679, 2.704, 2.729, 2.754, 2.780, 2.805, 2.831, 2.858, 2.884, 2.911, 2.938, 2.965, 2.992, 3.020, 3.048, 3.076, 3.105, 3.133, 3.162, 3.192, 3.221, 3.251, 3.281, 3.311, 3.342, 3.373, 3.404, 3.436, 3.467, 3.499, 3.532, 3.565, 3.597, 3.631, 3.664, 3.698, 3.733, 3.767, 3.802, 3.837, 3.873, 3.908, 3.945, 3.981, 4.018, 4.055, 4.093, 4.130, 4.169, 4.207, 4.246, 4.285, 4.325, 4.365, 4.406, 4.446, 4.487, 4.529, 4.571, 4.613, 4.656, 4.699, 4.742, 4.786, 4.831, 4.875, 4.920, 4.966, 5.012, 5.058, 5.105, 5.152, 5.200, 5.248, 5.297, 5.346, 5.395, 5.445, 5.495, 5.546, 5.598, 5.649, 5.702, 5.754, 5.808, 5.861, 5.916, 5.970, 6.026, 6.081, 6.138, 6.194, 6.252, 6.310, 6.368, 6.427, 6.486, 6.546, 6.607, 6.668, 6.730, 6.792, 6.855, 6.918, 6.982, 7.047, 7.112, 7.178, 7.244, 7.311, 7.379, 7.447, 7.516, 7.586, 7.656, 7.727, 7.798, 7.870, 7.943, 8.017, 8.091, 8.166, 8.241, 8.318, 8.395, 8.472, 8.551, 8.630, 8.710, 8.790, 8.872, 8.954, 9.036, 9.120, 9.204, 9.290, 9.376, 9.462, 9.550, 9.638, 9.727, 9.817, 9.908, 10.000, 10.090, 10.190, 10.280, 10.380, 10.470, 10.570, 10.670, 10.760, 10.860, 10.960, 11.070, 11.170, 11.270, 11.380, 11.480, 11.590, 11.690, 11.800, 11.910, 12.000, 12.020, 12.130, 12.250, 12.360, 12.470, 12.590, 12.710, 12.820, 12.940, 13.060, 13.180, 13.300, 13.430, 13.550, 13.680, 13.800, 13.930, 14.060, 14.190, 14.320, 14.450, 14.590, 14.720, 14.860, 15.000, 15.140, 15.280, 15.420, 15.560, 15.700, 15.850, 16.000, 16.140, 16.290, 16.440, 16.600, 16.750, 16.900, 17.060, 17.220, 17.380, 17.540, 17.700, 17.860, 18.030, 18.200, 18.370, 18.540, 18.710, 18.880, 19.050, 19.230, 19.410, 19.590, 19.770, 19.950, 20.140, 20.320, 20.510, 20.700, 20.890, 21.090, 21.280, 21.480, 21.680, 21.880, 22.080, 22.280, 22.490, 22.700, 22.910, 23.120, 23.330, 23.550, 23.770, 23.990, 24.210, 24.430, 24.660, 24.890, 25.120, 25.350, 25.590, 25.820, 26.060, 26.300, 26.550, 26.790, 27.040, 27.290, 27.540, 27.800, 28.050, 28.310, 28.580, 28.840, 29.110, 29.380, 29.650, 29.920, 30.200, 30.480, 30.760, 31.050, 31.330, 31.620, 31.920, 32.210, 32.510, 32.810, 33.110, 33.420, 33.730, 34.040, 34.360, 34.670, 34.990, 35.320, 35.650, 35.970, 36.310, 36.640, 36.980, 37.330, 37.670, 38.020, 38.370, 38.730, 39.080, 39.450, 39.810, 40.180, 40.550, 40.930, 41.300, 41.690, 42.070, 42.460, 42.850, 43.250, 43.650, 44.060, 44.460, 44.870, 45.290, 45.710, 46.130, 46.560, 46.990, 47.420, 47.860, 48.310, 48.750, 49.200, 49.660, 50.120, 50.580, 51.050, 51.520, 52.000, 52.480, 52.970, 53.460, 53.950, 54.450, 54.950, 55.460, 55.980, 56.490, 57.020, 57.540, 58.080, 58.610, 59.160, 59.700, 60.260, 60.810, 61.380, 61.940, 62.520, 63.100, 63.680, 64.270, 64.860, 65.460, 66.070, 66.680, 67.300, 67.920, 68.550, 69.180, 69.820, 70.470, 71.120, 71.780, 72.440, 73.110, 73.790, 74.470, 75.160, 75.860, 76.560, 77.270, 77.980, 78.700, 79.430, 80.170, 80.910, 81.660, 82.410, 83.180, 83.950, 84.720, 85.510, 86.300, 87.100, 87.900, 88.720, 89.540, 90.360, 91.200, 92.040, 92.900, 93.760, 94.620, 95.500, 96.380, 97.270, 98.170, 99.080, 100.000])
        self.log_wave = np.log10(self.wave / 1000)
        self.norm_index = np.argmin(np.abs(10**self.log_wave - 12))

    def process(self, sed):
        """Add the AGN contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        """

        fcov = self.parameters["fcov"]
        logCOOLlam = np.log10(self.parameters["COOLlam"])
        COOLlam = self.parameters["COOLlam"]
        COOLwidth = self.parameters["COOLwidth"]
        HOTfcov = self.parameters["HOTfcov"]
        logHOTlam = np.log10(self.parameters["HOTlam"])
        HOTlam = self.parameters["HOTlam"]
        HOTwidth = self.parameters["HOTwidth"]

        Si = self.parameters["Si"]
        SiEmAmpl = 0.4
        SiEmlam = self.parameters["SiEmlam"]
        SiEmWidth = self.parameters["SiEmWidth"]
        SiAbsAmpl = SiEmAmpl * self.parameters["SiRatio"]
        SiAbslam = self.parameters["SiAbslam"]
        SiAbsWidth = self.parameters["SiAbsWidth"]

        l_agn = sed.info['agn.lum5100A']

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.fcov', fcov)
        sed.add_info('agn.Si', Si)
        sed.add_info('agn.COOLlam', COOLlam, unit='um')
        sed.add_info('agn.COOLwidth', COOLwidth)
        sed.add_info('agn.HOTfcov', HOTfcov)
        sed.add_info('agn.HOTwidth', HOTwidth)
        sed.add_info('agn.HOTlam', HOTlam, unit='um')

        # Add torus for NIR-MIR continuum
        # formula of Netzer (readme)
        # l_torus * 12um = 2.5 * l_agn * 510nm * fcov
        # l_agn is defined at 510nm, l_torus at 12um
        # because both are nu*L_nu = lam*L_lam normalisations, we need a
        l_torus = 2.5 * l_agn * fcov / 12.0 * 0.510
        sed.add_info('agn.lum12um', l_torus, True, unit='W/nm')

        cool_spectrum = exp(-((self.log_wave - logCOOLlam) / COOLwidth)**2)
        
        hot_spectrum = HOTfcov * 10**(logCOOLlam - logHOTlam) * exp(-((self.log_wave - logHOTlam) / HOTwidth)**2)
        total_spectrum = cool_spectrum + hot_spectrum
        # apply normalisation at 12 um:
        torus_spectrum = l_torus * total_spectrum / total_spectrum[self.norm_index]
        sed.add_contribution('agn.activate_Torus', self.wave, torus_spectrum)

        si_spectrum = l_torus * Si * (
            SiEmAmpl * exp(-0.5 * ((sed.wavelength_grid - SiEmlam) / SiEmWidth)**2) - 
            SiAbsAmpl * exp(-0.5 * ((sed.wavelength_grid - SiAbslam) / SiAbsWidth)**2))
        # avoid negative fluxes, truncate to zero flux if the sum would go below zero
        mask_negative = si_spectrum < -sed.luminosities[-1]
        si_spectrum[mask_negative] = -sed.luminosities[-1,mask_negative]
        sed.add_contribution('agn.activate_Torus_Si', sed.wavelength_grid, si_spectrum)
        l_torus_6um = np.interp(6000., self.wave, torus_spectrum)
        sed.add_info('agn.lum6um', l_torus_6um, True, unit='W/nm')


# CreationModule to be returned by get_module
Module = ActivateGTorus
