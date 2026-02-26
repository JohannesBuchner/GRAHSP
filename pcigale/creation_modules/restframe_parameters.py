"""
Module that estimates other parameters, e.g., UV slope, Lick indices, etc.
==========================================================================

This module estimates additional parameters of interest close to the
observation, e.g., the ultraviolet slope (beta), the rest-frame, any type of
indices (Lick), etc.

This module has to be called right before redshifting as such it does not take
into account the IGM absorption so it should not be used to compute fluxes in
the observed frame. For that use the param_postz module.

"""

from itertools import chain

import numpy as np
from scipy.constants import c, parsec

from . import CreationModule
from ..sed.utils import flux_trapz

__category__ = "restframe_parameters"


class RestframeParam(CreationModule):
    """Compute miscellaneous parameters on the full SED. This is a separate
    module as computing these quantitites in other SED modules does not always
    make much sense. This module is to be called right before the redshifting
    module. This means it does not include IGM absorbtion

    """

    parameter_list = {
        "beta_calz94": (
            "bool",
            "Observed and intrinsic UV slopes β and β₀ measured in the same "
            "way as in Calzetti et al. (1994).",
            False
        ),
        "D4000": (
            "bool",
            "D4000 break using the Balogh et al. (1999) definition.",
            False
        ),
        "IRX": (
            "bool",
            "IRX computed from the GALEX FUV filter and the dust luminosity.",
            False
        ),
        "EW_lines": (
            "string",
            "Central wavelength of the nebular emission lines for which to compute "
            "the equivalent width. The half-bandwidth must be indicated "
            "after the '/' sign. For instance 656.3/1.0 means oth the nebular "
            "line and the continuum are integrated over 655.3-657.3 nm.",
            "500.7/1.0 & 656.3/1.0"
        ),
        "EW": (
            "string",
            "Name, central wavelength of the emission for which to compute "
            "the equivalent width. The lower and upper wavelength of the "
            "wavelength window are after the '/' sign. "
            "For instance K/393.4777/391/395 means the Fraunhofer K line "
            "and the continuum is integrated over 691-395 nm.",
            "K/393.4777/391/395"
        ),
        "luminosity_filters": (
            "string",
            "Filters for which the rest-frame luminosity will be computed. "
            "You can give several filter names separated by a & (don't use "
            "commas).",
            "FUV & V_B90"
        ),
        "colours_filters": (
            "string",
            "Rest-frame colours to be computed. You can give several colours "
            "separated by a & (don't use commas).",
            "FUV-NUV & NUV-r_prime"
        )
    }

    @staticmethod
    def _slope(x, y):
        """Computes the slope of y vs x from the covariance matrix.
        """
        X = np.stack((x, y), axis=0)
        X -= np.average(X, axis=1)[:, None]
        ssxm, ssxym, _, _ = np.dot(X, X.T).flat

        return ssxym / ssxm

    def calz94(self, sed):
        wl = sed.wavelength_grid

        # Attenuated (observed) UV slopes beta as defined in Calzetti et al.
        # (1994, ApJ 429, 582, Tab. 2) that excludes the 217.5 nm bump
        # wavelength range and other spectral features
        if 'nebular.lines_width' in sed.info:
            key = (wl.size, sed.info['nebular.lines_width'])
        else:
            key = (wl.size, )

        if key in self.w_calz94:
            w_calz94 = self.w_calz94[key]
        else:
            calz_wl = [(126.8, 128.4), (130.9, 131.6), (134.2, 137.1),
                       (140.7, 151.5), (156.2, 158.3), (167.7, 174.0),
                       (176.0, 183.3), (186.6, 189.0), (193.0, 195.0),
                       (240.0, 258.0)]
            w_calz94 = np.where(np.any([(wl >= wlseg[0]) & (wl <= wlseg[1])
                                        for wlseg in calz_wl], axis=0))

            self.w_calz94[key] = w_calz94

        lgwl = np.log10(wl[w_calz94])
        lglumin = np.log10(sed.luminosity[w_calz94])
        lglumin0 = np.log10(np.sum([sed.luminosities[i][w_calz94]
                                    for i, name in enumerate(sed.contribution_names)
                                    if 'attenuation' not in name], axis=0))

        return (self._slope(lgwl, lglumin), self._slope(lgwl, lglumin0))

    def D4000(self, sed):
        wl = sed.wavelength_grid
        fnu = sed.fnu

        # Strength of the D_4000 break using Balogh et al. (1999, ApJ 527, 54),
        # i.e., ratio of the flux in the red continuum to that in the blue
        # continuum: Blue continuum: 385.0-395.0 nm & red continuum:
        # 400.0-410.0 nm.
        if 'nebular.lines_width' in sed.info:
            key = (wl.size, sed.info['nebular.lines_width'])
        else:
            key = (wl.size, )

        if key in self.w_D4000blue:
            w_D4000blue = self.w_D4000blue[key]
            w_D4000red = self.w_D4000red[key]
        else:
            w_D4000blue = np.where((wl >= 385.0) & (wl <= 395.0))
            w_D4000red = np.where((wl >= 400.0) & (wl <= 410.0))
            self.w_D4000blue[key] = w_D4000blue
            self.w_D4000red[key] = w_D4000red

        return (np.trapezoid(fnu[w_D4000red], x=wl[w_D4000red]) /
                np.trapezoid(fnu[w_D4000blue], x=wl[w_D4000blue]))

    def EW(self, sed):
        wl = sed.wavelength_grid

        key = (wl.size, sed.info['nebular.lines_width'])
        if key in self.w_lines:
            w_lines = self.w_lines[key]
        else:
            w_lines = {line: np.where((wl >= line[0] - line[1]) &
                                      (wl <= line[0] + line[1]))
                       for line in self.lines}
            self.w_lines[key] = w_lines

        lumin_line = np.sum([sed.luminosities[k]
                             for k, name in enumerate(sed.contribution_names)
                             if 'nebular.lines' in name], axis=0)
        lumin_cont = sed.luminosity - lumin_line

        EW = {}
        for line in self.lines:
            w_line = w_lines[line]
            wl_line = wl[w_line]
            key = (wl_line.size, sed.info['nebular.lines_width'], line[0], 0.)
            EW[line] = (flux_trapz(lumin_line[w_line], wl_line, key) /
                        flux_trapz(lumin_cont[w_line], wl_line, key) *
                        (wl_line[-1] - wl_line[0]))

        return EW

    def EW_empirical(self, sed):
        for linename, ctr, lo, hi in self.empirical_lines:
            window, = np.where(np.logical_and(sed.wavelength_grid >= lo, sed.wavelength_grid <= hi))
            x = sed.wavelength_grid[window]
            y = sed.luminosity[window]
            # guess continuum at the center
            y_mid = (y[0] + y[-1]) / 2
            area = np.trapezoid(y=y - y_mid, x=x)
            if y_mid == 0:
                EW = 0
            else:
                EW = area / y_mid
            assert np.isfinite(EW), (area, y_mid, y[0], y[-1], EW)
            yield linename, EW

    def _init_code(self):
        self.parameters["beta_calz94"] = bool(self.parameters["beta_calz94"].lower() == "true")
        self.parameters["D4000"] = bool(self.parameters["D4000"].lower() == "true")
        self.parameters["IRX"] = bool(self.parameters["IRX"].lower() == "true")
        # Index of the wavelengths of interest. We use a dictionary with the
        # size of the wavelength array as a key to take into account that
        # different models may have a different sampling, for instance when
        # an AGN may be present or not.
        self.w_calz94 = {}
        self.w_D4000blue = {}
        self.w_D4000red = {}
        self.w_lines = {}

        # Extract the list of lines to compute the equivalent width
        self.lines = [item.strip() for item in
                      self.parameters["EW_lines"].split("&")
                      if item.strip() != '']
        self.lines = [(float(line.split('/')[0]),
                      float(line.split('/')[1]))
                      for line in self.lines]
        self.empirical_lines = [item.strip().split('/')
            for item in self.parameters["EW"].split("&") if item.strip() != '']
        self.empirical_lines = [(k, float(ctr), float(lo), float(hi)) 
            for k, ctr, lo, hi in self.empirical_lines]

        # Extract the list of filters to compute the luminosities
        self.lumin_filters = [item.strip() for item in
                              self.parameters["luminosity_filters"].split("&")
                              if item.strip() != '']

        # Extract the list of rest-frame colours
        self.colours = [item.strip().split("-") for item in
                        self.parameters["colours_filters"].split("&")
                        if item.strip() != '']

        # Extract the list of rest-frame colour filters
        self.colour_filters = list(set(chain(*self.colours)))

        # Compute the list of unique filters to compute both the luminosities
        # and the colours
        self.filters = list(set(self.lumin_filters + self.colour_filters +
                                (["FUV"] if self.parameters["IRX"]
                                 else [])))

        # Conversion factor to go from Fnu in mJy to Lnu in W
        self.to_lumin = 1e-29 * 4. * np.pi * (10. * parsec)**2

    def process(self, sed):
        """Computes the parameters for each model.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        fluxes = {filt: sed.compute_fnu(filt) for filt in self.filters}

        if self.parameters['beta_calz94']:
            beta, beta0 = self.calz94(sed)
            sed.add_info("param.beta_calz94", beta)
            sed.add_info("param.beta0_calz94", beta0)
        if self.parameters['D4000']:
            sed.add_info("param.D_4000", self.D4000(sed))
        if self.parameters['IRX']:
            sed.add_info("param.IRX", np.log10(sed.info['dust.luminosity'] /
                         (fluxes['FUV'] * self.to_lumin * c / 154e-9)))

        if self.lines and 'nebular.lines_young' in sed.contribution_names:
            for line, EW in self.EW(sed).items():
                sed.add_info(f"param.EW({line[0]}/{line[1]})", EW, unit='nm')

        if self.empirical_lines:
            for line, EW in self.EW_empirical(sed):
                sed.add_info("param.EW_%s" % line, EW, unit='nm')

        with np.errstate(invalid='ignore', divide='ignore'):
            for filt in self.lumin_filters:
                sed.add_info(f"param.restframe_Lnu({filt})",
                             fluxes[filt] * self.to_lumin, True, unit='W/Hz')
            for filt1, filt2 in self.colours:
                sed.add_info(f"param.restframe_{filt1}-{filt2}",
                             2.5 * np.log10(fluxes[filt2] / fluxes[filt1]),
                             unit='mag')


# SedModule to be returned by get_module
Module = RestframeParam
