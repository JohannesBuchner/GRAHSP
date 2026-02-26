# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Authors: Yannick Roehlly, Médéric Boquien, Laure Ciesla

"""
This script is used the build pcigale internal database containing:
- The various filter transmission tables;
- The Maraston 2005 single stellar population (SSP) data;
- The Dale and Helou 2002 infra-red templates.

"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import glob
import io
import itertools
import numpy as np
from scipy import interpolate
import scipy.constants as cst
from astropy.table import Table
from pcigale.data import (Database, Filter, M2005, BC03, Fritz2006,
                          Dale2014, DL2007, DL2014, NebularLines,
                          NebularContinuum, 
                          NetzerDisk, Pacifici2012Gal, MorNetzer2012Torus, FeII, MorNetzerEmLines,
                          AttenuationLaw)


def read_bc03_ssp(filename):
    """Read a Bruzual and Charlot 2003 ASCII SSP file

    The ASCII SSP files of Bruzual and Charlot 2003 have se special structure.
    A vector is stored with the number of values followed by the values
    separated by a space (or a carriage return). There are the time vector, 5
    (for Chabrier IMF) or 6 lines (for Salpeter IMF) that we don't care of,
    then the wavelength vector, then the luminosity vectors, each followed by
    a 52 value table, then a bunch of other table of information that are also
    in the *colors files.

    Parameters
    ----------
    filename : string

    Returns
    -------
    time_grid: numpy 1D array of floats
              Vector of the time grid of the SSP in Myr.
    wavelength: numpy 1D array of floats
                Vector of the wavelength grid of the SSP in nm.
    spectra: numpy 2D array of floats
             Array containing the SSP spectra, first axis is the wavelength,
             second one is the time.

    """

    def file_structure_generator():
        """Generator used to identify table lines in the SSP file

        In the SSP file, the vectors are store one next to the other, but
        there are 5 informational lines after the time vector. We use this
        generator to the if we are on lines to read or not.
        """
        if "chab" in filename:
            bad_line_number = 5
        else:
            bad_line_number = 6
        yield("data")
        for i in range(bad_line_number):
            yield("bad")
        while True:
            yield("data")

    file_structure = file_structure_generator()
    # Are we in a data line or a bad one.
    what_line = next(file_structure)
    # Variable conting, in reverse order, the number of value still to
    # read for the read vector.
    counter = 0

    time_grid = []
    full_table = []
    tmp_table = []

    with open(filename) as file_:
        # We read the file line by line.
        for line in file_:
            if what_line == "data":
                # If we are in a "data" line, we analyse each number.
                for item in line.split():
                    if counter == 0:
                        # If counter is 0, then we are not reading a vector
                        # and the first number is the length of the next
                        # vector.
                        counter = int(item)
                    else:
                        # If counter > 0, we are currently reading a vector.
                        tmp_table.append(float(item))
                        counter -= 1
                        if counter == 0:
                            # We reached the end of the vector. If we have not
                            # yet store the time grid (the first table) we are
                            # currently reading it.
                            if time_grid == []:
                                time_grid = tmp_table[:]
                            # Else, we store the vector in the full table,
                            # only if its length is superior to 250 to get rid
                            # of the 52 item unknown vector and the 221 (time
                            # grid length) item vectors at the end of the
                            # file.
                            elif len(tmp_table) > 250:
                                full_table.append(tmp_table[:])

                            tmp_table = []

            # If at the end of a line, we have finished reading a vector, it's
            # time to change to the next structure context.
            if counter == 0:
                what_line = next(file_structure)

    # The time grid is in year, we want Myr.
    time_grid = np.array(time_grid, dtype=float)
    time_grid = time_grid * 1.e-6

    # The first "long" vector encountered is the wavelength grid. The value
    # are in Ångström, we convert it to nano-meter.
    wavelength = np.array(full_table.pop(0), dtype=float)
    wavelength = wavelength * 0.1

    # The luminosities are in Solar luminosity (3.826.10^33 ergs.s-1) per
    # Ångström, we convert it to W/nm.
    luminosity = np.array(full_table, dtype=float)
    luminosity = luminosity * 3.826e27
    # Transposition to have the time in the second axis.
    luminosity = luminosity.transpose()

    # In the SSP, the time grid begins at 0, but not in the *colors file, so
    # we remove t=0 from the SSP.
    return time_grid[1:], wavelength, luminosity[:, 1:]

from  pcigale.data import DatabaseInsertError

def build_filters(base):
    filters_dir = os.path.join(os.path.dirname(__file__), 'filters/')
    for filter_file in sorted(glob.glob(filters_dir + '*.dat')) + sorted(glob.glob(filters_dir + 'gazpar/**/*.pb', recursive=True)) + sorted(glob.glob(filters_dir + 'jwst/**/*.dat', recursive=True)):
        # filter_name2 = '.'.join(filter_file.replace(filters_dir, '').replace('gazpar/', '').split('/')[:-1])
        with open(filter_file, 'r') as filter_file_read:
            filter_name = filter_file_read.readline().strip('# \n\t')
            filter_type = filter_file_read.readline().strip('# \n\t')
            filter_description = filter_file_read.readline().strip('# \n\t')
        filter_table = np.genfromtxt(filter_file)
        # The table is transposed to have table[0] containing the wavelength
        # and table[1] containing the transmission.
        filter_table = filter_table.transpose()
        # We convert the wavelength from Å to nm.
        filter_table[0] *= 0.1

        print("Importing %s... (%s points)" % (filter_name,
                                               filter_table.shape[1]))

        new_filter = Filter(filter_name, filter_description,
                            filter_type, filter_table)

        # We normalise the filter and compute the effective wavelength.
        # If the filter is a pseudo-filter used to compute line fluxes, it
        # should not be normalised.
        if not filter_name.startswith('PSEUDO'):
            new_filter.normalise()
        else:
            new_filter.effective_wavelength = np.mean(
                filter_table[0][filter_table[1] > 0]
            )

        try:
            base.add_filter(new_filter)
        except DatabaseInsertError:
            print("WARNING: could not insert filter %s, already loaded." % filter_file)

def build_cosmos_filters(base):
    filters_dir = os.path.join(os.path.dirname(__file__), 'filters/cosmos/')
    for filter_file in sorted(glob.glob(filters_dir + '*.*')):
        with open(filter_file, 'r') as filter_file_read:
            filter_name = 'cosmos/' + os.path.basename(filter_file)
            filter_type = 'energy'
            if 'irac' in filter_name.lower():
                 filter_type = 'photon'
            filter_description = filter_file_read.readline().strip('# \n\t')
        filter_table = np.genfromtxt(filter_file)
        # The table is transposed to have table[0] containing the wavelength
        # and table[1] containing the transmission.
        filter_table = filter_table.transpose()
        # We convert the wavelength from Å to nm.
        filter_table[0] *= 0.1

        print("Importing %s... (%s points)" % (filter_name,
                                               filter_table.shape[1]))

        if 'irac' in filter_name.lower():
            # weigh by wavelength. Filter type 1 in LePhare
            # this is necessary because FIR filters are energy-weighted, not
            # photon-weighted.
            print("    reweighing by wavelength")
            lam_mean = np.mean(filter_table[0][filter_table[1] > 0])
            filter_table[1] = filter_table[1] * filter_table[0] / lam_mean
        
        new_filter = Filter(filter_name, filter_description,
                            filter_type, filter_table)

        # We normalise the filter and compute the effective wavelength.
        # If the filter is a pseudo-filter used to compute line fluxes, it
        # should not be normalised.
        if not filter_name.startswith('PSEUDO'):
            new_filter.normalise()
        else:
            new_filter.effective_wavelength = np.mean(
                filter_table[0][filter_table[1] > 0]
            )

        base.add_filter(new_filter)


def build_m2005(base, quick=False):
    m2005_dir = os.path.join(os.path.dirname(__file__), 'maraston2005/')

    # Age grid (1 Myr to 13.7 Gyr with 1 Myr step)
    time_grid = np.arange(1, 13701)
    fine_time_grid = np.linspace(0.1, 13700, 137000)

    # Transpose the table to have access to each value vector on the first
    # axis
    kroupa_mass = np.genfromtxt(m2005_dir + 'stellarmass.kroupa').transpose()
    salpeter_mass = \
        np.genfromtxt(m2005_dir + '/stellarmass.salpeter').transpose()

    for spec_file in glob.glob(m2005_dir + '*.rhb'):

        print("Importing %s..." % spec_file)

        spec_table = np.genfromtxt(spec_file).transpose()
        metallicity = spec_table[1, 0]
        if quick:
            # only use solar metallicity if quick
            if metallicity != 0:
                continue

        if 'krz' in spec_file:
            imf = 'krou'
            mass_table = np.copy(kroupa_mass)
        elif 'ssz' in spec_file:
            imf = 'salp'
            mass_table = np.copy(salpeter_mass)
        else:
            raise ValueError('Unknown IMF!!!')

        # Keep only the actual metallicity values in the mass table
        # we don't take the first column which contains metallicity.
        # We also eliminate the turn-off mas which makes no send for composite
        # populations.
        mass_table = mass_table[1:7, mass_table[0] == metallicity]

        # Regrid the SSP data to the evenly spaced time grid. In doing so we
        # assume 10 bursts every 0.1 Myr over a period of 1 Myr in order to
        # capture short evolutionary phases.
        # The time grid starts after 0.1 Myr, so we assume the value is the same
        # as the first actual time step.
        mass_table = interpolate.interp1d(mass_table[0] * 1e3, mass_table[1:],
                                          assume_sorted=True)(fine_time_grid)
        mass_table = np.mean(mass_table.reshape(5, -1, 10), axis=-1)

        # Extract the age and convert from Gyr to Myr
        ssp_time = np.unique(spec_table[0]) * 1e3
        spec_table = spec_table[1:]

        # Remove the metallicity column from the spec table
        spec_table = spec_table[1:]

        # Extract the wavelength and convert from Å to nm
        ssp_wave = spec_table[0][:1221] * 0.1
        spec_table = spec_table[1:]

        # Extra the fluxes and convert from erg/s/Å to W/nm
        ssp_lumin = spec_table[0].reshape(ssp_time.size, ssp_wave.size).T
        ssp_lumin *= 10 * 1e-7

        # We have to do the interpolation-averaging in several blocks as it is
        # a bit RAM intensive
        ssp_lumin_interp = np.empty((ssp_wave.size, time_grid.size))
        for i in range(0, ssp_wave.size, 100):
            fill_value = (ssp_lumin[i:i+100, 0], ssp_lumin[i:i+100, -1])
            ssp_interp = interpolate.interp1d(ssp_time, ssp_lumin[i:i+100, :],
                                              fill_value=fill_value,
                                              bounds_error=False,
                                              assume_sorted=True)(fine_time_grid)
            ssp_interp = ssp_interp.reshape(ssp_interp.shape[0], -1, 10)
            ssp_lumin_interp[i:i+100, :] = np.mean(ssp_interp, axis=-1)

        # To avoid the creation of waves when interpolating, we refine the grid
        # beyond 10 μm following a log scale in wavelength. The interpolation
        # is also done in log space as the spectrum is power-law-like
        ssp_wave_resamp = np.around(np.logspace(np.log10(10000),
                                                   np.log10(160000), 50))
        argmin = np.argmin(10000.-ssp_wave > 0)-1
        ssp_lumin_resamp = 10.**interpolate.interp1d(
                                    np.log10(ssp_wave[argmin:]),
                                    np.log10(ssp_lumin_interp[argmin:, :]),
                                    assume_sorted=True,
                                    axis=0)(np.log10(ssp_wave_resamp))

        ssp_wave = np.hstack([ssp_wave[:argmin+1], ssp_wave_resamp])
        ssp_lumin = np.vstack([ssp_lumin_interp[:argmin+1, :],
                               ssp_lumin_resamp])

        # Use Z value for metallicity, not log([Z/H])
        metallicity = {-1.35: 0.001,
                       -0.33: 0.01,
                       0.0: 0.02,
                       0.35: 0.04}[metallicity]

        base.add_m2005(M2005(imf, metallicity, time_grid, ssp_wave,
                             mass_table, ssp_lumin))


def build_bc2003(base, quick=False):
    bc03_dir = os.path.join(os.path.dirname(__file__), 'bc03//')

    # Time grid (1 Myr to 14 Gyr with 1 Myr step)
    time_grid = np.arange(1, 14000)
    fine_time_grid = np.linspace(0.1, 13999, 139990)

    # Metallicities associated to each key
    metallicity = {
        "m22": 0.0001,
        "m32": 0.0004,
        "m42": 0.004,
        "m52": 0.008,
        "m62": 0.02,
        "m72": 0.05
    }
    if quick:
        metallicity = {
            "m62": 0.02,
        }

    for key, imf in itertools.product(metallicity, ["salp", "chab"]):
        base_filename = bc03_dir + "bc2003_lr_" + key + "_" + imf + "_ssp"
        ssp_filename = base_filename + ".ised_ASCII"
        color3_filename = base_filename + ".3color"
        color4_filename = base_filename + ".4color"

        print("Importing %s..." % base_filename)

        # Read the desired information from the color files
        color_table = []
        color3_table = np.genfromtxt(color3_filename).transpose()
        color4_table = np.genfromtxt(color4_filename).transpose()
        color_table.append(color4_table[6])        # Mstar
        color_table.append(color4_table[7])        # Mgas
        color_table.append(10 ** color3_table[5])  # NLy

        color_table = np.array(color_table)

        ssp_time, ssp_wave, ssp_lumin = read_bc03_ssp(ssp_filename)

        # Regrid the SSP data to the evenly spaced time grid. In doing so we
        # assume 10 bursts every 0.1 Myr over a period of 1 Myr in order to
        # capture short evolutionary phases.
        # The time grid starts after 0.1 Myr, so we assume the value is the same
        # as the first actual time step.
        fill_value = (color_table[:, 0], color_table[:, -1])
        color_table = interpolate.interp1d(ssp_time, color_table,
                                           fill_value=fill_value,
                                           bounds_error=False,
                                           assume_sorted=True)(fine_time_grid)
        color_table = np.mean(color_table.reshape(3, -1, 10), axis=-1)

        # We have to do the interpolation-averaging in several blocks as it is
        # a bit RAM intensive
        ssp_lumin_interp = np.empty((ssp_wave.size, time_grid.size))
        for i in range(0, ssp_wave.size, 100):
            fill_value = (ssp_lumin[i:i+100, 0], ssp_lumin[i:i+100, -1])
            ssp_interp = interpolate.interp1d(ssp_time, ssp_lumin[i:i+100, :],
                                              fill_value=fill_value,
                                              bounds_error=False,
                                              assume_sorted=True)(fine_time_grid)
            ssp_interp = ssp_interp.reshape(ssp_interp.shape[0], -1, 10)
            ssp_lumin_interp[i:i+100, :] = np.mean(ssp_interp, axis=-1)

        # To avoid the creation of waves when interpolating, we refine the grid
        # beyond 10 μm following a log scale in wavelength. The interpolation
        # is also done in log space as the spectrum is power-law-like
        ssp_wave_resamp = np.around(np.logspace(np.log10(10000),
                                                np.log10(160000), 50))
        argmin = np.argmin(10000.-ssp_wave > 0)-1
        ssp_lumin_resamp = 10.**interpolate.interp1d(
                                    np.log10(ssp_wave[argmin:]),
                                    np.log10(ssp_lumin_interp[argmin:, :]),
                                    assume_sorted=True,
                                    axis=0)(np.log10(ssp_wave_resamp))

        ssp_wave = np.hstack([ssp_wave[:argmin+1], ssp_wave_resamp])
        ssp_lumin = np.vstack([ssp_lumin_interp[:argmin+1, :],
                               ssp_lumin_resamp])

        base.add_bc03(BC03(
            imf,
            metallicity[key],
            time_grid,
            ssp_wave,
            color_table,
            ssp_lumin
        ))


def build_dale2014(base):
    dale2014_dir = os.path.join(os.path.dirname(__file__), 'dale2014/')

    # Getting the alpha grid for the templates
    d14cal = np.genfromtxt(dale2014_dir + 'dhcal.dat')
    alpha_grid = d14cal[:, 1]

    # Getting the lambda grid for the templates and convert from microns to nm.
    first_template = np.genfromtxt(dale2014_dir + 'spectra.0.00AGN.dat')
    wave = first_template[:, 0] * 1E3

    # Getting the stellar emission and interpolate it at the same wavelength
    # grid
    stell_emission_file = np.genfromtxt(dale2014_dir +
                                        'stellar_SED_age13Gyr_tau10Gyr.spec')
    # A -> to nm
    wave_stell = stell_emission_file[:, 0] * 0.1
    # W/A -> W/nm
    stell_emission = stell_emission_file[:, 1] * 10
    stell_emission_interp = np.interp(wave, wave_stell, stell_emission)

    # The models are in nuFnu and contain stellar emission.
    # We convert this to W/nm and remove the stellar emission.

    # Emission from dust heated by SB
    fraction = 0.0
    filename = dale2014_dir + "spectra.0.00AGN.dat"
    print("Importing {}...".format(filename))
    datafile = open(filename)
    data = "".join(datafile.readlines())
    datafile.close()

    for al in range(1, len(alpha_grid)+1, 1):
        lumin_with_stell = np.genfromtxt(io.BytesIO(data.encode()),
                                         usecols=(al))
        lumin_with_stell = pow(10, lumin_with_stell) / wave
        constant = lumin_with_stell[7] / stell_emission_interp[7]
        lumin = lumin_with_stell - stell_emission_interp * constant
        lumin[lumin < 0] = 0
        lumin[wave < 2E3] = 0
        norm = np.trapezoid(lumin, x=wave)
        lumin = lumin/norm

        base.add_dale2014(Dale2014(fraction, alpha_grid[al-1], wave, lumin))

    # Emission from dust heated by AGN - Quasar template
    filename = dale2014_dir + "shi_agn.regridded.extended.dat"
    print("Importing {}...".format(filename))

    wave, lumin_quasar = np.genfromtxt(filename, unpack=True)
    wave *= 1e3
    lumin_quasar = 10**lumin_quasar / wave
    norm = np.trapezoid(lumin_quasar, x=wave)
    lumin_quasar = lumin_quasar / norm

    base.add_dale2014(Dale2014(1.0, 0.0, wave, lumin_quasar))


def build_dl2007(base, quick=False):
    dl2007_dir = os.path.join(os.path.dirname(__file__), 'dl2007/')

    qpah = {
        "00": 0.47,
        "10": 1.12,
        "20": 1.77,
        "30": 2.50,
        "40": 3.19,
        "50": 3.90,
        "60": 4.58
    }

    umaximum = ["1e3", "1e4", "1e5", "1e6"]
    uminimum = ["0.10", "0.15", "0.20", "0.30", "0.40", "0.50", "0.70",
                "0.80", "1.00", "1.20", "1.50", "2.00", "2.50", "3.00",
                "4.00", "5.00", "7.00", "8.00", "10.0", "12.0", "15.0",
                "20.0", "25.0"]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {"00": 0.0100, "10": 0.0100, "20": 0.0101, "30": 0.0102,
            "40": 0.0102, "50": 0.0103, "60": 0.0104}

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    datafile = open(dl2007_dir + "U{}/U{}_{}_MW3.1_{}.txt".format(umaximum[0],
                                                                  umaximum[0],
                                                                  umaximum[0],
                                                                  "00"))
    data = "".join(datafile.readlines()[-1001:])
    datafile.close()

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # For some reason wavelengths are decreasing in the model files
    wave = wave[::-1]
    # We convert wavelengths from μm to nm
    wave *= 1000.

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4. * np.pi * 1e-30 / (cst.m_p+cst.m_e) * cst.c / (wave*wave) * 1e9

    for model in sorted(qpah.keys()):
        for umin in uminimum:
            if quick and umin != "1.00": continue
            filename = dl2007_dir + "U{}/U{}_{}_MW3.1_{}.txt".format(umin,
                                                                     umin,
                                                                     umin,
                                                                     model)
            print("Importing {}...".format(filename))
            datafile = open(filename)
            data = "".join(datafile.readlines()[-1001:])
            datafile.close()
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # For some reason fluxes are decreasing in the model files
            lumin = lumin[::-1]
            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv/MdMH[model]

            base.add_dl2007(DL2007(qpah[model], umin, umin, wave, lumin))
            for umax in umaximum:
                if quick: continue
                filename = dl2007_dir + "U{}/U{}_{}_MW3.1_{}.txt".format(umin,
                                                                         umin,
                                                                         umax,
                                                                         model)
                print("Importing {}...".format(filename))
                datafile = open(filename)
                data = "".join(datafile.readlines()[-1001:])
                datafile.close()
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                # For some reason fluxes are decreasing in the model files
                lumin = lumin[::-1]

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv/MdMH[model]

                base.add_dl2007(DL2007(qpah[model], umin, umax, wave, lumin))


def build_dl2014(base, quick=False):
    dl2014_dir = os.path.join(os.path.dirname(__file__), 'dl2014/')

    qpah = {"000": 0.47, "010": 1.12, "020": 1.77, "030": 2.50, "040": 3.19,
            "050": 3.90, "060": 4.58, "070": 5.26, "080": 5.95, "090": 6.63,
            "100": 7.32}

    uminimum = ["0.100", "0.120", "0.150", "0.170", "0.200", "0.250", "0.300",
                "0.350", "0.400", "0.500", "0.600", "0.700", "0.800", "1.000",
                "1.200", "1.500", "1.700", "2.000", "2.500", "3.000", "3.500",
                "4.000", "5.000", "6.000", "7.000", "8.000", "10.00", "12.00",
                "15.00", "17.00", "20.00", "25.00", "30.00", "35.00", "40.00",
                "50.00"]

    alpha = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8",
             "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
             "2.8", "2.9", "3.0"]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {"000": 0.0100, "010": 0.0100, "020": 0.0101, "030": 0.0102,
            "040": 0.0102, "050": 0.0103, "060": 0.0104, "070": 0.0105,
            "080": 0.0106, "090": 0.0107, "100": 0.0108}

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    datafile = open(dl2014_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                    .format(uminimum[0], uminimum[0], "000"))

    data = "".join(datafile.readlines()[-1001:])
    datafile.close()

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # For some reason wavelengths are decreasing in the model files
    wave = wave[::-1]
    # We convert wavelengths from μm to nm
    wave *= 1000.

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4. * np.pi * 1e-30 / (cst.m_p+cst.m_e) * cst.c / (wave*wave) * 1e9

    for model in sorted(qpah.keys()):
        for umin in uminimum:
            if quick and umin != "1.000": continue
            filename = (dl2014_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                        .format(umin, umin, model))
            print("Importing {}...".format(filename))
            with open(filename) as datafile:
                data = "".join(datafile.readlines()[-1001:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # For some reason fluxes are decreasing in the model files
            lumin = lumin[::-1]

            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv/MdMH[model]

            base.add_dl2014(DL2014(qpah[model], umin, umin, 1.0, wave, lumin))
            for al in alpha:
                if quick: continue
                filename = (dl2014_dir + "U{}_1e7_MW3.1_{}/spec_{}.dat"
                            .format(umin, model, al))
                print("Importing {}...".format(filename))
                with open(filename) as datafile:
                    data = "".join(datafile.readlines()[-1001:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                # For some reason fluxes are decreasing in the model files
                lumin = lumin[::-1]

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv/MdMH[model]

                base.add_dl2014(DL2014(qpah[model], umin, 1e7, al, wave,
                                       lumin))


def build_fritz2006(base):
    fritz2006_dir = os.path.join(os.path.dirname(__file__), 'fritz2006/')

    # Parameters of Fritz+2006
    psy = [0.001, 10.100, 20.100, 30.100, 40.100, 50.100, 60.100, 70.100,
           80.100, 89.990]  # Viewing angle in degrees
    opening_angle = ["20", "40", "60"]  # Theta = 2*(90 - opening_angle)
    gamma = ["0.0", "2.0", "4.0", "6.0"]
    beta = ["-1.00", "-0.75", "-0.50", "-0.25", "0.00"]
    tau = ["0.1", "0.3", "0.6", "1.0", "2.0", "3.0", "6.0", "10.0"]
    r_ratio = ["10", "30", "60", "100", "150"]

    # Read and convert the wavelength
    datafile = open(fritz2006_dir + "ct{}al{}be{}ta{}rm{}.tot"
                    .format(opening_angle[0], gamma[0], beta[0], tau[0],
                            r_ratio[0]))
    data = "".join(datafile.readlines()[-178:])
    datafile.close()
    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    wave *= 1e3
    # Number of wavelengths: 178; Number of comments lines: 28
    nskip = 28
    blocksize = 178

    iter_params = ((oa, gam, be, ta, rm)
                   for oa in opening_angle
                   for gam in gamma
                   for be in beta
                   for ta in tau
                   for rm in r_ratio)

    for params in iter_params:
        filename = fritz2006_dir + "ct{}al{}be{}ta{}rm{}.tot".format(*params)
        print("Importing {}...".format(filename))
        try:
            datafile = open(filename)
        except IOError:
            continue
        data = datafile.readlines()
        datafile.close()

        for n in range(len(psy)):
            block = data[nskip + blocksize * n + 4 * (n + 1) - 1:
                         nskip + blocksize * (n+1) + 4 * (n + 1) - 1]
            lumin_therm, lumin_scatt, lumin_agn = np.genfromtxt(
                io.BytesIO("".join(block).encode()), usecols=(2, 3, 4),
                unpack=True)
            # Remove NaN
            lumin_therm = np.nan_to_num(lumin_therm)
            lumin_scatt = np.nan_to_num(lumin_scatt)
            lumin_agn = np.nan_to_num(lumin_agn)
            # Conversion from erg/s/microns to W/nm
            lumin_therm *= 1e-4
            lumin_scatt *= 1e-4
            lumin_agn *= 1e-4
            # Normalization of the lumin_therm to 1W
            norm = np.trapezoid(lumin_therm, x=wave)
            lumin_therm = lumin_therm / norm
            lumin_scatt = lumin_scatt / norm
            lumin_agn = lumin_agn / norm

            base.add_fritz2006(Fritz2006(params[4], params[3], params[2],
                                         params[1], params[0], psy[n], wave,
                                         lumin_therm, lumin_scatt, lumin_agn))


def build_activate(base, fine_netzer_disk=False, all_spins=False):
    activate_dir = os.path.join(os.path.dirname(__file__), 'activate/')

    # conversion from Fnu to Flam:
    # nu = c/lam
    # Fnu = nu^3/c^2 = c^3/lam^3/c^2 = c/lam^3
    # Flam = c^2/lam^5 = Fnu * c / lam^2
    
    # Prevot attenuation template
    print("Importing Activate Prevot attenuation ...")
    filename = activate_dir + "absorption/SMC_prevot.dat"
    print("    parsing %s ..." % filename)
    data = np.genfromtxt(filename)
    wave = data[:,0] / 10. # A to nm
    k = data[:,1]
    base.add_AttenuationLaw(AttenuationLaw("Prevot", wave, k))
    del data, wave, k
    
    # galaxy template by Camilla Pacifici
    print("Importing Activate Pacifici2012Gal ...")
    for name in ['qui', 'sf']:
        for i in range(1, 1000):
            filename = activate_dir + "gal/template_%s%d.dat" % (name, i)
            if not os.path.exists(filename):
                break
            print("    parsing %s ..." % filename)
            data = np.genfromtxt(filename)
            wave = data[:,0] / 10. # A to nm
            Llam = data[:,1]
            base.add_ActivatePacifici2012Gal(Pacifici2012Gal("%s%d" % (name, i), wave, Llam))
    del data, Llam, wave
    
    # Mor Netzer disk templates: M, Mdot, a
    #M = ["6.0", "7.0", "8.0", "9.0"]
    #a = ["0.998", "0"]
    #Mdot = ["0.3", "0.03"]
    inc = ["0"]
    c = 3.0e18 # arbitrary units, we only care about the shape
    
    included = set()
    if fine_netzer_disk:
        print("Importing Activate NetzerDisk (fine grid, %s) ..." % ('all spins' if all_spins else 'spins a=0,0.7,0.998,-1'))
        # finer data table with spins
        header = np.loadtxt(activate_dir + "agn/mor_netzer_2012/table_of_models_mbh_03_Mdot_03_spin21_header")
        #header = np.loadtxt(activate_dir + "agn/mor_netzer_2012/table_all_MBH_Mdot_0.1_0.1_spin_21_header")
        logMBHs = header[:,0]
        Mdots = header[:,1]
        spins = header[:,5]
        data = np.loadtxt(activate_dir + "agn/mor_netzer_2012/table_of_models_mbh_03_Mdot_03_spin21")[::-1]
        #data = np.loadtxt(activate_dir + "agn/mor_netzer_2012/table_all_MBH_Mdot_0.1_0.1_spin_21.gz")[::-1]
        wave = data[:,1] * 0.1 # A -> nm
        freq = data[:,0]
        for i, (Mv, av, Mdotv) in enumerate(zip(logMBHs, spins, Mdots)):
            if av not in (0, 0.7, 0.998, -1) and not all_spins:
                # print("skipping spin", av)
                continue
            Lnu = data[:,2+i]
            assert (Lnu >= 0).all(), (Lnu)
            Llam = Lnu * freq**2 / c
            assert (Llam >= 0).all(), (Llam)
            params = (str(Mv), str(av), str(Mdotv), inc[0])
            included.add(params)
            # normalise so that at 510nm, it is 1
            norm = np.interp(510.0, wave, Llam)
            assert norm > 0, (norm, wave, Llam)
            Llam = Llam / norm
            assert (Llam >= 0).all(), (Llam, norm)
            assert np.isfinite(Llam).all(), (Llam, norm)
            assert np.isfinite(wave).all(), (wave)
            #print("  ", params)
            #if i == 0:
            #    print("    ", wave, Llam)
            base.add_ActivateNetzerDisk(NetzerDisk(params[0], params[1], params[2],
                                             params[3], wave, Llam))
    else:
        print("Importing Activate NetzerDisk (course grid, spin=0,0.998) ...")
        M = ["6.0", "7.0", "8.0", "9.0"]
        a = ["0.998", "0"]
        Mdot = ["0.3", "0.03"]
        inc = ["0"]
        datafile = open(activate_dir + "agn/mor_netzer_2012/table_of_disk_models")
        data = "".join(datafile.readlines()[23:][::-1]) # reverse
        datafile.close()
        data = np.genfromtxt(io.BytesIO(data.encode()))
        wave = data[:,1] * 0.1 # A -> nm
        freq = data[:,0]
        #included = set()
        #for i, (Mv, av, Mdotv) in enumerate([(0,1,1),(0,1,0),(0,0,0),(0,0,1),(1,1,1),(1,1,0),(1,0,0),(1,0,1)]):
        options = [(Mv, av, Mdotv) for av in a for Mdotv in Mdot for Mv in M]
        for i, (Mv, av, Mdotv) in enumerate(options):
            Lnu = data[:,2+i]
            assert (Lnu >= 0).all(), (Lnu)
            Llam = Lnu * freq**2 / c
            assert (Llam >= 0).all(), (Llam)
            params = (Mv, av, Mdotv, inc[0])
            if params in included:
                continue
            included.add(params)
            # normalise so that at 510nm, it is 1
            norm = np.interp(510.0, wave, Llam)
            assert norm > 0, (norm, wave, Llam)
            Llam = Llam / norm
            assert (Llam >= 0).all(), (Llam, norm)
            print("  ", params)
            base.add_ActivateNetzerDisk(NetzerDisk(params[0], params[1], params[2],
                                             params[3], wave, Llam))
            del Llam, Lnu
    del wave, freq
    
    # Mor Netzer torus template
    print("Importing Activate MorNetzer2012Torus ...")
    for filename, torustype, col, factor in [
        ('mor_netzer_mean_and_uncertainty_extended', 'mor-avg', 1, 0.51),
        ('mor_netzer_mean_and_uncertainty_extended', 'mor-lo', 2, 0.36),
        ('mor_netzer_mean_and_uncertainty_extended', 'mor-hi', 3, 0.73),
        # ('fuller_mean', 'type-2', 1, True)]:
    ]: 
        data = np.genfromtxt(activate_dir + "agn/mor_netzer_2012/" + filename)
        wave = data[:,0] * 1000 # micron to nm
        freq = c / wave
        nuLnu = data[:,col] # is multiplied by frequency
        
        #assert wave[0] == 510.0, wave[0] # normalisation
        Llam = nuLnu / freq * c / wave**2
        # normalise so that at 12um, it is 1
        norm = Llam[wave == 12000][0] # get normalisation at 12um
        assert norm > 0, (norm, Llam)
        Llam = Llam / norm
        # model is valid above 1um
        # here we only pick above 2um, and then extrapolate 
        mask = wave > 2000
        Llam[~mask] = Llam[mask][0]
        assert (Llam >= 0).all(), Llam
        base.add_ActivateMorNetzer2012Torus(MorNetzer2012Torus(torustype, wave, Llam))
        
        # add same, but subtract a hot dust approximation
        nuLnu = nuLnu/factor - np.exp(-(np.log10(data[:,0] / 2.6) / 0.5)**2)
        Llam = nuLnu / freq * c / wave**2
        # model is valid above 1um
        # here we only pick above 2um, and then extrapolate 
        mask = np.logical_and(wave > 2000, Llam > 0)
        Llam[~mask] = 0.0
        base.add_ActivateMorNetzer2012Torus(MorNetzer2012Torus(torustype + '-cold', wave, Llam))
        


    # Mullaney templates show a slight difference between high and low luminosity
    # in the Silicate feature depth
    # so we isolate the Si feature here:
    data = np.genfromtxt(activate_dir + "agn/mor_netzer_2012/othermodels/Mullaney.txt")
    wave = data[:,0] * 1000 # micron to nm
    freq = c / wave
    # make the two spectra equal at 18000:
    i = np.where(wave > 18000)[0][0]
    specA = data[:,2]
    specB = data[:,3] * (data[i,2] / data[i,3])
    nuLnu = (specA + specB) / 2.
    nuLnu *= freq
    Llam = nuLnu / freq * c / wave**2 
    # get continuum normalisation at 12um
    norm = Llam[wave == 12000][0]
    
    # get difference spectrum:
    # normalise the two templates at 18um, just before the infrared bump
    # this is approximately 98%.
    nuLnu = specA - specB
    # remove stuff outside the Si waveband
    nuLnu[np.logical_and(nuLnu < 0, wave < 8000)] = 0.0
    nuLnu[np.logical_and(nuLnu < 0, wave > 18000)] = 0.0
    nuLnu *= freq
    Llam = nuLnu / freq * c / wave**2 
    mask = np.logical_and(wave > 7000, wave < 19000)
    base.add_ActivateMorNetzer2012Torus(MorNetzer2012Torus('mullaney-silicate', wave[mask], Llam[mask] / norm))
    del Llam, mask, norm, wave

    
    
    # FeII template
    print("Importing Activate FeII ...")
    for FeIImodelname, filename, z in [
        # this template is slightly redshifted, by z=0.004
        ('BruhweilerVerner08', 'Fe_d11-m20-20.5.txt', 4593.4/4575 - 1), 
        ('Veron-Cetty04', 'Veron-Cetty_template.txt', 0)]:
        #data = np.genfromtxt(activate_dir + "agn/FeII_template/Fe_d11-m20-20.5.txt")
        data = np.genfromtxt(activate_dir + "agn/FeII_template/" + filename)
        wavez = data[:,0]
        wave = wavez / (1+z)
        Lnu = data[:,1]
        assert wave.shape == Lnu.shape, (wave.shape, Lnu.shape)
        Llam = Lnu * c / wavez**2
        assert wave.shape == Llam.shape, (wave.shape, Llam.shape)
        assert (Llam >= 0).all(), Llam
        # normalisation at FeII 4575
        print(wave[np.argmin(np.abs(wave - 4575))], wavez[np.argmin(np.abs(wave - 4575))])
        norm1 = np.max(Llam[np.argmin(np.abs(wave - 4575))])
        #norm = np.interp(457.5, wave, Llam)
        # normalisation over entire wavelength
        norm = np.trapezoid(Llam, x=wave*0.1)
        norm = norm1
        assert norm > 0, (norm, Llam)
        Llam = Llam / norm
        #print('    largest luminosity after normalising:', np.max(Llam))
        print('    FeII normalisation:', norm, norm1/norm)
        #assert norm == 8.30E+06 * c / 4593.4**2, (norm, 8.30E+06 * c / 4593.4**2, np.max(Llam))
        assert wave.shape == Llam.shape, (wave.shape, Llam.shape)
        base.add_ActivateFeII(FeII(FeIImodelname, wave * 0.1, Llam))
        del Llam, norm, wave, FeIImodelname
    
    # Emission line list & strengths
    print("Importing Activate MorNetzerEmLines ...")
    # normalise so that Hb = 1 -- already done so in that file
    data = np.loadtxt(activate_dir + 'agn/mor_netzer_2012/emission_line_table.formatted', 
    	dtype=[('name', 'S10'), ('wave', 'f'), ('broad', 'f'), ('S2', 'f'), ('LINER', 'f')])
    assert (data['broad'] >= 0).all(), data
    assert (data['S2'] >= 0).all(), data
    assert (data['LINER'] >= 0).all(), data
    base.add_ActivateMorNetzerEmLines(MorNetzerEmLines(data['wave'] * 0.1, data['broad'], data['S2'], data['LINER']))
    
default_lines = set([
        "ArIII-713.6",
        "CII-232.4",
        "CII-232.47",
        "CII-232.54",
        "CII-232.7",
        "CII-232.8",
        "CIII-190.7",
        "CIII-190.9",
        "H-alpha",
        "H-beta",
        "H-delta",
        "H-gamma",
        "HeII-164.0",
        "Ly-alpha",
        "NII-654.8",
        "NII-658.3",
        "NeIII-396.7",
        "OI-630.0",
        "OII-372.6",
        "OII-372.9",
        "OIII-495.9",
        "OIII-500.7",
        "Pa-alpha",
        "Pa-beta",
        "Pa-gamma",
        "SII-671.6",
        "SII-673.1",
        "SIII-906.9",
        "SIII-953.1",
])

def build_nebular(base):
    path = os.path.join(os.path.dirname(__file__), 'nebular/')

    filename = os.path.join(path, "lines.dat")
    print(f"Importing {filename}...")
    lines = np.genfromtxt(filename)

    tmp = Table.read(os.path.join(path, "line_wavelengths.dat"), format='ascii')
    name_lines = tmp['col2'].data
    mask_lines = np.array([name_line in default_lines for name_line in name_lines])
    mask_lines[:] = True
    wave_lines = tmp['col1'].data[mask_lines]
    name_lines = name_lines[mask_lines]

    # Build the parameters
    metallicities = np.unique(lines[:, 1])
    logUs = np.around(np.arange(-4., -.9, .1), 1)
    nes = np.array([10., 100., 1000.])

    filename = os.path.join(path, "continuum.dat")
    print(f"Importing {filename}...")
    cont = np.genfromtxt(filename)

    # Convert wavelength from Å to nm
    wave_lines *= 0.1
    wave_cont = cont[:1600, 0] * 0.1

    # Compute the wavelength grid to resample the models so as to eliminate
    # non-physical waves and compute the models faster by avoiding resampling
    # them at run time.
    
    wave_stellar = base.get_bc03(imf="salp", metallicity=0.02).wl
    wave_dust = base.get_dl2014(qpah=0.47, umin=1., umax=1., alpha=1.).wave
    wave_cont_interp = np.unique(np.hstack([wave_cont, wave_stellar, wave_dust,
                                            np.logspace(7., 9., 501)]))

    # Keep only the fluxes
    lines = lines[:, 2:]
    cont = cont[:, 1:]

    # Reshape the arrays so they are easier to handle
    cont = np.reshape(cont, (metallicities.size, wave_cont.size, logUs.size,
                             nes.size))
    lines = np.reshape(lines, (mask_lines.size, metallicities.size, logUs.size,
                               nes.size))[mask_lines,:,:,:]

    # Move the wavelength to the last position to ease later computations
    # 0: metallicity, 1: log U, 2: ne, 3: wavelength
    cont = np.moveaxis(cont, 1, -1)
    lines = np.moveaxis(lines, (0, 1, 2, 3), (3, 0, 1, 2))

    # Convert lines to a linear scale
    lines = 10.0 ** lines

    # Convert continuum to W/nm
    cont *= 1e-7 * cst.c * 1e9 / wave_cont**2

    # Import lines
    for idxZ, metallicity in enumerate(metallicities):
        for idxU, logU in enumerate(logUs):
            for ne, spectrum in zip(nes, lines[idxZ, idxU, :, :]):
                line = NebularLines(float(metallicity), float(logU), float(ne), wave_lines, spectrum)
                base.add_nebular_lines(line)

    # Import continuum
    #db = SimpleDatabase("nebular_continuum", writable=True)
    spectra = 10 ** interpolate.interp1d(np.log10(wave_cont), np.log10(cont),
                                         axis=-1)(np.log10(wave_cont_interp))
    spectra = np.nan_to_num(spectra)
    for idxZ, metallicity in enumerate(metallicities):
        for idxU, logU in enumerate(logUs):
            for ne, spectrum in zip(nes, spectra[idxZ, idxU, :, :]):
                cont = NebularContinuum(float(metallicity), float(logU), float(ne), wave_cont_interp, spectrum)
                base.add_nebular_continuum(cont)


def build_base(speed):
    base = Database(writable=True)
    base.upgrade_base()
    

    print('#' * 78)
    print("1- Importing filters...\n")
    build_filters(base)
    print("\nDONE\n")
    print('#' * 78)

    print("2- Importing Maraston 2005 SSP\n")
    build_m2005(base, quick=speed > 1)
    print("\nDONE\n")
    print('#' * 78)

    print("3- Importing Bruzual and Charlot 2003 SSP\n")
    build_bc2003(base, quick=speed > 1)
    print("\nDONE\n")
    print('#' * 78)

    if speed > 1:
        print("4- Importing Draine and Li (2007) models\n")
        build_dl2007(base, quick=speed > 0)
        print("\nDONE\n")
        print('#' * 78)

    print("5- Importing the updated Draine and Li (2007 models)\n")
    build_dl2014(base, quick=speed > 0)
    print("\nDONE\n")
    print('#' * 78)

    print("6- Importing Activate models\n")
    build_activate(base, fine_netzer_disk=True, all_spins=False if speed >= 2 else True)
    print("\nDONE\n")
    print('#' * 78)
    
    if speed < 2:
        print("6- Importing Fritz et al. (2006) models\n")
        build_fritz2006(base)
        print("\nDONE\n")
        print('#' * 78)

    print("7- Importing Dale et al (2014) templates\n")
    build_dale2014(base)
    print("\nDONE\n")
    print('#' * 78)

    print("8- Importing nebular lines and continuum\n")
    build_nebular(base)
    print("\nDONE\n")
    print('#' * 78)

    base.session.close_all()

if __name__ == '__main__':
    build_base(speed=int(os.environ.get('SPEED', '1')))
