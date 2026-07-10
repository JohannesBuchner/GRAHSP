"""
Create fiducial Gaussian filter curves for SPHEREx.

Written by Raphael Shirley, Johannes Buchner.
"""
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np

filter_bins = Table.read("spectral_channels_spx_cal-sch-v1-2026-106.fits.gz")

FILTER_LIST = []

colors = ['blue', 'lightblue', 'cyan', 'lightgreen', 'orange', 'red']

fig, ax = plt.subplots(figsize=(12, 5))

for row in filter_bins:
    l = row["WAVELENGTH"]
    det = row["DETECTOR"]
    subchan = row["SUBCHAN"]
    filter_name = f"SPHEREx{int(round(l*100)):03d}"
    filter_filename = f"{filter_name}.dat"
    half_width = row["BANDWIDTH"] * 1e4 / 2
    midpoint = l * 1e4  # um to Angstrom

    mean = midpoint
    # FWHM to standard deviation
    std = row["BANDWIDTH"] * 1e4 / 2.355

    # Define wavelengths and transmissions
    x = np.linspace(mean - 2 * std, mean + 2 * std, 100)
    gaussian = np.exp(-0.5 * ((x - mean) / std) ** 2)

    # Exactly zero at the ends
    gaussian[[0, -1]] = 0

    # Combine columns
    # data = np.column_stack((wavelength, transmission))
    data = np.column_stack((x, gaussian))

    # Output filename
    FILTER_LIST.append(filter_name)
    FILTER_LIST.append(filter_name + '_err')
    # Save file
    np.savetxt(filter_filename, data, fmt=["%.5f", "%.3f"], header=f"{filter_name}\nphoton\nSPHEREx detector {det} subchannel {subchan} from https://irsa.ipac.caltech.edu/ibe/data/spherex/qr2/spectral_channels/cal-sch-v1-2026-106/spectral_channels_spx_cal-sch-v1-2026-106.fits.gz assuming a Gaussian of sigma=Bandwidth")
    
    plt.plot(x / 10000, gaussian, label=filter_name, color=colors[det - 1], linewidth=0.8)

plt.xticks([0.75, 1.11, 1.64, 2.42, 3.82, 4.42, 5.0])
plt.xlim(0.72, 5.03)
plt.xlabel(r'Wavelength $\lambda_c$ [µm]')
plt.ylabel(r'Normalised response')
plt.savefig('curves.pdf')
print(', '.join(FILTER_LIST))
