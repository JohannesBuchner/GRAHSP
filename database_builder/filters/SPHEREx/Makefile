spectral_channels_spx_cal-sch-v1-2026-106.fits.gz:
	rm -f spectral_channels_spx_cal-sch-v1-2026-106.fits.gz
	wget https://irsa.ipac.caltech.edu/ibe/data/spherex/qr2/spectral_channels/cal-sch-v1-2026-106/spectral_channels_spx_cal-sch-v1-2026-106.fits.gz

curves.pdf: unpack.py spectral_channels_spx_cal-sch-v1-2026-106.fits.gz
	python3 $<
