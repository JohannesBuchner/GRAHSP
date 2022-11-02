# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

from distutils.command.build import build

from setuptools import find_packages, setup

if 'SPEED' not in os.environ:
    raise Exception("""
Set environment variable SPEED to one of:
    2 -- quick and small GRAHSP install, 700MB
    1 -- full physical agn models (Fritz,Netzer), 2800MB, or 
    0 -- like 1 but also include non-solar metallicity galaxies and Draine&Li dust models), 3200MB, 
""")

class custom_build(build):
    def run(self):
        import os
        if os.path.exists('pcigale/data/data.db'):
            os.unlink('pcigale/data/data.db')
        # Build the database.
        import database_builder
        speed = int(os.environ['SPEED'])
        database_builder.build_base(speed=speed)

        # Proceed with the build
        build.run(self)

entry_points = {
    'console_scripts': ['pcigale-grasp = pcigale:main',
                        'pcigale-grasp-plots = pcigale_plots:main']
}

setup(
    name="grahsp",
    version="0.8.1",
    packages=find_packages(exclude=["database_builder"]),

    install_requires=['numpy', 'scipy', 'sqlalchemy', 'matplotlib',
                      'configobj', 'astropy'],

    entry_points=entry_points,

    cmdclass={"build": custom_build},
    package_data={'pcigale': ['data/data.db'],
                  'pcigale_plots': ['data/CIGALE.png']},

    author="Johannes Buchner, The CIGALE team",
    author_email="johannes.buchner.acad@gmx.com",
    description="Grasping reliably the AGN host stellar population",
    license="CeCILL-V2",
    keywords="astrophysics, galaxy, SED fitting, quasars"
)
