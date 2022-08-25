# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

from distutils.command.build import build

from setuptools import find_packages, setup


class custom_build(build):
    def run(self):
        import os
        if os.path.exists('pcigale/data/data.db'):
            os.unlink('pcigale/data/data.db')
        # Build the database.
        import database_builder
        database_builder.build_base()

        # Proceed with the build
        build.run(self)

entry_points = {
    'console_scripts': ['pcigale-grasp = pcigale:main',
                        'pcigale-grasp-plots = pcigale_plots:main']
}

setup(
    name="grahsp",
    version="0.7.5",
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
