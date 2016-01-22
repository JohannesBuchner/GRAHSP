Overview of Inputs
======================

Host galaxies:

* Grid of many templates: includes emission lines **template id**
* Extinction law: **Av parameter**
  * absorption/SMC_prevot.dat

AGN:

* Disk: Powerlaw: 10 Templates: **M, Mdot, a, fcover, L5100**
  * File: agn/mor_netzer*/table_of_disk_models
* Torus: Mor+Netzer 1 Template
  * File: agn/mor_netzer*/mor_netzer*
* Emission Lines: List 
  * Broad, Narrow, LINER lines:
    * File: agn/mor_netzer*/emission_line_table
    * Recipes: agn/mor_netzer*/readme
  * FeII template
    * File: agn/FeII_template/Fe_d11-m20-20.5.txt


