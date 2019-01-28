Study analysis template with MNE
================================

Steps ...
------------------

First, clone the repository using git:

	$ git clone https://github.com/mne-tools/mne-study-analysis-template.git

Next, make sure your system is properly configured. You should have the dependencies installed,
namely: numpy, scipy, matplotlib, scikit-learn, pysurfer, mayavi, mne-python, sphinx_gallery,
and freesurfer.

The easiest way to get the Python dependencies is to use the `environment.yml` file that we ship
inside the `scripts/` folder. First, download [Anaconda](https://anaconda.org/) and then run the following two commands:

	$ conda env create -f environment.yml
	$ source activate mne

For freesurfer, follow the instructions on [their website](https://surfer.nmr.mgh.harvard.edu/). Then, go to the `scripts/processing` folder and do:

	$ make check

You may want to edit the file `scripts/processing/library/config.py` 
to specify the number of subjects you can to run in parallel (N_JOBS). Note that
using more `N_JOBS` will increase the memory requirements as the data will be
copied across parallel processes.

Authors
-------

- [Mainak Jas](http://perso.telecom-paristech.fr/~mjas/), Telecom ParisTech, Université Paris-Saclay
- [Eric Larson](http://larsoner.com), University of Washington ILABS
- [Denis Engemann](http://denis-engemann.de), Neurospin, CEA/INSERM, UNICOG Team
- Jaakko Leppäkangas, Telecom ParisTech, Université Paris-Saclay
- [Samu Taulu](http://ilabs.washington.edu/institute-faculty/bio/i-labs-samu-taulu-dsc), University of Washington, ILABS
- [Matti Hämäläinen](https://www.martinos.org/user/5923), Martinos Center, MGH, Harvard Medical School
- [Alexandre Gramfort](http://alexandre.gramfort.net), INRIA, Université Paris-Saclay
