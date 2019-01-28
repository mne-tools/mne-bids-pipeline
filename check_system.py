"""
==================
Check dependencies
==================

Check if dependencies are present
"""
import os
import warnings

try:
    import mne  # noqa
except:
    raise ValueError('mne is not installed.')

try:
    import nibabel  # noqa
except:
    raise ValueError('nibabel is not installed.')

try:
    import autoreject  # noqa
except:
    raise ValueError('autoreject is not installed.')

try:
    import sklearn  # noqa
except:
    raise ValueError('sklearn is not installed.')

mne.sys_info()

FREESURFER_HOME = os.environ.get("FREESURFER_HOME", None)
if FREESURFER_HOME is None:
    raise ValueError('freesurfer is not available on your system.')
else:
    print("FREESURFER_HOME: %s" % FREESURFER_HOME)

OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", None)
if OMP_NUM_THREADS is None:
    warnings.warn('OMP_NUM_THREADS is not set. We recommend you set it to '
                  '2 or 4 depending on your system.')
else:
    print("OMP_NUM_THREADS: %s" % OMP_NUM_THREADS)

# For building the doc:

try:
    import sphinx_gallery  # noqa
    print("sphinx_gallery: %s" % sphinx_gallery.__version__)
except:
    raise ValueError('sphinx_gallery is not installed. '
                     'Run: pip install sphinx_gallery')

try:
    import sphinx_bootstrap_theme  # noqa
    print("sphinx_bootstrap_theme: %s" % sphinx_bootstrap_theme.__version__)
except:
    raise ValueError('sphinx_bootstrap_theme is not installed. '
                     'Run: pip install sphinx_bootstrap_theme')
