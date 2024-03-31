"""MIND DATA.

M.P. Weisend, F.M. Hanlon, R. Montaño, S.P. Ahlfors, A.C. Leuthold,
D. Pantazis, J.C. Mosher, A.P. Georgopoulos, M.S. Hämäläinen, C.J.
Aine,, V. (2007).
Paving the way for cross-site pooling of magnetoencephalography (MEG) data.
International Congress Series, Volume 1300, Pages 615-618.
"""

# This has auditory, median, indx, visual, rest, and emptyroom but let's just
# process the auditory (it's the smallest after rest)
bids_root = "~/mne_data/ds004107"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds004107"
subjects = ["mind002"]
sessions = ["01"]
conditions = ["left", "right"]  # there are also tone and noise
task = "auditory"
ch_types = ["meg"]
crop_runs = (0, 120)  # to speed up computations
spatial_filter = "ssp"
l_freq = 1.0
h_freq = 40.0
