"""
=====
Clean
=====

Auxiliary file for Make
"""
import shutil
import os
import os.path as op
from library.config import meg_dir

shutil.rmtree(meg_dir, ignore_errors=True)
os.mkdir(meg_dir)
for subject_id in range(1, 20):
    subject = "sub%03d" % subject_id
    this_dir = op.join(meg_dir, subject)
    print('Creating %s' % this_dir)
    os.mkdir(op.join(meg_dir, subject))
