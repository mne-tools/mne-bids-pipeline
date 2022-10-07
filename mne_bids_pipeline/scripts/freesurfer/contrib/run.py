#!/usr/bin/env python3
import argparse
import os
import shutil
import nibabel
from glob import glob
from subprocess import Popen, PIPE
from shutil import rmtree
import subprocess
from warnings import warn
import pandas as pd
import re
import errno

def run(command, env={}, ignore_errors=False):
    merged_env = os.environ
    merged_env.update(env)
    # DEBUG env triggers freesurfer to produce gigabytes of files
    merged_env.pop('DEBUG', None)
    process = Popen(command, stdout=PIPE, stderr=subprocess.STDOUT, shell=True, env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break
    if process.returncode != 0 and not ignore_errors:
        raise Exception("Non zero return code: %d" % process.returncode)


__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'version')).read()

parser = argparse.ArgumentParser(description='FreeSurfer recon-all + custom template generation.')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir. '
                    '"group1" creates study specific group template. '
                    '"group2" exports group stats tables for cortical parcellation, subcortical segmentation '
                                                               'a table with euler numbers.',
                    choices=['participant', 'group1', 'group2'])
parser.add_argument('--participant_label', help='The label of the participant that should be analyzed. The label '
                    'corresponds to sub-<participant_label> from the BIDS spec '
                    '(so it does not include "sub-"). If this parameter is not '
                    'provided all subjects should be analyzed. Multiple '
                    'participants can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('--session_label', help='The label of the session that should be analyzed. The label '
                    'corresponds to ses-<session_label> from the BIDS spec '
                    '(so it does not include "ses-"). If this parameter is not '
                    'provided all sessions should be analyzed. Multiple '
                    'sessions can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('--n_cpus', help='Number of CPUs/cores available to use.',
                    default=1, type=int)
parser.add_argument('--stages', help='Autorecon stages to run.',
                    choices=["autorecon1", "autorecon2", "autorecon2-cp", "autorecon2-wm", "autorecon-pial", "autorecon3", "autorecon-all", "all"],
                    default=["autorecon-all"],
                    nargs="+")
parser.add_argument('--steps', help='Longitudinal pipeline steps to run.',
                    choices=['cross-sectional', 'template', 'longitudinal'],
                    default=['cross-sectional', 'template', 'longitudinal'],
                    nargs="+")
parser.add_argument('--template_name', help='Name for the custom group level template generated for this dataset',
                    default="average")
parser.add_argument('--license_file', help='Path to FreeSurfer license key file. To obtain it you need to register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html',
                    type=str, default='/license.txt')
parser.add_argument('--acquisition_label', help='If the dataset contains multiple T1 weighted images from different acquisitions which one should be used? Corresponds to "acq-<acquisition_label>"')
parser.add_argument('--reconstruction_label', help='If the dataset contains multiple T1 weighted images from different reconstructions which one should be used? Corresponds to "rec-<reconstruction_label>"')
parser.add_argument('--refine_pial_acquisition_label', help='If the dataset contains multiple T2 or FLAIR weighted images from different acquisitions which one should be used? Corresponds to "acq-<acquisition_label>"')
parser.add_argument('--refine_pial_reconstruction_label', help='If the dataset contains multiple T2 or FLAIR weighted images from different reconstructions which one should be used? Corresponds to "rec-<reconstruction_label>"')
parser.add_argument('--multiple_sessions', help='For datasets with multiday sessions where you do not want to '
                    'use the longitudinal pipeline, i.e., sessions were back-to-back, '
                    'set this to multiday, otherwise sessions with T1w data will be '
                    'considered independent sessions for longitudinal analysis.',
                    choices=["longitudinal", "multiday"],
                    default="longitudinal")
parser.add_argument('--refine_pial', help='If the dataset contains 3D T2 or T2 FLAIR weighted images (~1x1x1), '
                    'these can be used to refine the pial surface. If you want to ignore these, specify None or '
                    ' T1only to base surfaces on the T1 alone.',
                    choices=['T2', 'FLAIR', 'None', 'T1only'],
                    default=['T2'])
parser.add_argument('--allow_lowresT2', help='Use T2 images that are lower resolution than (~1x1x1), to refine pial surface.',
                    action='store_true')
parser.add_argument('--hires_mode', help="Submilimiter (high resolution) processing. 'auto' - use only if <1.0mm data detected, 'enable' - force on, 'disable' - force off",
                    choices=['auto', 'enable', 'disable'],
                    default='auto')
parser.add_argument('--parcellations', help='Group2 option: cortical parcellation(s) to extract stats from.',
                    choices=["aparc", "aparc.a2009s"],
                    default=["aparc"],
                    nargs="+")
parser.add_argument('--measurements', help='Group2 option: cortical measurements to extract stats for.',
                    choices=["area", "volume", "thickness", "thicknessstd", "meancurv", "gauscurv", "foldind",
                             "curvind"],
                    default=["thickness"],
                    nargs="+")
parser.add_argument('--qcache', help="Enable qcache", action='store_true')
parser.add_argument('-v', '--version', action='version',
                    version='BIDS-App example version {}'.format(__version__))
parser.add_argument('--bids_validator_config', help='JSON file specifying configuration of '
                    'bids-validator: See https://github.com/INCF/bids-validator for more info')
parser.add_argument('--skip_bids_validator',
                    help='skips bids validation',
                    action='store_true')
parser.add_argument('--3T',
                    help='enables the two 3T specific options that recon-all supports: nu intensity correction params, and the special schwartz atlas',
                    choices = ['true', 'false'],
                    default = 'true')
args = parser.parse_args()


three_T = vars(args)['3T']

if args.bids_validator_config:
    run("bids-validator --config {config} {bids_dir}".format(
        config=args.bids_validator_config,
        bids_dir=args.bids_dir))
elif args.skip_bids_validator:
    print('skipping bids-validator...')
else:
    run("bids-validator {bids_dir}".format(bids_dir=args.bids_dir))

subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))

#Got to combine acq_tpl and rec_tpl
if args.acquisition_label and not args.reconstruction_label:
    ar_tpl = "*acq-%s*" % args.acquisition_label
elif args.reconstruction_label and not args.acquisition_label:
    ar_tpl = "*rec-%s*" % args.reconstruction_label
elif args.reconstruction_label and args.acquisition_label:
    ar_tpl = "*acq-%s*_rec-%s*" % (args.acquisition_label, args.reconstruction_label)
else:
    ar_tpl = "*"

#Got to combine acq_tpl and rec_tpl
if args.refine_pial_acquisition_label and not args.refine_pial_reconstruction_label:
    ar_t2 = "*acq-%s*" % args.refine_pial_acquisition_label
elif args.refine_pial_reconstruction_label and not args.refine_pial_acquisition_label:
    ar_t2 = "*rec-%s*" % args.refine_pial_reconstruction_label
elif args.refine_pial_reconstruction_label and args.refine_pial_acquisition_label:
    ar_t2 = "*acq-%s*_rec-%s*" % (args.refine_pial_acquisition_label, args.refine_pial_reconstruction_label)
else:
    ar_t2 = "*"

# if there are session folders, check if study is truly longitudinal by
# searching for the first subject with more than one valid sessions
multi_session_study = False
if glob(os.path.join(args.bids_dir, "sub-*", "ses-*")):
    subjects = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]
    for subject_label in subjects:
        session_dirs = glob(os.path.join(args.bids_dir, "sub-%s" % subject_label, "ses-*"))
        sessions = [os.path.split(dr)[-1].split("-")[-1] for dr in session_dirs]
        n_valid_sessions = 0
        for session_label in sessions:
            if glob(os.path.join(args.bids_dir, "sub-%s" % subject_label,
                                                "ses-%s" % session_label,
                                                "anat",
                                                "%s_T1w.nii*" % ar_tpl)):
                n_valid_sessions += 1
        if n_valid_sessions > 1:
            multi_session_study = True
            break

if multi_session_study and (args.multiple_sessions == "longitudinal"):
    longitudinal_study = True
else:
    longitudinal_study = False


subjects_to_analyze = []
# only for a subset of subjects
if args.participant_label:
    subjects_to_analyze = args.participant_label
# for all subjects
else:
    subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

# workaround for https://mail.nmr.mgh.harvard.edu/pipermail//freesurfer/2016-July/046538.html
output_dir = os.path.abspath(args.output_dir)

if os.path.exists(args.license_file):
    absPathLic = os.path.abspath(args.license_file)
    env = {'FS_LICENSE': absPathLic}
else:
    raise Exception("Provided license file does not exist")

# running participant level
if args.analysis_level == "participant":
    fst_links_to_make = ["fsaverage", "lh.EC_average","rh.EC_average"]
    for fst in fst_links_to_make:
        try:
            os.symlink(os.path.join(os.environ["SUBJECTS_DIR"], fst),os.path.join(output_dir, fst))
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("Symbolic link to {0} already exists".format(fst))
            else:
                print("ERROR: Symbolic link to {0} unable to be created because: {1}".format(fst,str(e)))
                raise e

    for subject_label in subjects_to_analyze:
        if glob(os.path.join(args.bids_dir, "sub-%s" % subject_label, "ses-*")):
            T1s = glob(os.path.join(args.bids_dir,
                                    "sub-%s" % subject_label,
                                    "ses-*",
                                    "anat",
                                    "%s_T1w.nii*" % (ar_tpl)))
            sessions = set([os.path.normpath(t1).split(os.sep)[-3].split("-")[-1] for t1 in T1s])
            if args.session_label:
                sessions = sessions.intersection(args.session_label)

            if len(sessions) > 0 and longitudinal_study is True:
                timepoints = ["sub-%s_ses-%s" % (subject_label, session_label) for session_label in sessions]
                if 'cross-sectional' in args.steps:
                    # Running each session separately, prior to doing longitudinal pipeline
                    for session_label in sessions:
                        T1s = glob(os.path.join(args.bids_dir,
                                                "sub-%s" % subject_label,
                                                "ses-%s" % session_label,
                                                "anat",
                                                "%s_T1w.nii*" % (ar_tpl)))
                        input_args = ""

                        if three_T == 'true':
                            input_args += " -3T"

                        if args.qcache:
                            input_args += ' -qcache'

                        for T1 in T1s:
                            if (round(max(nibabel.load(T1).header.get_zooms()), 1) < 1.0 and args.hires_mode == "auto") or args.hires_mode == "enable":
                                input_args += " -hires"
                            input_args += " -i %s" % T1

                        T2s = glob(os.path.join(args.bids_dir, "sub-%s" % subject_label,
                                                "ses-%s" % session_label, "anat",
                                                "%s_T2w.nii*" % (ar_t2)))
                        FLAIRs = glob(os.path.join(args.bids_dir, "sub-%s" % subject_label,
                                                   "ses-%s" % session_label, "anat",
                                                   "%s_FLAIR.nii*" % (ar_t2)))
                        if args.refine_pial == "T2":
                            for T2 in T2s:
                                if (max(nibabel.load(T2).header.get_zooms()) < 1.2) | args.allow_lowresT2:
                                    input_args += " " + " ".join(["-T2 %s" % T2])
                                    input_args += " -T2pial"
                        elif args.refine_pial == "FLAIR":
                            for FLAIR in FLAIRs:
                                if (max(nibabel.load(FLAIR).header.get_zooms()) < 1.2) | args.allow_lowresT2:
                                    input_args += " " + " ".join(["-FLAIR %s" % FLAIR])
                                    input_args += " -FLAIRpial"

                        fsid = "sub-%s_ses-%s" % (subject_label, session_label)
                        stages = " ".join(["-" + stage for stage in args.stages])


                        cmd = "recon-all -subjid %s -sd %s %s %s -openmp %d" % (fsid,
                                                                                output_dir,
                                                                                input_args,
                                                                                stages,
                                                                                args.n_cpus)
                        resume_cmd = "recon-all -subjid %s -sd %s %s -openmp %d" % (fsid,
                                                                                    output_dir,
                                                                                    stages,
                                                                                    args.n_cpus)

                        if os.path.isfile(os.path.join(output_dir, fsid, "scripts/IsRunning.lh+rh")):
                            rmtree(os.path.join(output_dir, fsid))
                            print("DELETING OUTPUT SUBJECT DIR AND RE-RUNNING COMMAND:")
                            print(cmd)
                            run(cmd, env=env)
                        elif os.path.isfile(os.path.join(output_dir, fsid, "label/BA_exvivo.thresh.ctab")):
                            print("SUBJECT ALREADY SEGMENTED, SKIPPING")
                        elif os.path.exists(os.path.join(output_dir, fsid)):
                            print("SUBJECT DIR ALREADY EXISTS (without IsRunning.lh+rh), RUNNING COMMAND:")
                            print(resume_cmd)
                            run(resume_cmd, env=env)
                        else:
                            print(cmd)
                            run(cmd, env=env)

                if 'template' in args.steps:
                    # creating a subject specific template
                    input_args = " ".join(["-tp %s" % tp for tp in timepoints])
                    fsid = "sub-%s" % subject_label
                    stages = " ".join(["-" + stage for stage in args.stages])

                    cmd = "recon-all -base %s -sd %s %s %s -openmp %d" % (fsid,
                                                                          output_dir,
                                                                          input_args,
                                                                          stages,
                                                                          args.n_cpus)

                    if os.path.isfile(os.path.join(output_dir, fsid, "scripts/IsRunning.lh+rh")):
                        rmtree(os.path.join(output_dir, fsid))
                        print("DELETING OUTPUT SUBJECT DIR AND RE-RUNNING COMMAND:")
                        print(cmd)
                        run(cmd, env=env)
                    elif os.path.isfile(os.path.join(output_dir, fsid, "label/BA_exvivo.thresh.ctab")):
                        print("TEMPLATE ALREADY CREATED, SKIPPING")
                    elif os.path.exists(os.path.join(output_dir, fsid)):
                        print("SUBJECT DIR ALREADY EXISTS (without IsRunning.lh+rh), RUNNING COMMAND:")
                        print(cmd)
                        run(cmd, env=env)
                    else:
                        print(cmd)
                        run(cmd, env=env)

                if 'longitudinal' in args.steps:
                    for tp in timepoints:
                        # longitudinally process all timepoints
                        fsid = "sub-%s" % subject_label
                        stages = " ".join(["-" + stage for stage in args.stages])

                        cmd = "recon-all -long %s %s -sd %s %s -openmp %d" % (tp,
                                                                              fsid,
                                                                              output_dir,
                                                                              stages,
                                                                              args.n_cpus)

                        if os.path.isfile(os.path.join(output_dir, tp + ".long." + fsid, "scripts/IsRunning.lh+rh")):
                            rmtree(os.path.join(output_dir, tp + ".long." + fsid))
                            print("DELETING OUTPUT SUBJECT DIR AND RE-RUNNING COMMAND:")
                            print(cmd)
                            run(cmd, env=env)
                        elif os.path.isfile(os.path.join(output_dir, tp + ".long." + fsid, "label/BA_exvivo.thresh.ctab")):
                            print("SUBJECT ALREADY SEGMENTED, SKIPPING")
                        else:
                            print(cmd)
                            run(cmd, env=env)

            elif len(sessions) > 0 and longitudinal_study is False:
                # grab all T1s/T2s from multiple sessions and combine
                T1s = glob(os.path.join(args.bids_dir,
                                        "sub-%s" % subject_label,
                                        "ses-*",
                                        "anat",
                                        "%s_T1w.nii*" % (ar_tpl)))
                input_args = ""

                if three_T == 'true':
                    input_args += " -3T"

                if args.qcache:
                    input_args += " -qcache"

                for T1 in T1s:
                    if (round(max(nibabel.load(T1).header.get_zooms()), 1) < 1.0 and args.hires_mode == "auto") or args.hires_mode == "enable":
                        input_args += " -hires"
                    input_args += " -i %s" % T1

                T2s = glob(os.path.join(args.bids_dir,
                                        "sub-%s" % subject_label,
                                        "ses-*",
                                        "anat",
                                        "%s_T2w.nii*" % (ar_t2)))
                FLAIRs = glob(os.path.join(args.bids_dir,
                                           "sub-%s" % subject_label,
                                           "ses-*",
                                           "anat",
                                           "%s_FLAIR.nii*" % (ar_t2)))
                if args.refine_pial == "T2":
                    for T2 in T2s:
                        if (max(nibabel.load(T2).header.get_zooms()) < 1.2) | args.allow_lowresT2:
                            input_args += " " + " ".join(["-T2 %s" % T2])
                            input_args += " -T2pial"
                elif args.refine_pial == "FLAIR":
                    for FLAIR in FLAIRs:
                        if (max(nibabel.load(FLAIR).header.get_zooms()) < 1.2) | args.allow_lowresT2:
                            input_args += " " + " ".join(["-FLAIR %s" % FLAIR])
                            input_args += " -FLAIRpial"

                fsid = "sub-%s" % subject_label
                stages = " ".join(["-" + stage for stage in args.stages])

                cmd = "recon-all -subjid %s -sd %s %s %s -openmp %d" % (fsid,
                                                                        output_dir,
                                                                        input_args,
                                                                        stages,
                                                                        args.n_cpus)
                resume_cmd = "recon-all -subjid %s -sd %s %s -openmp %d" % (fsid,
                                                                            output_dir,
                                                                            stages,
                                                                            args.n_cpus)

                if os.path.isfile(os.path.join(output_dir, fsid, "scripts/IsRunning.lh+rh")):
                    rmtree(os.path.join(output_dir, fsid))
                    print("DELETING OUTPUT SUBJECT DIR AND RE-RUNNING COMMAND:")
                    print(cmd)
                    run(cmd, env=env)
                elif os.path.isfile(os.path.join(output_dir, fsid, "label/BA_exvivo.thresh.ctab")):
                    print("SUBJECT ALREADY SEGMENTED, SKIPPING")
                elif os.path.exists(os.path.join(output_dir, fsid)):
                    print("SUBJECT DIR ALREADY EXISTS (without IsRunning.lh+rh), RUNNING COMMAND:")
                    print(resume_cmd)
                    run(resume_cmd, env=env)
                else:
                    print(cmd)
                    run(cmd, env=env)
            else:
                print("SKIPPING SUBJECT %s (no valid session)." % subject_label)

        else:
            # grab all T1s/T2s from single session (no ses-* directories)
            T1s = glob(os.path.join(args.bids_dir,
                                    "sub-%s" % subject_label,
                                    "anat",
                                    "%s_T1w.nii*" % (ar_tpl)))
            if not T1s:
                print("No T1w nii files found for subject %s. Skipping subject." % subject_label)
                continue

            input_args = ""

            if three_T == 'true':
                input_args += " -3T"

            if args.qcache:
                input_args += " -qcache"

            for T1 in T1s:
                if (round(max(nibabel.load(T1).header.get_zooms()), 1) < 1.0 and args.hires_mode == "auto") or args.hires_mode == "enable":
                    input_args += " -hires"
                input_args += " -i %s" % T1
            T2s = glob(os.path.join(args.bids_dir, "sub-%s" % subject_label, "anat",
                                    "%s_T2w.nii*" % (ar_t2)))
            FLAIRs = glob(os.path.join(args.bids_dir, "sub-%s" % subject_label, "anat",
                                       "%s_FLAIR.nii*" % (ar_t2)))
            if args.refine_pial == "T2":
                for T2 in T2s:
                    if max(nibabel.load(T2).header.get_zooms()) < 1.2:
                        input_args += " " + " ".join(["-T2 %s" % T2])
                        input_args += " -T2pial"
            elif args.refine_pial == "FLAIR":
                for FLAIR in FLAIRs:
                    if max(nibabel.load(FLAIR).header.get_zooms()) < 1.2:
                        input_args += " " + " ".join(["-FLAIR %s" % FLAIR])
                        input_args += " -FLAIRpial"

            fsid = "sub-%s" % subject_label
            stages = " ".join(["-" + stage for stage in args.stages])

            cmd = "recon-all -subjid %s -sd %s %s %s -openmp %d" % (fsid,
                                                                    output_dir,
                                                                    input_args,
                                                                    stages,
                                                                    args.n_cpus)
            resume_cmd = "recon-all -subjid %s -sd %s %s -openmp %d" % (fsid,
                                                                        output_dir,
                                                                        stages,
                                                                        args.n_cpus)

            if os.path.isfile(os.path.join(output_dir, fsid, "scripts/IsRunning.lh+rh")):
                rmtree(os.path.join(output_dir, fsid))
                print("DELETING OUTPUT SUBJECT DIR AND RE-RUNNING COMMAND:")
                print(cmd)
                run(cmd, env=env)
            elif os.path.isfile(os.path.join(output_dir, fsid, "label/BA_exvivo.thresh.ctab")):
                print("SUBJECT ALREADY SEGMENTED, SKIPPING")
            elif os.path.exists(os.path.join(output_dir, fsid)):
                print("SUBJECT DIR ALREADY EXISTS (without IsRunning.lh+rh), RUNNING COMMAND:")
                print(resume_cmd)
                run(resume_cmd, env=env)
            else:
                print(cmd)
                run(cmd, env=env)

elif args.analysis_level == "group1":        # running group level
    if len(subjects_to_analyze) > 1:
        # generate study specific template
        fsids = ["sub-%s" % s for s in subjects_to_analyze]
        # skipping volumetric average due to https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg51822.html
        cmd = "make_average_subject --no-symlink --no-vol --out " + args.template_name + " --subjects " + " ".join(fsids)
        print(cmd)
        if os.path.exists(os.path.join(output_dir, args.template_name)):
            rmtree(os.path.join(output_dir, args.template_name))
        run(cmd, env={"SUBJECTS_DIR": output_dir, 'FS_LICENSE': args.license_file})
        for subject_label in subjects_to_analyze:
            for hemi in ["lh", "rh"]:
                tif_file = os.path.join(output_dir, args.template_name, hemi+".reg.template.tif")
                fsid = "sub-%s" % subject_label
                sphere_file = os.path.join(output_dir, fsid, "surf", hemi+".sphere")
                reg_file = os.path.join(output_dir, fsid, "surf", hemi+".sphere.reg." + args.template_name)
                cmd = "mris_register -curv %s %s %s" % (sphere_file, tif_file, reg_file)
                run(cmd, env={"SUBJECTS_DIR": output_dir, 'FS_LICENSE': args.license_file})
    else:
        print("Only one subject included in the analysis. Skipping group1 level")


elif args.analysis_level == "group2":  # running stats tables
    table_dir = os.path.join(output_dir, "00_group2_stats_tables")
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
    print("Writing stats tables to %s." % table_dir)

    # To make the group analysis independent of participant_level --multiple_sessions option, we are looking for
    # *long* folders in the output_dir. If there exists one, we assume the study is longitudinal and we only
    # consider *long* freesurfer folders. Else we search for sub-<subject_label> freesurfer folders. If subjects
    #  cannot be found in freesurfer folder, an exception is raised.
    subjects = []
    if glob(os.path.join(output_dir, "sub-*_ses-*.long.sub-*")):
        for s in subjects_to_analyze:
            fs_sessions = sorted(glob(os.path.join(output_dir, "sub-{s}_ses-*.long.sub-{s}*".format(s=s))))
            if fs_sessions:
                subjects += [os.path.basename(fssub) for fssub in fs_sessions]
            else:
                raise Exception("No freesurfer sessions found for %s in %s" % (s, output_dir))
    else:
        for s in subjects_to_analyze:
            if os.path.isdir(os.path.join(output_dir, "sub-" + s)):
                subjects.append("sub-" + s)
            else:
                raise Exception("No freesurfer subject found for %s in %s" % (s, output_dir))
    subjects_str = " ".join(subjects)

    if len(subjects) > 0:
        # create cortical stats
        for p in args.parcellations:
            for h in ["lh", "rh"]:
                for m in args.measurements:
                    table_file = os.path.join(table_dir, "{h}.{p}.{m}.tsv".format(h=h, p=p, m=m))
                    if os.path.isfile(table_file):
                        warn("Replace old file %s" % table_file)
                        os.remove(table_file)
                    cmd = "python2 `which aparcstats2table` --hemi {h} --subjects {subjects} --parc {p} --meas {m} " \
                          "--tablefile {table_file}".format(h=h, subjects=subjects_str, p=p, m=m,
                                                            table_file=table_file)
                    print("Creating cortical stats table for {h} {p} {m}".format(h=h, p=p, m=m))
                    run(cmd, env={"SUBJECTS_DIR": output_dir, 'FS_LICENSE': args.license_file})

        # create subcortical stats
        table_file = os.path.join(table_dir, "aseg.tsv")
        if os.path.isfile(table_file):
            warn("Replace old file %s" % table_file)
            os.remove(table_file)
        cmd = "python2 `which asegstats2table` --subjects {subjects} --meas volume --tablefile {" \
              "table_file}".format(subjects=subjects_str, table_file=table_file)
        print("Creating subcortical stats table.")
        run(cmd, env={"SUBJECTS_DIR": output_dir, 'FS_LICENSE': args.license_file})

        print("\nTable export finished for %d subjects/sessions." % len(subjects))

    else:
        print("\nNo subjects included in the analysis. Skipping group2 level stats tables.")


    # This extracts the euler numbers for the orig.nofix surfaces from the recon-all.log file
    # see Rosen et al. (2017), https://www.biorxiv.org/content/early/2017/10/01/125161
    def extract_euler(logfile):
        with open(logfile) as fi:
            logtext = fi.read()
        p = re.compile(r"orig.nofix lheno =\s+(-?\d+), rheno =\s+(-?\d+)")
        results = p.findall(logtext)
        if len(results) != 1:
            raise Exception("Euler number could not be extracted from {}".format(logfile))
        lh_euler, rh_euler = results[0]
        return int(lh_euler), int(rh_euler)



    euler_out_file = os.path.join(table_dir, "euler.tsv")
    print("Writing euler tables to %s." % euler_out_file)

    # get freesurfer subjects
    os.chdir(output_dir)
    subjects = []
    for s in subjects_to_analyze:
        subjects += glob("sub-{}*".format(s))
    # remove long subjects as they don't have orig.nofix surfaces,
    # therefore no euler numbers
    subjects = list(filter(lambda s: ".long.sub-" not in s, subjects))
    if len(subjects) > 0:
        df = pd.DataFrame([], columns=["subject", "lh_euler", "rh_euler"])
        for subject in subjects:
            logfile = os.path.join(output_dir, subject, "scripts/recon-all.log")
            lh_euler, rh_euler = extract_euler(logfile)
            df_subject = pd.DataFrame({"subject": [subject],
                                        "lh_euler": [lh_euler],
                                        "rh_euler": [rh_euler]},
                                        columns=["subject", "lh_euler", "rh_euler"])
            df = df.append(df_subject)
        df["mean_euler_bh"] = df[["lh_euler", "rh_euler"]].mean(1)
        df.sort_values("subject", inplace=True)
        df.to_csv(euler_out_file, sep="\t", index=False)
    else:
        print("\nNo subjects included in the analysis. Skipping group2 level euler number table.")
