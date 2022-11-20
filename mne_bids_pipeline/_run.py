"""Script-running utilities."""

import copy
import functools
import hashlib
import inspect
import os
import pathlib
import pdb
import sys
import traceback
import time
from typing import Callable, Optional, Dict, List
from types import SimpleNamespace
import warnings

from joblib import Memory
import json_tricks
from openpyxl import load_workbook
import pandas as pd
from mne_bids import BIDSPath

from ._config_utils import get_task
from ._logging import logger, gen_log_kwargs


def failsafe_run(
    get_input_fnames: Optional[Callable] = None,
    get_output_fnames: Optional[Callable] = None,
) -> Callable:
    def failsafe_run_decorator(func):
        @functools.wraps(func)  # Preserve "identity" of original function
        def wrapper(*args, **kwargs):
            exec_params = kwargs['exec_params']
            on_error = exec_params.on_error
            memory = ConditionalStepMemory(
                exec_params=exec_params,
                get_input_fnames=get_input_fnames,
                get_output_fnames=get_output_fnames,
            )
            kwargs_copy = copy.deepcopy(kwargs)
            t0 = time.time()
            kwargs_copy["cfg"] = json_tricks.dumps(
                kwargs_copy["cfg"], sort_keys=False, indent=4
            )
            log_info = pd.concat([
                pd.Series(kwargs_copy, dtype=object),
                pd.Series(index=['time', 'success', 'error_message'],
                          dtype=object)
            ])

            try:
                assert len(args) == 0, args  # make sure params are only kwargs
                out = memory.cache(func)(*args, **kwargs)
                assert out is None  # nothing should be returned
                log_info['success'] = True
                log_info['error_message'] = ''
            except Exception as e:
                # Only keep what gen_log_kwargs() can handle
                kwargs_copy = {
                    k: v for k, v in kwargs_copy.items()
                    if k in ('subject', 'session', 'task', 'run')
                }
                message = (
                    f'A critical error occurred. '
                    f'The error message was: {str(e)}'
                )
                log_info['success'] = False
                log_info['error_message'] = str(e)

                if on_error == 'abort':
                    message += (
                        '\n\nAborting pipeline run. The full traceback '
                        'is:\n\n'
                    )
                    if sys.version_info >= (3, 10):
                        message += '\n'.join(
                            traceback.format_exception(e)
                        )
                    else:
                        message += '\n'.join(
                            traceback.format_exception(
                                etype=type(e),
                                value=e,
                                tb=e.__traceback__
                            )
                        )
                    if os.getenv('_MNE_BIDS_STUDY_TESTING', '') == 'true':
                        raise
                    logger.critical(**gen_log_kwargs(
                        message=message, **kwargs_copy, emoji='❌'
                    ))
                    sys.exit(1)
                elif on_error == 'debug':
                    message += '\n\nStarting post-mortem debugger.'
                    logger.critical(**gen_log_kwargs(
                        message=message, **kwargs_copy, emoji='🐛'
                    ))
                    extype, value, tb = sys.exc_info()
                    traceback.print_exc()
                    pdb.post_mortem(tb)
                    sys.exit(1)
                else:
                    message += '\n\nContinuing pipeline run.'
                    logger.critical(**gen_log_kwargs(
                        message=message, **kwargs_copy, emoji='🔂'
                    ))
            log_info['time'] = round(time.time() - t0, ndigits=1)
            return log_info
        return wrapper
    return failsafe_run_decorator


def hash_file_path(path: pathlib.Path) -> str:
    with open(path, 'rb') as f:
        md5_hash = hashlib.md5(f.read())
        md5_hashed = md5_hash.hexdigest()
    return md5_hashed


class ConditionalStepMemory:
    def __init__(self, *, exec_params, get_input_fnames, get_output_fnames):
        memory_location = exec_params.memory_location
        if memory_location is True:
            use_location = exec_params.deriv_root / 'joblib'
        elif not memory_location:
            use_location = None
        else:
            use_location = pathlib.Path(memory_location)
        # Actually make the Memory object only if necessary
        if use_location is not None and get_input_fnames is not None:
            self.memory = Memory(
                use_location, verbose=exec_params.memory_verbose)
        else:
            self.memory = None
        # Ignore these as they have no effect on the output
        self.ignore = ['exec_params']
        self.get_input_fnames = get_input_fnames
        self.get_output_fnames = get_output_fnames
        self.memory_file_method = exec_params.memory_file_method

    def cache(self, func):

        def wrapper(*args, **kwargs):
            in_files = out_files = None
            force_run = kwargs.pop('force_run', False)
            these_kwargs = kwargs.copy()
            these_kwargs.pop('exec_params', None)
            if self.get_output_fnames is not None:
                out_files = self.get_output_fnames(
                    **these_kwargs)
            if self.get_input_fnames is not None:
                in_files = kwargs['in_files'] = self.get_input_fnames(
                    **these_kwargs)
            del these_kwargs
            if self.memory is None:
                func(*args, **kwargs)
                return

            # This is an implementation detail so we don't need a proper error
            assert isinstance(in_files, dict), type(in_files)

            # Deal with cases (e.g., custom cov) where input files are unknown
            unknown_inputs = in_files.pop('__unknown_inputs__', False)
            # If this is ever true, we'll need to improve the logic below
            assert not (unknown_inputs and force_run)

            def hash_(k, v):
                if isinstance(v, BIDSPath):
                    v = v.fpath
                assert isinstance(v, pathlib.Path), \
                    f'Bad type {type(v)}: in_files["{k}"] = {v}'
                assert v.exists(), f'missing in_files["{k}"] = {v}'
                if self.memory_file_method == 'mtime':
                    this_hash = v.lstat().st_mtime
                else:
                    assert self.memory_file_method == 'hash'  # guaranteed
                    this_hash = hash_file_path(v)
                return (str(v), this_hash)

            hashes = []
            for k, v in in_files.items():
                hashes.append(hash_(k, v))
                # also hash the sidecar files if this is a BIDSPath and
                # MNE-BIDS is new enough
                if not hasattr(v, 'find_matching_sidecar'):
                    continue
                # from mne_bids/read.py
                # The v.datatype is maybe not right, might need to use
                # _infer_datatype like in read.py...
                for suffix, extension in (('events', '.tsv'),
                                          ('channels', '.tsv'),
                                          ('electrodes', '.tsv'),
                                          ('coordsystem', '.json'),
                                          (v.datatype, '.json')):
                    sidecar = v.find_matching_sidecar(
                        suffix=suffix, extension=extension, on_error='ignore')
                    if sidecar is None:
                        continue
                    hashes.append(hash_(k, sidecar))

            kwargs['cfg'] = copy.deepcopy(kwargs['cfg'])
            kwargs['cfg'].hashes = hashes
            del in_files  # will be modified by func call

            # Someday we could modify the joblib API to combine this with the
            # call (https://github.com/joblib/joblib/issues/1342), but our hash
            # should be plenty fast so let's not bother for now.
            memorized_func = self.memory.cache(func, ignore=self.ignore)
            msg = emoji = None
            short_circuit = False
            subject = kwargs.get('subject', None)
            session = kwargs.get('session', None)
            run = kwargs.get('run', None)
            try:
                done = memorized_func.check_call_in_cache(*args, **kwargs)
            except Exception:
                done = False
            if done:
                if unknown_inputs:
                    msg = ('Computation forced because input files cannot '
                           f'be determined ({unknown_inputs}) …')
                    emoji = '🤷'
                elif force_run:
                    msg = 'Computation forced despite existing cached result …'
                    emoji = '🔂'
                else:
                    msg = 'Computation unnecessary (cached) …'
                    emoji = 'cache'
            # When out_files is not None, we should check if the output files
            # exist and stop if they do (e.g., in bem surface or coreg surface
            # creation)
            elif out_files is not None:
                have_all = all(path.exists() for path in out_files.values())
                if not have_all:
                    msg = 'Output files missing, will recompute …'
                    emoji = '🧩'
                elif force_run:
                    msg = 'Computation forced despite existing output files …'
                    emoji = '🔂'
                else:
                    msg = 'Computation unnecessary (output files exist) …'
                    emoji = '🔍'
                    short_circuit = True
            if msg is not None:
                step = _short_step_path(pathlib.Path(inspect.getfile(func)))
                logger.info(**gen_log_kwargs(
                    message=msg, subject=subject, session=session, run=run,
                    emoji=emoji, step=step))
            if short_circuit:
                return

            # https://joblib.readthedocs.io/en/latest/memory.html#joblib.memory.MemorizedFunc.call  # noqa: E501
            if force_run or unknown_inputs:
                out_files, _ = memorized_func.call(*args, **kwargs)
            else:
                out_files = memorized_func(*args, **kwargs)
            assert isinstance(out_files, dict), type(out_files)
            out_files_missing_msg = '\n'.join(
                f'- {key}={fname}' for key, fname in out_files.items()
                if not pathlib.Path(fname).exists()
            )
            if out_files_missing_msg:
                raise ValueError('Missing at least one output file: \n'
                                 + out_files_missing_msg + '\n' +
                                 'This should not happen unless some files '
                                 'have been manually moved or deleted. You '
                                 'need to flush your cache to fix this.')
        return wrapper

    def clear(self) -> None:
        self.memory.clear()


def save_logs(
    *,
    config: SimpleNamespace,
    logs  # TODO add type
) -> None:
    fname = config.deriv_root / f'task-{get_task(config)}_log.xlsx'

    # Get the script from which the function is called for logging
    sheet_name = _short_step_path(_get_step_path()).replace('/', '-')
    sheet_name = sheet_name[-30:]  # shorten due to limit of excel format

    df = pd.DataFrame(logs)

    columns = df.columns
    if "cfg" in columns:
        columns = list(columns)
        idx = columns.index("cfg")
        del columns[idx]
        columns.insert(-3, "cfg")  # put it before time, success & err cols

    df = df[columns]

    if fname.exists():
        book = None
        try:
            book = load_workbook(fname)
        except Exception:  # bad file
            pass
        else:
            if sheet_name in book:
                book.remove(book[sheet_name])
        writer = pd.ExcelWriter(fname, engine='openpyxl')
        if book is not None:
            try:
                writer.book = book
            except Exception:
                pass  # AttributeError: can't set attribute 'book' (?)
    else:
        writer = pd.ExcelWriter(fname, engine='openpyxl')

    df.to_excel(writer, sheet_name=sheet_name, index=False)
    # TODO: "FutureWarning: save is not part of the public API, usage can give
    # in unexpected results and will be removed in a future version"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        writer.save()
    writer.close()


def _update_for_splits(
    files_dict: Dict[str, BIDSPath],
    key: str,
    *,
    single: bool = False,
    allow_missing: bool = False
) -> BIDSPath:
    if not isinstance(files_dict, dict):  # fake it
        assert key is None
        files_dict, key = dict(x=files_dict), 'x'
    bids_path = files_dict[key]
    if bids_path.fpath.exists():
        return bids_path  # no modifications needed
    if bids_path.copy().update(run=None).fpath.exists():
        # Remove the run information
        return bids_path.copy().update(run=None)
    bids_path = bids_path.copy().update(split='01')
    missing = not bids_path.fpath.exists()
    if not allow_missing:
        assert not missing, f'Missing file: {bids_path.fpath}'
    if missing:
        return bids_path.update(split=None)
    files_dict[key] = bids_path
    # if we only need the first file (i.e., when reading), quit now
    if single:
        return bids_path
    for split in range(2, 100):
        split_key = f'{split:02d}'
        bids_path_next = bids_path.copy().update(split=split_key)
        if not bids_path_next.fpath.exists():
            break
        files_dict[f'{key}_split-{split_key}'] = bids_path_next
    return bids_path


def _sanitize_callable(val):
    # Callables are not nicely pickleable, so let's pass a string instead
    if callable(val):
        return 'custom'
    else:
        return val


def _get_step_path(
    stack: Optional[List[inspect.FrameInfo]] = None,
) -> pathlib.Path:
    if stack is None:
        stack = inspect.stack()
    for frame in stack:
        fname = pathlib.Path(frame.filename)
        if 'steps' in fname.parts:
            return fname
    else:  # pragma: no cover
        raise RuntimeError('Could not find step path')


def _short_step_path(step_path: pathlib.Path) -> str:
    return f'{step_path.parent.name}/{step_path.stem}'
