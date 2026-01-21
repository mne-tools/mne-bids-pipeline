"""Script-running utilities."""

import copy
import functools
import hashlib
import inspect
import pathlib
import pdb
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any, Literal

import json_tricks
import pandas as pd
from filelock import FileLock
from joblib import Memory
from mne_bids import BIDSPath

from ._config_utils import get_task
from ._logging import _is_testing, gen_log_kwargs, logger
from .typing import InFilesPathT, InFilesT, OutFilesT


def failsafe_run(
    *,
    get_input_fnames: Callable[..., Any] | None = None,
    get_output_fnames: Callable[..., Any] | None = None,
    require_output: bool = True,
) -> Callable[..., Any]:
    def failsafe_run_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)  # Preserve "identity" of original function
        def __mne_bids_pipeline_failsafe_wrapper__(*args, **kwargs):  # type: ignore
            __mne_bids_pipeline_step__ = pathlib.Path(inspect.getfile(func))  # noqa
            exec_params = kwargs["exec_params"]
            on_error = exec_params.on_error
            memory = ConditionalStepMemory(
                exec_params=exec_params,
                get_input_fnames=get_input_fnames,
                get_output_fnames=get_output_fnames,
                require_output=require_output,
                func_name=f"{__mne_bids_pipeline_step__}::{func.__name__}",
            )
            t0 = time.time()
            log_info = pd.concat(
                [
                    pd.Series(kwargs, dtype=object),
                    pd.Series(index=["time", "success", "error_message"], dtype=object),
                ]
            )

            try:
                assert len(args) == 0, args  # make sure params are only kwargs
                out = memory.cache(func)(*args, **kwargs)
                assert out is None  # nothing should be returned
                log_info["success"] = True
                log_info["error_message"] = ""
            except Exception as e:
                # Only keep what gen_log_kwargs() can handle
                kwargs_log = {
                    k: kwargs[k]
                    for k in ("subject", "session", "task", "run")
                    if k in kwargs
                }
                message = f"A critical error occurred. The error message was: {str(e)}"
                log_info["success"] = False
                log_info["error_message"] = str(e)

                # Find the limit / step where the error occurred
                step_dir = pathlib.Path(__file__).parent / "steps"
                tb_list = list(traceback.extract_tb(e.__traceback__))
                for fi, frame in enumerate(tb_list):
                    is_step = pathlib.Path(frame.filename).parent.parent == step_dir
                    del frame
                    if is_step:
                        # omit everything before the "step" dir, which will
                        # generally be stuff from this file and joblib
                        tb_list = tb_list[fi:]
                        break
                tb = "".join(traceback.format_list(tb_list))

                if on_error == "abort":
                    message += f"\n\nAborting pipeline run. The traceback is:\n\n{tb}"

                    if _is_testing():
                        raise
                    logger.error(
                        **gen_log_kwargs(message=message, **kwargs_log, emoji="❌")
                    )
                    sys.exit(1)
                elif on_error == "debug":
                    message += "\n\nStarting post-mortem debugger."
                    logger.error(
                        **gen_log_kwargs(message=message, **kwargs_log, emoji="🐛")
                    )
                    _, _, tb_ = sys.exc_info()
                    print(tb)
                    pdb.post_mortem(tb_)
                    sys.exit(1)
                else:
                    message += "\n\nContinuing pipeline run."
                    logger.error(
                        **gen_log_kwargs(message=message, **kwargs_log, emoji="🔂")
                    )
            log_info["time"] = round(time.time() - t0, ndigits=1)
            return log_info

        return __mne_bids_pipeline_failsafe_wrapper__

    return failsafe_run_decorator


def hash_file_path(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        md5_hash = hashlib.md5(f.read())
        md5_hashed = md5_hash.hexdigest()
    return md5_hashed


class ConditionalStepMemory:
    def __init__(
        self,
        *,
        exec_params: SimpleNamespace,
        get_input_fnames: Callable[..., Any] | None,
        get_output_fnames: Callable[..., Any] | None,
        require_output: bool,
        func_name: str,
    ) -> None:
        memory_location = exec_params.memory_location
        if memory_location is True:
            use_location = exec_params.deriv_root / exec_params.memory_subdir
        elif not memory_location:
            use_location = None
        else:
            use_location = pathlib.Path(memory_location)
        # Actually make the Memory object only if necessary
        if use_location is not None and get_input_fnames is not None:
            self.memory = Memory(use_location, verbose=exec_params.memory_verbose)
        else:
            self.memory = None
        # Ignore these as they have no effect on the output
        self.ignore = ["exec_params"]
        self.get_input_fnames = get_input_fnames
        self.get_output_fnames = get_output_fnames
        self.memory_file_method = exec_params.memory_file_method
        self.require_output = require_output
        self.func_name = func_name

    def cache(self, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> None:
            in_files = out_files = None
            force_run = kwargs.pop("force_run", False)
            these_kwargs = kwargs.copy()
            these_kwargs.pop("exec_params", None)
            if self.get_output_fnames is not None:
                out_files = self.get_output_fnames(**these_kwargs)
            if self.get_input_fnames is not None:
                in_files = kwargs["in_files"] = self.get_input_fnames(**these_kwargs)
            del these_kwargs
            if self.memory is None:
                func(*args, **kwargs)
                return

            # This is an implementation detail so we don't need a proper error
            assert isinstance(in_files, dict), type(in_files)

            # Deal with cases (e.g., custom cov) where input files are unknown
            unknown_inputs = in_files.pop("__unknown_inputs__", False)
            # If this is ever true, we'll need to improve the logic below
            assert not (unknown_inputs and force_run)

            hash_ = functools.partial(_path_to_str_hash, method=self.memory_file_method)
            hashes = []
            for k, v in in_files.items():
                hashes.append(hash_(k, v))
                # also hash the sidecar files if this is a BIDSPath and
                # MNE-BIDS is new enough
                if not hasattr(v, "find_matching_sidecar"):
                    continue
                # from mne_bids/read.py
                # The v.datatype is maybe not right, might need to use
                # _infer_datatype like in read.py...
                for suffix, extension in (
                    ("events", ".tsv"),
                    ("channels", ".tsv"),
                    ("electrodes", ".tsv"),
                    ("coordsystem", ".json"),
                    (v.datatype, ".json"),
                ):
                    sidecar = v.find_matching_sidecar(
                        suffix=suffix, extension=extension, on_error="ignore"
                    )
                    if sidecar is None:
                        continue
                    hashes.append(hash_(k, sidecar))

            kwargs["cfg"] = copy.deepcopy(kwargs["cfg"])
            assert isinstance(kwargs["cfg"], SimpleNamespace), type(kwargs["cfg"])
            # make sure we don't pass in a bare/complete `config`.
            # We should *always* limit it to what is needed for a given step, otherwise
            # unnecessary cache hits will occur, including those when command-line
            # arguments are changed (e.g., `--pdb`). If it's not limited, it will have
            # some entries like this, which we inject ourselves during config import:
            NON_SPECIFIC_CONFIG_KEY = "PIPELINE_NAME"
            assert NON_SPECIFIC_CONFIG_KEY not in kwargs["cfg"].__dict__, (
                "\nInternal error: cfg should be limited to step-specific entries only "
                f"for:\n\n{self.func_name}\n\nPlease report this to MNE-BIDS-Pipeline "
                "developers."
            )
            kwargs["cfg"].hashes = hashes
            del in_files  # will be modified by func call

            # Someday we could modify the joblib API to combine this with the
            # call (https://github.com/joblib/joblib/issues/1342), but our hash
            # should be plenty fast so let's not bother for now.
            memorized_func = self.memory.cache(func, ignore=self.ignore)
            msg: str | None = None
            emoji: str | None = None
            short_circuit = False
            # Used for logging automatically
            subject = kwargs.get("subject", None)  # noqa
            session = kwargs.get("session", None)  # noqa
            run = kwargs.get("run", None)  # noqa
            task = kwargs.get("task", None)  # noqa
            bad_out_files = False
            logger_call = logger.info
            try:
                done = memorized_func.check_call_in_cache(*args, **kwargs)
            except Exception as exc:
                done = False
                msg = f"Computation forced because of caching error: {exc}"
                emoji = "🤷"
            if done:
                if unknown_inputs:
                    msg = (
                        "Computation forced because input files cannot "
                        f"be determined ({unknown_inputs}) …"
                    )
                    emoji = "🤷"
                elif force_run:
                    msg = "Computation forced despite existing cached result …"
                    emoji = "🔂"
                else:
                    # Check our output file hashes
                    # Need to make a copy of kwargs["in_files"] in particular
                    use_kwargs = copy.deepcopy(kwargs)
                    out_files_hashes = memorized_func(*args, **use_kwargs)
                    for key, (fname, this_hash) in out_files_hashes.items():
                        fname = pathlib.Path(fname)
                        if not fname.exists():
                            msg = f"Output file missing, will recompute: {fname}"
                            emoji = "✖️"
                            bad_out_files = True
                            break
                        got_hash = hash_(key, fname, kind="out")[1]
                        if this_hash != got_hash:
                            msg = (
                                f"Output file {self.memory_file_method} mismatch "
                                f"({this_hash} != {got_hash}), will recompute: {fname}"
                            )
                            emoji = "🚫"
                            bad_out_files = True
                            break
                    else:
                        msg = "Computation unnecessary (cached) …"
                        emoji = "cache"
            # When out_files_expected is not None, we should check if the output files
            # exist and stop if they do (e.g., in bem surface or coreg surface
            # creation)
            elif out_files is not None:
                have_all = all(path.exists() for path in out_files.values())
                if not have_all:
                    msg = "Output files missing, will recompute …"
                    emoji = "🧩"
                elif force_run:
                    msg = "Computation forced despite existing output files …"
                    emoji = "🔂"
                else:
                    msg = "Computation unnecessary (output files exist) …"
                    emoji = "🔍"
                    short_circuit = True
            else:
                # Ensure memorized_func.check_call_in_cache returned False
                # as opposed to raised an error (which already sets `msg` above)
                if msg is None:
                    logger_call = logger.debug

                    msg = "Cached result not found, computing …"
                    emoji = "🆕"
            del out_files

            assert msg is not None
            assert emoji is not None
            logger_call(**gen_log_kwargs(message=msg, emoji=emoji))
            del logger_call
            if short_circuit:
                return

            # https://joblib.readthedocs.io/en/latest/memory.html#joblib.memory.MemorizedFunc.call  # noqa: E501
            if force_run or unknown_inputs or bad_out_files:
                # Joblib 1.4.0 only returns the output, but 1.3.2 returns both.
                # Fortunately we can use tuple-ness to tell the difference (we always
                # return None or a dict)
                out_files = memorized_func.call(*args, **kwargs)
                if isinstance(out_files, tuple):
                    out_files = out_files[0]
            else:
                out_files = memorized_func(*args, **kwargs)
            if self.require_output:
                assert isinstance(out_files, dict) and len(out_files), (
                    f"Internal error: step must return non-empty out_files dict, got "
                    f"{type(out_files).__name__} for:\n{self.func_name}"
                )
            else:
                assert out_files is None, (
                    f"Internal error: step must return None, got {type(out_files)} "
                    f"for:\n{self.func_name}"
                )

        return wrapper

    def clear(self) -> None:
        self.memory.clear()


def save_logs(*, config: SimpleNamespace, logs: Iterable[pd.Series]) -> None:
    fname = config.deriv_root / f"task-{get_task(config)}_log.xlsx"

    # Get the script from which the function is called for logging
    sheet_name = _short_step_path(_get_step_path()).replace("/", "-")
    sheet_name = sheet_name[-30:]  # shorten due to limit of excel format

    df = pd.DataFrame(logs)
    del logs

    with FileLock(fname.with_suffix(fname.suffix + ".lock")):
        append = fname.exists()
        writer = pd.ExcelWriter(
            fname,
            engine="openpyxl",
            mode="a" if append else "w",
            if_sheet_exists="replace" if append else None,
        )
        assert isinstance(config, SimpleNamespace), type(config)
        cf_dict = dict()
        for key, val in config.__dict__.items():
            # We need to be careful about functions, json_tricks does not work with them
            if inspect.isfunction(val):
                new_val = ""
                if func_file := inspect.getfile(val):
                    new_val += f"{func_file}:"
                if getattr(val, "__qualname__", None):
                    new_val += val.__qualname__
                val = "custom callable" if not new_val else new_val
            val = json_tricks.dumps(val, indent=4, sort_keys=False)
            # 32767 char limit per cell (could split over lines but if something is
            # this long, you'll probably get the gist from the first 32k chars)
            if len(val) > 32767:
                val = val[:32765] + " …"
            cf_dict[key] = val
        cf_df = pd.DataFrame([cf_dict], dtype=object)
        with writer:
            # Config first then the data
            cf_df.to_excel(writer, sheet_name="config", index=False)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def _update_for_splits(
    files_dict: InFilesT | BIDSPath,
    key: str | None,
    *,
    single: bool = False,
    allow_missing: bool = False,
) -> BIDSPath:
    if not isinstance(files_dict, dict):  # fake it
        assert key is None
        files_dict, key = dict(x=files_dict), "x"
    assert isinstance(key, str), type(key)
    bids_path = files_dict[key]
    if bids_path.fpath.exists():
        return bids_path  # no modifications needed
    if bids_path.copy().update(run=None).fpath.exists():
        # Remove the run information
        return bids_path.copy().update(run=None)
    bids_path = bids_path.copy().update(split="01")
    missing = not bids_path.fpath.exists()
    if not allow_missing:
        assert not missing, f"Missing file: {bids_path.fpath}"
    if missing:
        return bids_path.update(split=None)
    files_dict[key] = bids_path
    # if we only need the first file (i.e., when reading), quit now
    if single:
        return bids_path
    for split in range(2, 100):
        split_key = f"{split:02d}"
        bids_path_next = bids_path.copy().update(split=split_key)
        if not bids_path_next.fpath.exists():
            break
        files_dict[f"{key}_split-{split_key}"] = bids_path_next
    return bids_path


def _sanitize_callable(val: Any) -> Any:
    # Callables are not nicely pickleable, so let's pass a string instead
    if callable(val):
        return "custom"
    else:
        return val


def _get_step_path(
    stack: list[inspect.FrameInfo] | None = None,
) -> pathlib.Path:
    if stack is None:
        stack = inspect.stack()
    paths: list[str] = list()
    for frame in stack:
        fname = pathlib.Path(frame.filename)
        paths.append(frame.filename)
        if "steps" in fname.parts:
            return fname
        else:  # pragma: no cover
            try:
                out = frame.frame.f_locals["__mne_bids_pipeline_step__"]
            except KeyError:
                pass
            else:
                assert isinstance(out, pathlib.Path)
                return out
    else:  # pragma: no cover
        paths_str = "\n".join(paths)
        raise RuntimeError(f"Could not find step path in call stack:\n{paths_str}")


def _short_step_path(step_path: pathlib.Path) -> str:
    return f"{step_path.parent.name}/{step_path.stem}"


def _prep_out_files(
    *,
    exec_params: SimpleNamespace,
    out_files: InFilesT,
    check_relative: pathlib.Path | None = None,
) -> OutFilesT:
    for key, fname in out_files.items():
        assert isinstance(fname, BIDSPath), (
            f'out_files["{key}"] must be a BIDSPath, got {type(fname)}'
        )
        if fname.suffix not in ("raw", "epo"):
            assert fname.split is None, fname
    return _prep_out_files_path(
        exec_params=exec_params,
        out_files=out_files,
        check_relative=check_relative,
    )


def _prep_out_files_path(
    *,
    exec_params: SimpleNamespace,
    out_files: InFilesPathT,
    check_relative: pathlib.Path | None = None,
) -> OutFilesT:
    if check_relative is None:
        check_relative = exec_params.deriv_root
    for key, fname in out_files.items():
        # Sanity check that we only ever write to the derivatives directory
        # raw and epochs can split on write, and .save should check for us now, so
        # we only need to check *other* types (these should never split)
        fname = pathlib.Path(fname)
        if not fname.is_relative_to(check_relative):
            raise RuntimeError(
                f"Output BIDSPath not relative to expected root {check_relative}:"
                f"\n{fname}"
            )
        out_files[key] = _path_to_str_hash(
            key,
            fname,
            method=exec_params.memory_file_method,
            kind="out",
        )
    return out_files


def _path_to_str_hash(
    k: str,
    v: BIDSPath | pathlib.Path,
    *,
    method: Literal["mtime", "hash"],
    kind: str = "in",
) -> tuple[str, str | float]:
    if isinstance(v, BIDSPath):
        v = v.fpath
    assert isinstance(v, pathlib.Path), f'Bad type {type(v)}: {kind}_files["{k}"] = {v}'
    assert v.exists(), f'missing {kind}_files["{k}"] = {v}'
    this_hash: str | float = ""
    if method == "mtime":
        this_hash = v.stat().st_mtime
    else:
        assert method == "hash"  # guaranteed
        this_hash = hash_file_path(v)
    return (str(v), this_hash)
