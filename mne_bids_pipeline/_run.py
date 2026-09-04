"""Script-running utilities."""

import contextlib
import copy
import datetime
import functools
import hashlib
import inspect
import pathlib
import pdb
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Iterable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from filelock import FileLock
from joblib import Memory
from mne_bids import BIDSPath

from ._config_utils import _get_step_title
from ._flow import FlowEntryT, _write_flow_entry
from ._io import _LOCK_TIMEOUT
from ._logging import _is_testing, gen_log_kwargs, logger
from .typing import InFilesPathT, InFilesT, OutFilesT

if TYPE_CHECKING:
    import pandas as pd


def failsafe_run(
    *,
    get_input_fnames: Callable[..., Any] | None = None,
    get_output_fnames: Callable[..., Any] | None = None,
    require_output: bool = True,
    sidecars: bool = False,
) -> Callable[..., Any]:
    def failsafe_run_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)  # Preserve "identity" of original function
        def __mne_bids_pipeline_failsafe_wrapper__(
            *args: list[Any], **kwargs: dict[str, Any]
        ) -> "pd.Series | None":
            __mne_bids_pipeline_step__ = pathlib.Path(inspect.getfile(func))  # noqa
            exec_params = kwargs["exec_params"]
            on_error = exec_params.on_error
            memory = ConditionalStepMemory(
                exec_params=exec_params,
                get_input_fnames=get_input_fnames,
                get_output_fnames=get_output_fnames,
                require_output=require_output,
                func_name=f"{__mne_bids_pipeline_step__}::{func.__name__}",
                sidecars=sidecars,
            )
            t0 = time.time()

            success = True
            error_message = ""
            did_run = True
            try:
                assert len(args) == 0, args  # make sure params are only kwargs
                did_run = memory.cache(func)(*args, **kwargs)
                assert isinstance(did_run, bool)  # whether or not it ran
            except Exception as e:
                # Only keep what gen_log_kwargs() can handle
                kwargs_log = {
                    k: kwargs[k]
                    for k in ("subject", "session", "task", "run")
                    if k in kwargs
                }
                e_str = "\n".join(traceback.format_exception_only(e)).strip()
                message = f"A critical error occurred. The error message was: {e_str}"
                success = False
                error_message = e_str

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
                tb = "".join(traceback.format_list(tb_list) + [e_str])

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
            if not did_run:
                return None  # no log info to return
            import pandas as pd

            log_info = pd.concat(
                [
                    pd.Series(kwargs, dtype=object),
                    pd.Series(index=["time", "success", "error_message"], dtype=object),
                ]
            )
            log_info["time"] = round(time.time() - t0, ndigits=1)
            log_info["success"] = success
            log_info["error_message"] = error_message
            return log_info

        return __mne_bids_pipeline_failsafe_wrapper__

    return failsafe_run_decorator


def hash_file_path(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        md5_hash = hashlib.md5(f.read())
        md5_hashed = md5_hash.hexdigest()
    return md5_hashed


# A step is decorated once but called once per (subject, session, run), and rebuilding
# these per call costs ~0.2 ms and re-does joblib's func_code introspection every time
@functools.cache
def _get_memory(location: pathlib.Path, verbose: int) -> Memory:
    return Memory(location, verbose=verbose)


@functools.cache
def _get_memorized_func(memory: Memory, func: Callable[..., Any]) -> Any:
    # exec_params has no effect on the output, so it must not affect the hash
    return memory.cache(func, ignore=["exec_params"])


class ConditionalStepMemory:
    def __init__(
        self,
        *,
        exec_params: SimpleNamespace,
        get_input_fnames: Callable[..., Any] | None,
        get_output_fnames: Callable[..., Any] | None,
        require_output: bool,
        func_name: str,
        sidecars: bool = False,
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
            self.memory = _get_memory(use_location, exec_params.memory_verbose)
        else:
            self.memory = None
        self.get_input_fnames = get_input_fnames
        self.get_output_fnames = get_output_fnames
        self.memory_file_method = exec_params.memory_file_method
        self.ignore_warnings = exec_params.ignore_warnings
        self.require_output = require_output
        self.func_name = func_name
        self.sidecars = sidecars
        self.deriv_root = exec_params.deriv_root

    def cache(self, func: Callable[..., Any]) -> Callable[..., Any]:
        step = _short_step_path(pathlib.Path(inspect.getfile(func)))
        func_name = func.__name__
        title = _get_step_title(inspect.getmodule(func))

        def __mbp_cached_func_wrapper__(
            *args: list[Any], **kwargs: dict[str, Any]
        ) -> bool:
            t0 = time.time()
            in_files = out_files = None
            force_run = kwargs.pop("force_run", False)
            these_kwargs = kwargs.copy()
            these_kwargs.pop("exec_params", None)
            if self.get_output_fnames is not None:
                out_files = self.get_output_fnames(**these_kwargs)
            if self.get_input_fnames is not None:
                in_files = kwargs["in_files"] = self.get_input_fnames(**these_kwargs)
            del these_kwargs
            # Snapshot now: the worker funcs pop entries off in_files as they go
            record = functools.partial(
                _record_flow,
                deriv_root=self.deriv_root,
                step=step,
                func_name=func_name,
                title=title,
                kwargs=kwargs,
                in_files=_flow_files(in_files),
                t0=t0,
            )
            # Steps write to the report before they return, so seed the recording with
            # the inputs now or the last step to run would be missing from its own
            # diagram until some later run rewrites the report.
            prior = record(out_files=None, only_if_new=True)
            # A completed prior entry means the disk already describes the original
            # computation, so cache hits below need no rewrite at all
            prior_done = prior.get("finished") is not None
            if self.memory is None:
                out_files = func(*args, **kwargs)
                record(out_files=out_files, cached=False)
                return True

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
                # also hash the sidecar files if this is a BIDSPath
                if not (self.sidecars and isinstance(v, BIDSPath)):
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
                    sidecar = _find_matching_sidecar_cached(
                        v, suffix=suffix, extension=extension
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
            if func.__name__ != "init_dataset":
                assert NON_SPECIFIC_CONFIG_KEY not in kwargs["cfg"].__dict__, (
                    "\nInternal error: cfg should be limited to step-specific entries "
                    f"only for:\n\n{self.func_name}\n\nPlease report this to "
                    "MNE-BIDS-Pipeline developers."
                )
            kwargs["cfg"].hashes = hashes
            del in_files  # will be modified by func call

            # The cache-hit path below reuses the result of the one call it makes, so
            # only check_call_in_cache still hashes the arguments a second time; folding
            # that in needs https://github.com/joblib/joblib/issues/1342.
            memorized_func = _get_memorized_func(self.memory, func)
            msg: str | None = None
            emoji: str | None = None
            short_circuit = False
            # Used for logging automatically
            subject = kwargs.get("subject", None)  # noqa
            session = kwargs.get("session", None)  # noqa
            run = kwargs.get("run", None)  # noqa
            task = kwargs.get("task", None)  # noqa
            bad_out_files = False
            cached_out_files = None  # the loaded cache entry, when it checks out
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
                    out_files_hashes = memorized_func(*args, **kwargs)
                    for key, (fname, this_hash) in out_files_hashes.items():
                        fname = pathlib.Path(fname)
                        if not fname.exists():
                            msg = f"Output file missing, will recompute: {fname}"
                            emoji = "✖️"
                            bad_out_files = True
                            break
                        if this_hash == "exists":  # existence-only, see _prep_out_files
                            continue
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
                        cached_out_files = out_files_hashes
                        msg = (
                            f"Computation unnecessary (cached "
                            f"{func.__name__}(…){_ran_when(prior)}) …"
                        )
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
                    short_circuit = True
                    if not prior_done:
                        # must record before the `del out_files` below
                        record(out_files=out_files, cached=True)
                    msg = (
                        "Computation unnecessary (output files exist"
                        f"{_ran_when(prior)}) …"
                    )
                    emoji = "🔍"
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
                return False  # did not run

            # https://joblib.readthedocs.io/en/latest/memory.html#joblib.memory.MemorizedFunc.call  # noqa: E501
            if force_run or unknown_inputs or bad_out_files:
                done = False
                with _ignore_warnings(self.ignore_warnings):
                    out_files, _ = memorized_func.call(*args, **kwargs)
            elif cached_out_files is not None:
                out_files = cached_out_files  # same call, already hashed and loaded
            else:
                with _ignore_warnings(self.ignore_warnings):
                    out_files = memorized_func(*args, **kwargs)
            if not (done and prior_done):  # cache hits with a full record skip the IO
                record(out_files=out_files, cached=done)
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
            return not done

        return __mbp_cached_func_wrapper__

    def clear(self) -> None:
        self.memory.clear()


def _ran_when(prior: Mapping[str, Any]) -> str:
    """Get ", ran <when>" when a call's original computation is on record."""
    if not prior.get("cached") and prior.get("finished"):
        return f", ran {prior['finished'][:16]}"
    return ""


def _flow_files(files: Mapping[str, object] | None) -> dict[str, str]:
    """Normalize an in_files/out_files mapping to plain path strings."""
    out: dict[str, str] = dict()
    for key, value in (files or dict()).items():
        if key == "__unknown_inputs__":
            continue
        if isinstance(value, tuple):  # out_files carry (path, hash) pairs
            value = value[0]
        if isinstance(value, BIDSPath):
            value = value.fpath
        if isinstance(value, str | pathlib.Path):
            out[key] = str(value)
    return out


def _record_flow(
    *,
    deriv_root: pathlib.Path,
    step: str,
    func_name: str,
    title: str | None,
    kwargs: dict[str, Any],
    in_files: dict[str, str],
    t0: float,
    out_files: Any,
    cached: bool | None = None,
    only_if_new: bool = False,
) -> Mapping[str, Any]:
    """Record a step call's files; get back the stored entry ({} if it failed)."""
    try:
        # cached is None for the pre-run seed entry, which has not completed
        duration = finished = None
        if cached is not None:
            duration = round(time.time() - t0, 3)
            finished = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        entry: FlowEntryT = {
            "step": step,
            "func": func_name,
            "title": title,
            "subject": kwargs.get("subject", None),
            "session": kwargs.get("session", None),
            "run": kwargs.get("run", None),
            "task": kwargs.get("task", None),
            "duration": duration,
            "finished": finished,
            "cached": cached,
            "in_files": in_files,
            "out_files": _flow_files(out_files),
        }
        # Roots let the report show paths as <bids_root>/... instead of absolute
        roots = {"deriv_root": str(deriv_root)}
        for name in ("bids_root", "fs_subjects_dir"):
            value = getattr(kwargs.get("cfg", None), name, None)
            if value is not None:
                roots[name] = str(value)
        return _write_flow_entry(
            deriv_root=deriv_root, entry=entry, roots=roots, only_if_new=only_if_new
        )
    except Exception as exc:
        msg = f"Could not record pipeline flow information: {exc}"
        logger.warning(**gen_log_kwargs(message=msg, emoji="⚠️"))
        return dict()


@contextlib.contextmanager
def _ignore_warnings(ignore_warnings: Iterable[str] | str) -> Iterable[None]:
    if isinstance(ignore_warnings, str):
        ignore_warnings = [ignore_warnings]
    with warnings.catch_warnings():
        for msg in ignore_warnings:
            warnings.filterwarnings("ignore", message=rf"[\S\s]*{msg}[\S\s]*")
        yield


def save_logs(*, config: SimpleNamespace, logs: "Iterable[pd.Series | None]") -> None:
    usable_logs = [log for log in logs if log is not None]
    if not usable_logs:
        return
    import json_tricks
    import pandas as pd

    all_tasks = "+".join(map(str, config.all_tasks))
    fname = config.deriv_root / f"task-{all_tasks}_log.xlsx"

    # Get the script from which the function is called for logging
    sheet_name = _short_step_path(_get_step_path()).replace("/", "-")
    sheet_name = sheet_name[-30:]  # shorten due to limit of excel format

    df = pd.DataFrame(usable_logs)
    del logs

    with FileLock(f"{fname}.lock", timeout=_LOCK_TIMEOUT):
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
    exist_only: tuple[str, ...] = (),
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
        exist_only=exist_only,
    )


def _prep_out_files_path(
    *,
    exec_params: SimpleNamespace,
    out_files: InFilesPathT,
    check_relative: pathlib.Path | None = None,
    exist_only: tuple[str, ...] = (),
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
        if key in exist_only:
            # freely rewritten by later steps (e.g. reports): only require existence
            out_files[key] = (str(fname), "exists")
        else:
            out_files[key] = _path_to_str_hash(
                key,
                fname,
                method=exec_params.memory_file_method,
                kind="out",
            )
    return out_files


def _find_matching_sidecar_cached(
    bids_path: BIDSPath, suffix: str | None, extension: str
) -> pathlib.Path | None:
    state = bids_path.entities
    for key in ("root", "suffix", "extension", "datatype", "check"):
        val = getattr(bids_path, key)
        state[key] = val
    return _find_matching_sidecar_cached_impl(
        **state, pass_suffix=suffix, pass_extension=extension
    )


@functools.cache
def _find_matching_sidecar_cached_impl(
    *, pass_suffix: str | None, pass_extension: str, **state: dict[str, Any]
) -> pathlib.Path | None:
    # We have to do this dance because BIDSPath objects are not hashable, but we
    # want to cache the sidecar finding (which can be expensive when there are
    # many files in a directory). So we cache based on the BIDSPath's state, which
    # is hashable (as it's just a dict of strings and bools).
    bids_path = BIDSPath(**state)
    return bids_path.find_matching_sidecar(
        suffix=pass_suffix,
        extension=pass_extension,
        on_error="ignore",
    )


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
