import ast
import inspect
import re
from collections import defaultdict
from pathlib import Path
from types import FunctionType
from typing import Any

from tqdm import tqdm

from . import _config_utils, _decoding, _import_data

_CONFIG_RE = re.compile(r"config\.([a-zA-Z_]+)")

_NO_CONFIG = {
    "freesurfer/_01_recon_all",
}
_IGNORE_OPTIONS = {
    "PIPELINE_NAME",
    "VERSION",
    "CODE_URL",
    "all_tasks",
}
# We don't need to parse the config itself, just the steps
_MANUAL_KWS = {
    "preprocessing/_01_data_quality:get_config:extra_kwargs": (
        "mf_cal_fname",
        "mf_ctc_fname",
        "mf_head_origin",
        "find_flat_channels_meg",
        "find_noisy_channels_meg",
    ),
    "preprocessing/_06a1_fit_ica:get_config:epochs_tmin": ("epochs_tmin",),
    "preprocessing/_06a1_fit_ica:get_config:epochs_tmax": ("epochs_tmax",),
    "source/_04_make_forward:get_config:t1_bids_path": ("mri_t1_path_generator",),
    "source/_04_make_forward:get_config:landmarks_kind": ("mri_landmarks_kind",),
}
# Some don't show up so force them to be empty
_EXECUTION_OPTIONS = (
    # Eventually we could deduplicate these with the execution.md list
    "n_jobs",
    "parallel_backend",
    "dask_open_dashboard",
    "dask_temp_dir",
    "dask_worker_memory_limit",
    "log_level",
    "mne_log_level",
    "on_error",
    "memory_location",
    "memory_file_method",
    "memory_subdir",
    "memory_verbose",
    "config_validation",
    "interactive",
)
_FORCE_EMPTY = _EXECUTION_OPTIONS + (
    # Plus some BIDS one we don't detect because _bids_kwargs etc. above,
    # which we could cross-check against the general.md list. A notable
    # exception is random_state, since this does have more localized effects.
    # These are used a lot at the very beginning, so adding them will lead
    # to long lists. Instead, let's just mention at the top of General that
    # messing with basic BIDS params will affect almost every step.
    "bids_root",
    "deriv_root",
    "subjects_dir",
    "sessions",
    "acq",
    "proc",
    "rec",
    "space",
    "task",
    "runs",
    "exclude_runs",
    "subjects",
    "crop_runs",
    "process_empty_room",
    "process_rest",
    "eeg_bipolar_channels",
    "eeg_reference",
    "eeg_template_montage",
    "drop_channels",
    "reader_extra_params",
    "read_raw_bids_verbose",
    "plot_psd_for_runs",
    "shortest_event",
    "find_breaks",
    "min_break_duration",
    "t_break_annot_start_after_previous_event",
    "t_break_annot_stop_before_next_event",
    "rename_events",
    "on_rename_missing_events",
    "fix_stim_artifact",
    "stim_artifact_tmin",
    "stim_artifact_tmax",
    # And some that we force to be empty because they affect too many things
    # and what they affect is an incomplete list anyway
    "exclude_subjects",
    "ch_types",
    "task_is_rest",
    "data_type",
    "allow_missing_sessions",
)
# Eventually we could parse AST to get these, but this is simple enough
_EXTRA_FUNCS = {
    "_import_data_kwargs": ("get_mf_reference_run_task",),
    "get_sessions": ("_get_sessions",),
    "_decoding_preproc_steps": ("_get_rank",),
}


class _ParseConfigSteps:
    def __init__(self, force_empty: tuple[str, ...] | None = None) -> None:
        """Build a mapping from config options to tuples of steps that use each option.

        The mapping is stored in `self.steps`.
        """
        self._force_empty = _FORCE_EMPTY if force_empty is None else force_empty
        steps: dict[str, Any] = defaultdict(list)

        def _add_step_option(step: str, option: str) -> None:
            if option not in _IGNORE_OPTIONS and step not in steps[option]:
                steps[option].append(step)

        # Add a few helper functions
        for func_extra in (
            _config_utils._get_task_conditions_dict,
            _config_utils._get_task_contrasts,
            _config_utils._get_task_decoding_contrasts,
            _config_utils._limit_which_clean,
            _config_utils.get_eeg_reference,
            _config_utils.get_fs_subject,
            _config_utils.get_fs_subjects_dir,
            _config_utils.get_mf_cal_fname,
            _config_utils.get_mf_ctc_fname,
            _config_utils.get_subjects_sessions,
            _import_data._get_mf_reference_path,
        ):
            this_list: list[str] = []
            assert isinstance(func_extra, FunctionType)
            for attr in ast.walk(ast.parse(inspect.getsource(func_extra))):
                if not isinstance(attr, ast.Attribute):
                    continue
                if not (isinstance(attr.value, ast.Name) and attr.value.id == "config"):
                    continue
                if attr.attr not in this_list:
                    this_list.append(attr.attr)
            _MANUAL_KWS[func_extra.__name__] = tuple(this_list)

        for module in tqdm(
            sum(_config_utils._get_step_modules().values(), tuple()),
            desc="Generating option->step mapping",
        ):
            step = "/".join(module.__name__.split(".")[-2:])
            found = False  # found at least one?
            # Walk the module file for "get_config*" functions (can be multiple!)
            assert module.__file__ is not None
            source = Path(module.__file__).read_text("utf-8")
            for func in ast.walk(ast.parse(source)):
                if not isinstance(func, ast.FunctionDef):
                    continue
                where = f"{step}:{func.name}"
                # Also look at config.* args in main(), e.g. config.recreate_bem
                # and config.recreate_scalp_surface
                if func.name == "main":
                    # Look at all function calls in main()
                    for call in ast.walk(func):
                        if not isinstance(call, ast.Call):
                            continue
                        if not isinstance(call.func, ast.Name):
                            continue
                        key = call.func.id
                        for keyword in call.keywords:
                            if not isinstance(keyword.value, ast.Attribute):
                                continue
                            assert isinstance(keyword.value.value, ast.Name)
                            if keyword.value.value.id != "config":
                                continue
                            if keyword.value.attr in ("exec_params",):
                                continue
                            _add_step_option(step, keyword.value.attr)
                        if key in _MANUAL_KWS:
                            for option in _MANUAL_KWS[key]:
                                _add_step_option(step, option)

                    # Also look for root-level conditionals like use_maxwell_filter,
                    # spatial_filter, or inverse_targets
                    for cond in ast.walk(func):
                        # is a conditional in main()
                        if not isinstance(cond, ast.If):
                            continue
                        # missing a return statement inside it
                        if not any(isinstance(c, ast.Return) for c in ast.walk(cond)):
                            if isinstance(cond.test, ast.Compare) and isinstance(
                                cond.test.comparators[0], ast.Attribute
                            ):
                                attr = cond.test.comparators[0]
                                if (
                                    isinstance(attr.value, ast.Name)
                                    and attr.value.id == "config"
                                ):
                                    _add_step_option(step, attr.attr)
                            # This sort of debugging can help next time something
                            # isn't added in a main() conditional:
                            #
                            # this_source = ast.get_source_segment(source, cond.test)
                            # if (
                            #     this_source is not None
                            #     and "inverse_targets" in this_source
                            # ):
                            #     1/0
                            #
                            continue
                        # Okay, we know it has a return statement inside somewhere
                        for attr in ast.walk(cond.test):
                            if not isinstance(attr, ast.Attribute):
                                continue
                            assert isinstance(attr.value, ast.Name)
                            if attr.value.id != "config":
                                continue
                            _add_step_option(step, attr.attr)

                # Now look at get_config* functions anywhere in the file
                if not func.name.startswith("get_config"):
                    continue
                found = True
                for call in ast.walk(func):
                    if not isinstance(call, ast.Call):
                        continue
                    # We want named functions, not things like class attributes
                    # (i.e., my_func() not my_dict.items())
                    assert isinstance(call.func, ast.Name)
                    if call.func.id != "SimpleNamespace":
                        continue
                    break
                else:
                    raise RuntimeError(f"Could not find SimpleNamespace in {func}")
                assert call.args == []
                for keyword in call.keywords:
                    if isinstance(keyword.value, ast.Call):
                        assert isinstance(keyword.value.func, ast.Name)
                        key = keyword.value.func.id
                        if key in _MANUAL_KWS:
                            for option in _MANUAL_KWS[key]:
                                _add_step_option(step, option)
                            continue
                        func_id = keyword.value.func.id
                        if func_id in (
                            "_sanitize_callable",
                            "_get_task_float",
                        ):
                            assert len(keyword.value.args) == 1
                            assert isinstance(keyword.value.args[0], ast.Attribute)
                            assert isinstance(keyword.value.args[0].value, ast.Name)
                            assert keyword.value.args[0].value.id == "config"
                            option = keyword.value.args[0].attr
                            _add_step_option(step, option)
                            continue
                        # Allowlist of function names that we ignore when deciding
                        # which config options are used. Things like `_bids_kwargs`
                        # for example are used in every func and use lots of config
                        # values, so they would just add unnecessary noise (people
                        # will understand that changing something like "runs" will
                        # affect many steps).
                        if key not in (
                            "_bids_kwargs",
                            "_import_data_kwargs",
                            "_get_runs_sst",
                            "get_datatype",
                            "get_runs_tasks",
                            "get_subjects",
                            "get_sessions",
                        ):
                            raise RuntimeError(
                                f"{where} cannot handle call {keyword.value.func.id=} "
                                f"for {key}"
                            )
                        # Get the source and regex for config values
                        if key == "_import_data_kwargs":
                            funcs = [getattr(_import_data, key)]
                        elif key == "_decoding_preproc_steps":
                            funcs = [getattr(_decoding, key)]
                        else:
                            funcs = [getattr(_config_utils, key)]
                        for func_name in _EXTRA_FUNCS.get(key, ()):
                            funcs.append(getattr(_config_utils, func_name))
                        for fi, func in enumerate(funcs):
                            assert isinstance(func, FunctionType), func
                            func_source = inspect.getsource(func)
                            assert "config: SimpleNamespace" in func_source, key
                            if fi == 0:
                                for func_name in _EXTRA_FUNCS.get(key, ()):
                                    assert f"{func_name}(" in func_source, (
                                        key,
                                        func_name,
                                    )
                            attrs = _CONFIG_RE.findall(func_source)
                            # pure wrappers
                            if key not in (
                                "get_sessions",
                                "get_runs_tasks",
                            ):
                                assert len(attrs), (
                                    f"No config.* found in source of {key}"
                                )
                            for attr in attrs:
                                _add_step_option(step, attr)
                        continue
                    if isinstance(keyword.value, ast.Name):
                        key = f"{where}:{keyword.value.id}"
                        if key in _MANUAL_KWS:
                            for option in _MANUAL_KWS[key]:
                                _add_step_option(step, option)
                            continue
                        raise RuntimeError(f"{where} cannot handle Name {key=}")
                    if isinstance(keyword.value, ast.IfExp):  # conditional
                        if keyword.arg == "processing":  # inline conditional for proc
                            continue
                    if not isinstance(keyword.value, ast.Attribute):
                        raise RuntimeError(
                            f"{where} cannot handle type {keyword.value=}"
                        )
                    option = keyword.value.attr
                    if option in _IGNORE_OPTIONS:
                        continue
                    assert isinstance(keyword.value.value, ast.Name)
                    assert keyword.value.value.id == "config", f"{where} {keyword.value.value.id}"  # noqa: E501  # fmt: skip
                    _add_step_option(step, option)
            if step in _NO_CONFIG:
                assert not found, f"Found unexpected get_config* in {step}"
            else:
                assert found, f"Could not find get_config* in {step}"
        for key in self._force_empty:
            steps[key] = list()
        for key, val in steps.items():
            assert len(val) == len(set(val)), f"{key} {val}"
        self.steps: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in steps.items()}

    def __call__(self, option: str) -> tuple[str, ...]:
        return self.steps[option]
