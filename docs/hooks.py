"""Custom hooks for MkDocs-Material."""

import ast
import inspect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from tqdm import tqdm

from mne_bids_pipeline import _config_utils

logger = logging.getLogger("mkdocs")

config_updated = False


class _ParseConfigSteps:
    def __init__(self):
        self.steps = defaultdict(list)
        # We don't need to parse the config itself, just the steps
        no_config = {
            "freesurfer/_01_recon_all",
        }
        ignore_options = {
            "PIPELINE_NAME",
            "VERSION",
            "CODE_URL",
        }
        ignore_calls = {
            # TODO: These are used a lot at the very beginning, so adding them will lead
            # to long lists. Instead, let's just mention at the top of General that
            # messing with basic BIDS params will affect almost every step.
            "_bids_kwargs",
            "_import_data_kwargs",
            "get_runs",
            "get_subjects",
            "get_sessions",
        }
        manual_kws = {
            "source/_04_make_forward:get_config:t1_bids_path": (
                "mri_t1_path_generator",
            ),
            "source/_04_make_forward:get_config:landmarks_kind": (
                "mri_landmarks_kind",
            ),
            "preprocessing/_01_data_quality:get_config:extra_kwargs": (
                "mf_cal_fname",
                "mf_ctc_fname",
                "mf_head_origin",
                "find_flat_channels_meg",
                "find_noisy_channels_meg",
            ),
        }
        # Add a few helper functions
        for func in (
            _config_utils.get_eeg_reference,
            _config_utils.get_all_contrasts,
            _config_utils.get_decoding_contrasts,
            _config_utils.get_fs_subject,
            _config_utils.get_fs_subjects_dir,
            _config_utils.get_mf_cal_fname,
            _config_utils.get_mf_ctc_fname,
        ):
            this_list = []
            for attr in ast.walk(ast.parse(inspect.getsource(func))):
                if not isinstance(attr, ast.Attribute):
                    continue
                if not (isinstance(attr.value, ast.Name) and attr.value.id == "config"):
                    continue
                if attr.attr not in this_list:
                    this_list.append(attr.attr)
            manual_kws[func.__name__] = tuple(this_list)

        for module in tqdm(
            sum(_config_utils._get_step_modules().values(), tuple()),
            desc="Generating option->step mapping",
        ):
            step = "/".join(module.__name__.split(".")[-2:])
            found = False  # found at least one?
            # Walk the module file for "get_config*" functions (can be multiple!)
            for func in ast.walk(ast.parse(Path(module.__file__).read_text("utf-8"))):
                if not isinstance(func, ast.FunctionDef):
                    continue
                where = f"{step}:{func.name}"
                # Also look at config.* args in main(), e.g. config.recreate_bem
                # and config.recreate_scalp_surface
                if func.name == "main":
                    for call in ast.walk(func):
                        if not isinstance(call, ast.Call):
                            continue
                        for keyword in call.keywords:
                            if not isinstance(keyword.value, ast.Attribute):
                                continue
                            if keyword.value.value.id != "config":
                                continue
                            if keyword.value.attr in ("exec_params",):
                                continue
                            self._add_step_option(step, keyword.value.attr)
                    # Also look for root-level conditionals like use_maxwell_filter
                    # or spatial_filter
                    for cond in ast.iter_child_nodes(func):
                        # is a conditional
                        if not isinstance(cond, ast.If):
                            continue
                        # has a return statement
                        if not any(isinstance(c, ast.Return) for c in ast.walk(cond)):
                            continue
                        # look at all attributes in the conditional
                        for attr in ast.walk(cond.test):
                            if not isinstance(attr, ast.Attribute):
                                continue
                            if attr.value.id != "config":
                                continue
                            self._add_step_option(step, attr.attr)
                # Now look at get_config* functions
                if not func.name.startswith("get_config"):
                    continue
                found = True
                for call in ast.walk(func):
                    if not isinstance(call, ast.Call):
                        continue
                    if call.func.id != "SimpleNamespace":
                        continue
                    break
                else:
                    raise RuntimeError(f"Could not find SimpleNamespace in {func}")
                assert call.args == []
                for keyword in call.keywords:
                    if isinstance(keyword.value, ast.Call):
                        key = keyword.value.func.id
                        if key in ignore_calls:
                            continue
                        if key in manual_kws:
                            for option in manual_kws[key]:
                                self._add_step_option(step, option)
                            continue
                        if keyword.value.func.id == "_sanitize_callable":
                            assert len(keyword.value.args) == 1
                            assert isinstance(keyword.value.args[0], ast.Attribute)
                            assert keyword.value.args[0].value.id == "config"
                            self._add_step_option(step, keyword.value.args[0].attr)
                            continue
                        raise RuntimeError(
                            f"{where} cannot handle call {keyword.value.func.id=}"
                        )
                    if isinstance(keyword.value, ast.Name):
                        key = f"{where}:{keyword.value.id}"
                        if key in manual_kws:
                            for option in manual_kws[f"{where}:{keyword.value.id}"]:
                                self._add_step_option(step, option)
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
                    if option in ignore_options:
                        continue
                    assert keyword.value.value.id == "config", f"{where} {keyword.value.value.id}"  # noqa: E501  # fmt: skip
                    self._add_step_option(step, option)
            if step in no_config:
                assert not found, f"Found unexpected get_config* in {step}"
            else:
                assert found, f"Could not find get_config* in {step}"
        # Some don't show up so force them to be empty
        force_empty = (
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
            # Plus some BIDS one we don't detect because _bids_kwargs etc. above,
            # which we could cross-check against the general.md list. A notable
            # exception is random_state, since this does have more localized effects.
            "study_name",
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
            "mf_reference_run",  # TODO: Make clearer that this changes a lot
            "fix_stim_artifact",
            "stim_artifact_tmin",
            "stim_artifact_tmax",
            # And some that we force to be empty because they affect too many things
            # and what they affect is an incomplete list anyway
            "exclude_subjects",
            "ch_types",
            "task_is_rest",
            "data_type",
        )
        for key in force_empty:
            self.steps[key] = list()
        for key, val in self.steps.items():
            assert len(val) == len(set(val)), f"{key} {val}"
        self.steps = {k: tuple(v) for k, v in self.steps.items()}  # no defaultdict

    def _add_step_option(self, step, option):
        if step not in self.steps[option]:
            self.steps[option].append(step)

    def __call__(self, option: str) -> list[str]:
        return self.steps[option]


_parse_config_steps = _ParseConfigSteps()


# This hack can be cleaned up once this is resolved:
# https://github.com/mkdocstrings/mkdocstrings/issues/615#issuecomment-1971568301
def on_pre_build(config: MkDocsConfig) -> None:
    """Monkey patch mkdocstrings-python jinja template to have global vars."""
    import mkdocstrings_handlers.python.handler

    old_update_env = mkdocstrings_handlers.python.handler.PythonHandler.update_env

    def update_env(self, md, config: dict) -> None:
        old_update_env(self, md=md, config=config)
        self.env.globals["pipeline_steps"] = _parse_config_steps

    mkdocstrings_handlers.python.handler.PythonHandler.update_env = update_env


# Ideally there would be a better hook, but it's unclear if context can
# be obtained any earlier
def on_template_context(
    context: dict[str, Any],
    template_name: str,
    config: MkDocsConfig,
) -> None:
    """Update the copyright in the footer."""
    global config_updated
    if not config_updated:
        config_updated = True
        now = context["build_date_utc"].strftime("%Y/%m/%d")
        config.copyright = f"{config.copyright}, last updated {now}"
        logger.info(f"Updated copyright to {config.copyright}")


_EMOJI_MAP = {
    "🏆": ":trophy:",
    "🛠️": ":tools:",
    "📘": ":blue_book:",
    "🧑‍🤝‍🧑": ":people_holding_hands_tone1:",
    "💻": ":computer:",
    "🆘": ":sos:",
    "👣": ":footprints:",
    "⏩": ":fast_forward:",
    "⏏️": ":eject:",
    "☁️": ":cloud:",
}


def on_page_markdown(
    markdown: str,
    page: Page,
    config: MkDocsConfig,
    files: Files,
) -> str:
    """Replace emojis."""
    if page.file.name == "index" and page.title == "Home":
        for rd, md in _EMOJI_MAP.items():
            markdown = markdown.replace(rd, md)
    return markdown
