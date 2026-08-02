import ast
import inspect
import textwrap
from collections import defaultdict
from pathlib import Path
from types import FunctionType
from typing import Any

from tqdm import tqdm

from . import _config_utils

_IGNORE_OPTIONS = {
    "PIPELINE_NAME",
    "VERSION",
    "CODE_URL",
    "all_tasks",
    "exec_params",
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
    "read_raw_bids_verbose",
    "ignore_warnings",
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


def _collect_options(
    node: ast.AST,
    namespace: dict[str, Any],
    options: list[str],
    seen: set[tuple[str, str]],
) -> None:
    """Collect config.* attribute accesses in node, recursing into called functions."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            if (
                isinstance(sub.value, ast.Name)
                and sub.value.id == "config"
                and not sub.attr.startswith("__")
                and sub.attr not in _IGNORE_OPTIONS
                and sub.attr not in options
            ):
                options.append(sub.attr)
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            func = namespace.get(sub.func.id)
            if func is not None:
                func = inspect.unwrap(func)
            if not isinstance(func, FunctionType):
                continue
            # Only follow functions defined in this package
            if not (func.__module__ or "").startswith(__package__):
                continue
            key = (func.__module__, func.__qualname__)
            if key in seen:
                continue
            seen.add(key)
            source = textwrap.dedent(inspect.getsource(func))
            _collect_options(ast.parse(source), func.__globals__, options, seen)


class _ParseConfigSteps:
    def __init__(self, force_empty: tuple[str, ...] | None = None) -> None:
        """Build a mapping from config options to tuples of steps that use each option.

        Each step module's ``get_config*`` and ``main`` functions are statically
        analyzed: every ``config.<option>`` attribute access counts as a use, and
        calls to functions defined in this package are followed recursively so
        that options used indirectly via helpers are attributed to the step, too.

        The mapping is stored in `self.steps`.
        """
        self._force_empty = _FORCE_EMPTY if force_empty is None else force_empty
        steps: dict[str, list[str]] = defaultdict(list)
        modules = dict.fromkeys(  # deduplicate ("all" repeats the other groups)
            sum(_config_utils._get_step_modules().values(), tuple())
        )
        for module in tqdm(modules, desc="Generating option->step mapping"):
            step = "/".join(module.__name__.split(".")[-2:])
            assert module.__file__ is not None
            tree = ast.parse(Path(module.__file__).read_text("utf-8"))
            # Walk get_config* (can be multiple!) and main
            funcs = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and (node.name == "main" or node.name.startswith("get_config"))
            ]
            assert any(node.name.startswith("get_config") for node in funcs), (
                f"Could not find get_config* in {step}"
            )
            options: list[str] = []
            seen: set[tuple[str, str]] = set()
            for func in funcs:
                _collect_options(func, vars(module), options, seen)
            for option in options:
                steps[option].append(step)
        for key in self._force_empty:
            steps[key] = list()
        self.steps: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in steps.items()}

    def __call__(self, option: str) -> tuple[str, ...]:
        return self.steps[option]
