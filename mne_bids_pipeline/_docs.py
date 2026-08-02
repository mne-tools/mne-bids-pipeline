import ast
import inspect
import textwrap
from collections import defaultdict
from pathlib import Path
from types import FunctionType, ModuleType
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
    # Plus the basic BIDS params, which come in via _bids_kwargs and friends and
    # so land on nearly every step. Listing them all would be noise, so instead we
    # mention at the top of General that changing them affects almost every step.
    # A notable exception is random_state, which does have more localized effects.
    "bids_root",
    "deriv_root",
    "sessions",
    "acq",
    "proc",
    "rec",
    "space",
    "task",
    "runs",
    "exclude_runs",
    "subjects",
    "eeg_reference",
    # And some that we force to be empty because they affect too many things
    # and what they affect is an incomplete list anyway
    "exclude_subjects",
    "ch_types",
    "task_is_rest",
    "data_type",
    "allow_missing_sessions",
)


def _bound_names(call: ast.Call, func: FunctionType, names: frozenset[str]) -> set[str]:
    """Get the parameters of `func` that receive one of `names` at this call site.

    This is what keeps the traversal honest: a helper's ``config`` parameter only
    holds our object if the call site actually handed it over. Steps call helpers
    both as ``f(cfg=cfg)`` and ``f(config=cfg)``, while unrelated functions (e.g.
    ``_import_config``) have a ``config`` of their own that must not be followed.
    """
    try:
        params = list(inspect.signature(func).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        params = []
    bound = set()
    for pos, arg in enumerate(call.args):
        if isinstance(arg, ast.Name) and arg.id in names and pos < len(params):
            bound.add(params[pos])
    for keyword in call.keywords:
        if (
            keyword.arg is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id in names
        ):
            bound.add(keyword.arg)
    return bound


def _collect_attrs(
    node: ast.AST,
    namespace: dict[str, Any],
    names: frozenset[str],
    attrs: set[str],
    seen: set[tuple[str, str, frozenset[str]]],
    *,
    follow_get_config: bool,
) -> None:
    """Collect ``<name>.<attr>`` accesses, recursing into functions that are called.

    `names` are the local names currently bound to the object of interest (the
    config for the "write" side, the cfg namespace for the "read" side). Only
    functions defined in this package are followed, and only when the call site
    passes the object to them. Each (function, binding) pair is visited at most
    once, so recursive and mutually recursive helpers terminate.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            if (
                isinstance(sub.value, ast.Name)
                and sub.value.id in names
                and not sub.attr.startswith("__")
                and sub.attr not in _IGNORE_OPTIONS
            ):
                attrs.add(sub.attr)
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            func = _resolve_func(sub.func.id, namespace)
            if func is None:
                continue
            # get_config* builds cfg; it is the "write" side, never a cfg reader
            if not follow_get_config and func.__name__.startswith("get_config"):
                continue
            bound = _bound_names(sub, func, names)
            if not bound:
                continue
            key = (func.__module__, func.__qualname__, frozenset(bound))
            if key in seen:
                continue
            seen.add(key)
            source = textwrap.dedent(inspect.getsource(func))
            _collect_attrs(
                ast.parse(source),
                func.__globals__,
                frozenset(bound),
                attrs,
                seen,
                follow_get_config=follow_get_config,
            )


def _step_module_funcs(module: ModuleType) -> list[ast.FunctionDef]:
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text("utf-8"))
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def get_step_options(module: ModuleType) -> set[str]:
    """Get the config options a step writes into its ``cfg`` (or uses in ``main``).

    This is the "write" side: ``get_config*`` and ``main`` are walked for
    ``config.<option>`` accesses, following calls into package helpers.
    """
    funcs = _step_module_funcs(module)
    assert any(func.name.startswith("get_config") for func in funcs), (
        f"Could not find get_config* in {module.__name__}"
    )
    options: set[str] = set()
    seen: set[tuple[str, str, frozenset[str]]] = set()
    for func in funcs:
        if func.name == "main" or func.name.startswith("get_config"):
            _collect_attrs(
                func,
                vars(module),
                frozenset({"config"}),
                options,
                seen,
                follow_get_config=True,
            )
    return options


def _resolve_func(name: str, namespace: dict[str, Any]) -> FunctionType | None:
    func = namespace.get(name)
    if func is not None:
        func = inspect.unwrap(func)
    if isinstance(func, FunctionType) and (func.__module__ or "").startswith(
        __package__
    ):
        return func
    return None


def _dict_keys_assigned(scope: ast.AST, name: str) -> set[str]:
    """Get the literal string keys assigned into a local dict, e.g. ``d["k"] = v``."""
    keys = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
    return keys


def _fields_from_value(
    value: ast.expr, namespace: dict[str, Any], fields: set[str], scope: ast.AST
) -> None:
    """Collect the field names an expression assigned to a kwargs dict contributes."""
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
        return
    if value.func.id in ("dict", "SimpleNamespace"):
        _fields_from_call(value, namespace, fields, scope)
        return
    func = _resolve_func(value.func.id, namespace)
    if func is None:
        return
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("dict", "SimpleNamespace")
        ):
            _fields_from_call(node, func.__globals__, fields, tree)


def _fields_from_call(
    call: ast.Call, namespace: dict[str, Any], fields: set[str], scope: ast.AST
) -> None:
    """Collect the field names a ``SimpleNamespace(...)``/``dict(...)`` call builds."""
    for keyword in call.keywords:
        if keyword.arg is not None:
            fields.add(keyword.arg)
            continue
        # ``**extra_kwargs`` where a local dict was built up conditionally, either
        # by subscript assignment or by assigning a helper's return value
        if isinstance(keyword.value, ast.Name):
            name = keyword.value.id
            fields |= _dict_keys_assigned(scope, name)
            for node in ast.walk(scope):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets
                ):
                    _fields_from_value(node.value, namespace, fields, scope)
            continue
        # ``**_helper(...)`` expansion: recurse into the dict the helper returns
        if not (
            isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Name)
        ):
            continue
        func = _resolve_func(keyword.value.func.id, namespace)
        if func is None:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("dict", "SimpleNamespace")
            ):
                _fields_from_call(node, func.__globals__, fields, tree)


def get_kwargs_helper_fields(func: FunctionType) -> set[str]:
    """Get the cfg field names a ``*_kwargs`` helper contributes to a step."""
    fields: set[str] = set()
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("dict", "SimpleNamespace")
        ):
            _fields_from_call(node, func.__globals__, fields, tree)
    return fields - _IGNORE_OPTIONS


def get_step_cfg_fields(module: ModuleType) -> set[str]:
    """Get the ``cfg`` fields a step's ``get_config*`` builds.

    Unlike :func:`get_step_options` (which reports *config options* consulted,
    including ones read to compute a derived value), this reports the names that
    end up on the ``cfg`` namespace, so it can be compared against
    :func:`get_step_reads` to find fields that are passed but never read.
    """
    fields: set[str] = set()
    for func in _step_module_funcs(module):
        if not func.name.startswith("get_config"):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "SimpleNamespace"
            ):
                _fields_from_call(node, vars(module), fields, func)
    # symmetry with the read side, which also ignores these
    return fields - _IGNORE_OPTIONS


def get_step_reads(module: ModuleType) -> set[str]:
    """Get the ``cfg`` fields a step actually reads at run time.

    This is the "read" side, and the counterpart of :func:`get_step_options`.
    Every function in the step module except ``get_config*`` is walked, and calls
    into package helpers are followed.

    Only the name ``cfg`` seeds the traversal, which is what makes ``main`` safe
    to walk like any other function: its ``config`` is the real config, so
    ``config.<option>`` there is not a cfg read and ``f(config=config)`` is not
    followed, while ``f(config=cfg)`` -- which ``main`` does do -- is.
    """
    reads: set[str] = set()
    seen: set[tuple[str, str, frozenset[str]]] = set()
    for func in _step_module_funcs(module):
        if func.name.startswith("get_config"):
            continue
        _collect_attrs(
            func,
            vars(module),
            frozenset({"cfg"}),
            reads,
            seen,
            follow_get_config=False,
        )
    return reads


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
            for option in sorted(get_step_options(module)):
                steps[option].append(step)
        for key in self._force_empty:
            steps[key] = list()
        self.steps: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in steps.items()}

    def __call__(self, option: str) -> tuple[str, ...]:
        return self.steps[option]
