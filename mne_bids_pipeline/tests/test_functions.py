"""Test some properties of our core processing-step functions."""

import ast
import inspect

import pytest

from mne_bids_pipeline._config_utils import _get_step_modules

# mne_bids_pipeline.init._01_init_derivatives_dir: <module>
FLAT_MODULES = {x.__name__: x for x in sum(_get_step_modules().values(), ())}


@pytest.mark.parametrize("module_name", list(FLAT_MODULES))
def test_all_functions_return(module_name: str) -> None:
    """Test that all functions decorated with failsafe_run return a dict."""
    # Find the functions within the module that use the failsafe_run decorator
    module = FLAT_MODULES[module_name]
    funcs = list()
    ignore = {  # decorated functions that we can safely ignore
        "_fake_sss_context"
    }
    for name in dir(module):
        obj = getattr(module, name)
        if not callable(obj):
            continue
        if getattr(obj, "__module__", None) != module_name:
            continue
        if not hasattr(obj, "__wrapped__"):  # not decorated
            continue
        if name in ignore:
            continue
        # All our failsafe_run decorated functions should look like this
        assert "__mne_bids_pipeline_failsafe_wrapper__" in repr(obj.__code__), (
            f"{module_name}.{name} is missing failsafe_run decorator"
        )
        funcs.append(obj)
    # Some module names we know don't have any
    if module_name.split(".")[-1] in ("_01_recon_all",):
        assert len(funcs) == 0
        return

    assert len(funcs) != 0, f"No failsafe_runs functions found in {module_name}"

    # Adapted from numpydoc RT01 validation
    def get_returns_not_on_nested_functions(node: ast.AST) -> list[ast.Return]:
        returns = [node] if isinstance(node, ast.Return) else []
        for child in ast.iter_child_nodes(node):
            # Ignore nested functions and its subtrees.
            if not isinstance(child, ast.FunctionDef):
                child_returns = get_returns_not_on_nested_functions(child)
                returns.extend(child_returns)
        return returns

    for func in funcs:
        what = f"{module_name}.{func.__name__}"
        tree = ast.parse(inspect.getsource(func.__wrapped__)).body
        if func.__closure__[-1].cell_contents is False:
            continue  # last closure node is require_output=False
        assert tree, f"Failed to parse source code for {what}"
        returns = get_returns_not_on_nested_functions(tree[0])
        return_values = [r.value for r in returns]
        # Replace Constant nodes valued None for None.
        for i, v in enumerate(return_values):
            if isinstance(v, ast.Constant) and v.value is None:
                return_values[i] = None
        assert len(return_values), f"Function does not return anything: {what}"
        for r in return_values:
            what = f"Function does _prep_out_files: {what}"
            assert isinstance(r, ast.Call), what
            assert isinstance(r.func, ast.Name), what
            assert r.func.id in ("_prep_out_files", "_prep_out_files_path"), what
