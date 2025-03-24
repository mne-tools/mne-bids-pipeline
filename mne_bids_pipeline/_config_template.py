import ast
from pathlib import Path

from ._logging import gen_log_kwargs, logger

CONFIG_SOURCE_PATH = Path(__file__).parent / "_config.py"


def create_template_config(
    target_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Create a template configuration file."""
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"The specified path already exists: {target_path}")

    # Create a template by commenting out most of the lines in _config.py
    config: list[str] = ["# Template config file for mne_bids_pipeline.", ""]
    text = CONFIG_SOURCE_PATH.read_text(encoding="utf-8")
    # skip file header
    to_strip = "# Default settings for data processing and analysis.\n\n"
    if text.startswith(to_strip):
        text = text[len(to_strip) :]
    lines = text.split("\n")
    # make sure we catch all imports and assignments
    tree = ast.parse(text, type_comments=True)
    for ix, line in enumerate(lines, start=1):  # ast.parse assigns 1-indexed `lineno`!
        nodes = [_node for _node in tree.body if _node.lineno <= ix <= _node.end_lineno]  # type:ignore[operator]
        if not nodes:
            # blank lines and comments aren't parsed by `ast.parse`:
            assert line == "" or line.startswith("#"), line
        else:
            assert len(nodes) == 1, nodes
            node = nodes[0]
            # config value assignments should become commented out:
            if isinstance(node, ast.AnnAssign):
                line = f"# {line}"
            # imports get written as-is (not commented out):
            elif isinstance(node, ast.Import | ast.ImportFrom):
                pass
            # everything else should be (multiline) string literals:
            else:
                assert isinstance(node, ast.Expr), node
                assert isinstance(node.value, ast.Constant), node.value
                assert isinstance(node.value.value, str), node.value.value
        config.append(line)

    target_path.write_text("\n".join(config), encoding="utf-8")
    message = f"Successfully created template configuration file at: {target_path}"
    logger.info(**gen_log_kwargs(message=message, emoji="✅"))

    message = "Please edit the file before running the pipeline."
    logger.info(**gen_log_kwargs(message=message, emoji="💡"))
