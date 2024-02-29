#!/usr/bin/env python
"""Generate steps.md."""

import importlib
from pathlib import Path

from mne_bids_pipeline._config_utils import _get_step_modules

autogen_header = f"""\
[//]: # (AUTO-GENERATED, TO CHANGE EDIT {'/'.join(Path(__file__).parts[-4:])})
"""

steps_pre = f"""\
{autogen_header}

# List of processing steps

The following table provides a concise summary of each processing step. The
step names can be used to run individual steps or entire groups of steps by
passing their name(s) to `mne_bids_pipeline` via the `steps=...` argument. However,
we recommend using `mne_bids_pipeline config.py` to run the entire pipeline
instead to ensure that all steps affected by a given change are re-run.
"""  # noqa: E501

overview_pre = f"""\
{autogen_header}

# MNE-BIDS-Pipeline overview

MNE-BIDS-Pipeline processes your data in a sequential manner, i.e., one step
at a time. The next step is only run after the previous steps have been
successfully completed. There are, of course, exceptions; for example, if you
chose not to apply ICA or SSP, the spatial filtering steps will simply be omitted and
we'll directly move to the subsequent steps. See [the flowchart below](#flowchart) for
a visualization of the steps, or check out the
[list of processing steps](steps.md) for more information.

All intermediate results are saved to disk for later inspection, and an
**extensive report** is generated. Analyses are conducted on individual (per-subject)
as well as group level.

## Caching

MNE-BIDS-Pipeline offers automated caching of intermediate results. This means that
running `mne_bids_pipeline config.py` once will generate all outputs, and running it
again will only re-run the steps that need rerunning based on:

1. Changes to files on disk (e.g., updates to `bids_root` files), and
2. Changes to `config.py`

This is particularly useful when you are developing your pipeline, as you can quickly
iterate over changes to your pipeline without having to re-run the entire pipeline
every time -- only the steps that need to be re-run will be executed.

## Flowchart

For more detailed information on each step, please refer to the [detailed list
of processing steps](steps.md).
"""

icon_map = {
    "Filesystem initialization and dataset inspection": ":open_file_folder:",
    "Preprocessing": ":broom:",
    "Sensor-space analysis": ":satellite:",
    "Source-space analysis": ":brain:",
    "FreeSurfer-related processing": ":person_surfing:",
}
out_dir = Path(__file__).parent

print("Generating steps …")
step_modules = _get_step_modules()
char_start = ord("A")

# In principle we could try to sort this out based on naming, but for now let's just
# set our hierarchy manually and update it when we move files around since that's easy
# (and rare) enough to do.
manual_order = {
    "Preprocessing": (
        ("01", "02"),
        ("02", "03"),
        ("03", "04"),
        ("04", "05"),
        ("05", "06a1"),
        ("06a1", "06a2"),
        ("05", "06b"),
        ("05", "07"),
        # technically we could have the raw data flow here, but it doesn't really help
        # ("05", "08a"),
        # ("05", "08b"),
        ("06a2", "08a"),
        # Force the artifact-fitting and epoching steps on the same level, in this order
        """\
    subgraph Z[" "]
    direction LR
      B06a1
      B07
      B06b
    end
    style Z fill:#0000,stroke-width:0px
""",
        ("06b", "08b"),
        ("07", "08a"),
        ("07", "08b"),
        ("08a", "09"),
        ("08b", "09"),
    ),
}

# Construct the lines of steps.md
lines = [steps_pre]
overview_lines = [overview_pre]
used_titles = set()
for di, (dir_, modules) in enumerate(step_modules.items(), 1):
    # Steps
    if dir_ == "all":
        continue  # this is an alias
    dir_module = importlib.import_module(f"mne_bids_pipeline.steps.{dir_}")
    dir_header = dir_module.__doc__.split("\n")[0].rstrip(".")
    dir_body = dir_module.__doc__.split("\n", maxsplit=1)
    if len(dir_body) > 1:
        dir_body = dir_body[1].strip()
    else:
        dir_body = ""
    icon = icon_map[dir_header]
    module_header = f"{di}. {icon} {dir_header}"
    lines.append(f"## {module_header}\n")
    if dir_body:
        lines.append(f"{dir_body}\n")
    lines.append("| Step name | Description |")
    lines.append("|:----------|:------------|")
    # the "all" option
    dir_name, step_title = dir_, f"Run all {dir_header.lower()} steps."
    lines.append(f"`{dir_name}` | {step_title} |")
    for module in modules:
        step_name = f"{dir_name}/{Path(module.__file__).name}"[:-3]
        step_title = module.__doc__.split("\n")[0]
        lines.append(f"`{step_name}` | {step_title} |")
    lines.append("")

    # Overview
    overview_lines.append(
        f"""\
### {module_header}

<details>
<summary>Click to expand</summary>

```mermaid
flowchart TD"""
    )
    chr_pre = chr(char_start + di - 1)  # A, B, C, ...
    start = None
    prev_idx = None
    title_map = {}
    for mi, module in enumerate(modules, 1):
        step_title = module.__doc__.split("\n")[0].rstrip(".")
        idx = module.__name__.split(".")[-1].split("_")[1]  # 01, 05a, etc.
        # Need to quote the title to deal with parens, and sanitize quotes
        step_title = step_title.replace('"', "'")
        assert step_title not in used_titles, f"Redundant title: {step_title}"
        used_titles.add(step_title)
        this_block = f'{chr_pre}{idx}["{step_title}"]'
        # special case: manual order
        title_map[idx] = step_title
        if dir_header in manual_order:
            continue
        if mi == 1:
            start = this_block
            assert prev_idx is None
            continue
        if start is not None:
            assert mi == 2, mi
            overview_lines.append(f"    {start} --> {this_block}")
            start = None
        else:
            overview_lines.append(f"    {chr_pre}{prev_idx} --> {this_block}")
        prev_idx = idx
    if dir_header in manual_order:
        mapped = set()
        for a_b in manual_order[dir_header]:
            if isinstance(a_b, str):  # insert directly
                overview_lines.append(a_b)
                continue
            assert isinstance(a_b, tuple), type(a_b)
            a_b = list(a_b)  # allow modification
            for ii, idx in enumerate(a_b):
                assert idx in title_map, (dir_header, idx, sorted(title_map))
                if idx not in mapped:
                    mapped.add(idx)
                    a_b[ii] = f'{idx}["{title_map[idx]}"]'
            overview_lines.append(f"    {chr_pre}{a_b[0]} --> {chr_pre}{a_b[1]}")
        all_steps = set(
            sum(
                [a_b for a_b in manual_order[dir_header] if not isinstance(a_b, str)],
                (),
            )
        )
        assert mapped == all_steps, all_steps.symmetric_difference(mapped)
    overview_lines.append("```\n\n</details>\n")

(out_dir / "steps.md").write_text("\n".join(lines), encoding="utf8")
(out_dir / "overview.md").write_text("\n".join(overview_lines), encoding="utf8")
