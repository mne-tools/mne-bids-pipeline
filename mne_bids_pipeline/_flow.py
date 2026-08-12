"""Auto-generated pipeline flow diagrams for the HTML reports.

Steps record what they read and wrote (see ``_run.py``); this module turns those
recordings into the step dependency graph and its labels. The generic layout and
SVG rendering live in ``_graph.py``.
"""

import json
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence

from filelock import FileLock
from mne_bids import BIDSPath

from ._graph import _ID_PREFIX, _Edge, _Graph, _graph_html, _layout_graph, _Node
from .typing import TypedDict

_FLOW_DIRNAME = ".pipeline_flow"
_FLOW_VERSION = 1
_SOURCE_ID = "__bids_input__"
_SOURCE_LABEL = "BIDS raw data"


class FlowEntryT(TypedDict):
    """One recorded call of a pipeline step's worker function."""

    step: str
    func: str
    subject: str | None
    session: str | None
    run: str | None
    task: str | None
    in_files: dict[str, str]
    out_files: dict[str, str]


# -- Storage -----------------------------------------------------------------


def _flow_dir(deriv_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(deriv_root) / _FLOW_DIRNAME


def _flow_fname(*, deriv_root: pathlib.Path, subject: str | None) -> pathlib.Path:
    # One file per subject keeps the read-modify-write cheap and keeps parallel
    # subjects off each other's lock.
    stem = "dataset" if subject is None else f"sub-{subject}"
    return _flow_dir(deriv_root) / f"{stem}.json"


def _flow_entry_key(entry: FlowEntryT) -> str:
    keys = ("step", "func", "subject", "session", "run", "task")
    return "|".join(str(entry[key] or "") for key in keys)  # type: ignore[literal-required]


def _read_flow_file(fname: pathlib.Path) -> dict[str, FlowEntryT]:
    if not fname.is_file():
        return dict()
    with FileLock(f"{fname}.lock"):
        try:
            content = json.loads(fname.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            content = dict()
    if content.get("version", None) != _FLOW_VERSION:
        return dict()
    entries: dict[str, FlowEntryT] = content["entries"]
    return entries


def _write_flow_entry(
    *, deriv_root: pathlib.Path, entry: FlowEntryT, only_if_new: bool = False
) -> None:
    """Merge a single recorded step call into the on-disk recording."""
    fname = _flow_fname(deriv_root=deriv_root, subject=entry["subject"])
    fname.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{fname}.lock"):
        entries: dict[str, FlowEntryT] = dict()
        if fname.is_file():
            try:
                content = json.loads(fname.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                content = dict()
            if content.get("version", None) == _FLOW_VERSION:
                entries = content["entries"]
        key = _flow_entry_key(entry)
        if only_if_new and key in entries:
            return
        entries[key] = entry
        fname.write_text(
            json.dumps(dict(version=_FLOW_VERSION, entries=entries), indent=1),
            encoding="utf-8",
        )


def _read_flow_entries(
    *, deriv_root: pathlib.Path, subject: str, session: str | None
) -> list[FlowEntryT]:
    """Get the recorded step calls relevant for one report."""
    fnames = [_flow_fname(deriv_root=deriv_root, subject=None)]
    if subject == "average":
        # Group steps read individual subjects' files, so the group report needs every
        # subject's recording to attribute those inputs to the steps that made them.
        fnames += sorted(_flow_dir(deriv_root).glob("sub-*.json"))
    else:
        fnames.append(_flow_fname(deriv_root=deriv_root, subject=subject))
    entries: list[FlowEntryT] = list()
    for fname in fnames:
        for entry in _read_flow_file(fname).values():
            if entry["session"] in (None, session):
                entries.append(entry)
    entries.sort(key=_flow_entry_key)
    return entries


# -- Graph -------------------------------------------------------------------


def _node_id(step: str) -> str:
    return f"{_ID_PREFIX}-node-{re.sub(r'[^A-Za-z0-9_-]+', '-', step)}"


def _step_lines(step: str) -> list[str]:
    group, _, name = step.partition("/")
    return [group, name] if name else [group]


def _fname_bits(path: str) -> tuple[str | None, str | None, str | None]:
    """Get the (processing, suffix, extension) descriptors of a filename."""
    name = pathlib.Path(path).name
    stem, _, _ = name.partition(".")
    extension = name[len(stem) :] or None
    parts = stem.split("_")
    if not any("-" in part for part in parts):  # not a BIDS-style name
        return None, None, extension
    processing = None
    for part in parts:
        key, sep, value = part.partition("-")
        if sep and key == "proc":
            processing = value
    suffix = parts[-1] if "-" not in parts[-1] else None
    return processing, suffix, extension


def _fname_run(path: str) -> str | None:
    for part in pathlib.Path(path).name.split("_"):
        key, sep, value = part.partition("-")
        if sep and key == "run":
            return value
    return None


def _collapse_runs(runs: Iterable[str]) -> str:
    """Turn a set of run labels into something like ``runs 01–03, 07``."""
    runs = sorted(set(runs))
    if not runs:
        return ""
    label = "run" if len(runs) == 1 else "runs"
    try:
        numbers = sorted(int(run) for run in runs)
    except ValueError:
        return f"{label} {', '.join(runs)}"
    width = max(len(run) for run in runs)
    groups: list[list[int]] = [[numbers[0]]]
    for number in numbers[1:]:
        if number == groups[-1][-1] + 1:
            groups[-1].append(number)
        else:
            groups.append([number])
    chunks = [
        f"{group[0]:0{width}d}"
        if len(group) == 1
        else f"{group[0]:0{width}d}–{group[-1]:0{width}d}"
        for group in groups
    ]
    return f"{label} {', '.join(chunks)}"


def _edge_lines(paths: Sequence[str]) -> list[str]:
    """Get a compact semantic descriptor for a set of files."""
    processings: set[str] = set()
    suffixes: set[str] = set()
    extensions: set[str] = set()
    runs: set[str] = set()
    for path in paths:
        processing, suffix, extension = _fname_bits(path)
        if processing is not None:
            processings.add(f"proc-{processing}")
        if suffix is not None:
            suffixes.add(suffix)
        elif extension is not None:
            extensions.add(extension)
        run = _fname_run(path)
        if run is not None:
            runs.add(run)
    descriptors = sorted(processings) + sorted(suffixes) + sorted(extensions)
    # e.g. proc- entities built from contrast names get long; the tooltip has the rest
    descriptors = [d if len(d) <= 24 else f"{d[:23]}…" for d in descriptors]
    if len(descriptors) > 3:  # e.g. one proc- entity per contrast; keep the label sane
        descriptors = descriptors[:3] + [f"+{len(descriptors) - 3} more"]
    lines = [" ".join(descriptors)] if descriptors else [f"{len(paths)} files"]
    run_line = _collapse_runs(runs)
    if run_line:
        lines.append(run_line)
    return lines


def _build_flow_graph(entries: Sequence[FlowEntryT]) -> _Graph:
    """Build the step graph implied by a set of recorded step calls."""
    produced_by: dict[str, set[str]] = dict()
    step_paths: dict[str, set[str]] = dict()
    for entry in entries:
        for path in entry["out_files"].values():
            produced_by.setdefault(path, set()).add(entry["step"])
        step_paths.setdefault(entry["step"], set()).update(entry["out_files"].values())

    edge_paths: dict[tuple[str, str], set[str]] = dict()
    for entry in entries:
        for path in entry["in_files"].values():
            # Files nobody produced came from outside the pipeline (raw BIDS data).
            for src in produced_by.get(path, {_SOURCE_ID}):
                if src == entry["step"]:  # e.g. a reference run feeding later runs
                    continue
                edge_paths.setdefault((src, entry["step"]), set()).add(path)

    graph = _Graph()
    used_steps = {step for edge in edge_paths for step in edge}
    for step in sorted(used_steps):
        if step == _SOURCE_ID:
            node = _Node(id=_node_id(step), lines=[_SOURCE_LABEL], paths=[])
        else:
            node = _Node(
                id=_node_id(step),
                lines=_step_lines(step),
                paths=sorted(step_paths.get(step, set())),
            )
        graph.nodes.append(node)
    for ii, (src, dst) in enumerate(sorted(edge_paths)):
        paths = sorted(edge_paths[(src, dst)])
        graph.edges.append(
            _Edge(
                id=f"{_ID_PREFIX}-edge-{ii}",
                src=_node_id(src),
                dst=_node_id(dst),
                lines=_edge_lines(paths),
                paths=paths,
            )
        )
    return graph


def _report_flow_html(
    *, deriv_root: pathlib.Path, subject: str, session: str | None
) -> str | None:
    """Get the flow diagram HTML for one report, or None if nothing was recorded."""
    entries = _read_flow_entries(
        deriv_root=deriv_root, subject=subject, session=session
    )
    graph = _build_flow_graph(entries)
    if not graph.edges:
        return None
    # No links on the nodes: mne.Report derives its anchors from content titles, and
    # nothing maps a pipeline step to the titles it happens to add to the report.
    return _graph_html(_layout_graph(graph), source_id=_node_id(_SOURCE_ID))


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
