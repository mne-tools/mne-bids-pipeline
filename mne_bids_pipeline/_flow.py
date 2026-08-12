"""Auto-generated pipeline flow diagrams for the HTML reports.

Steps record what they read and wrote (see ``_run.py``); this module turns those
recordings into the step dependency graph and its labels. The generic layout and
SVG rendering live in ``_graph.py``.
"""

import pathlib
import re
from collections.abc import Sequence
from typing import Any

from filelock import FileLock
from mne_bids import get_entities_from_fname

from ._graph import _ID_PREFIX, _Edge, _Graph, _graph_html, _layout_graph, _Node
from ._io import _read_json, _write_json
from ._logging import _collapse_runs, _shorten_paths
from .typing import TypedDict

_FLOW_DIRNAME = ".pipeline_flow"
_FLOW_VERSION = 1
_SOURCE_ID = "__bids_input__"
_SOURCE_LABEL = "BIDS raw data"


class FlowEntryT(TypedDict):
    """One recorded call of a pipeline step's worker function."""

    step: str
    func: str
    title: str | None
    subject: str | None
    session: str | None
    run: str | None
    task: str | None
    duration: float | None  # None (with finished/cached) until the call completes
    finished: str | None
    cached: bool | None
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


def _parse_flow_file(fname: pathlib.Path) -> dict[str, Any]:
    """Parse one recording file; the caller holds the lock when it matters."""
    if not fname.is_file():
        return dict()
    try:
        content = _read_json(fname)
    except ValueError:  # includes JSONDecodeError
        return dict()
    return content if content.get("version", None) == _FLOW_VERSION else dict()


def _write_flow_entry(
    *,
    deriv_root: pathlib.Path,
    entry: FlowEntryT,
    roots: dict[str, str] | None = None,
    only_if_new: bool = False,
) -> FlowEntryT:
    """Merge a step call into the on-disk recording; get back the stored entry."""
    fname = _flow_fname(deriv_root=deriv_root, subject=entry["subject"])
    fname.parent.mkdir(parents=True, exist_ok=True)
    # Steps parallelize over runs within a subject, so concurrent worker processes
    # read-modify-write the same file; the lock prevents lost/torn updates
    with FileLock(f"{fname}.lock"):
        content = _parse_flow_file(fname)
        entries: dict[str, FlowEntryT] = content.get("entries", dict())
        have_roots: dict[str, str] = content.get("roots", dict())
        key = _flow_entry_key(entry)
        old = entries.get(key, None)
        if only_if_new and old is not None:
            return old
        if (
            entry["cached"]
            and old is not None
            and not old.get("cached", False)
            and old.get("finished") is not None
            and old.get("out_files") == entry["out_files"]
        ):
            # A cache hit replays the same computation, so keep when the original
            # run happened and how long it took rather than the cache-check timing
            entry = dict(  # type: ignore[assignment]
                entry,
                duration=old["duration"],
                finished=old["finished"],
                cached=old["cached"],
            )
        entries[key] = entry
        have_roots.update(roots or dict())
        content = dict(version=_FLOW_VERSION, roots=have_roots, entries=entries)
        _write_json(fname, content, indent=1)
    return entry


def _flow_fnames(*, deriv_root: pathlib.Path, subject: str) -> list[pathlib.Path]:
    fnames = [_flow_fname(deriv_root=deriv_root, subject=None)]
    if subject == "average":
        # Group steps read individual subjects' files, so the group report needs every
        # subject's recording to attribute those inputs to the steps that made them.
        fnames += sorted(_flow_dir(deriv_root).glob("sub-*.json"))
    else:
        fnames.append(_flow_fname(deriv_root=deriv_root, subject=subject))
    return fnames


def _read_flow(
    *, deriv_root: pathlib.Path, subject: str, session: str | None
) -> tuple[list[FlowEntryT], dict[str, str]]:
    """Get the recorded step calls and roots relevant for one report."""
    entries: list[FlowEntryT] = list()
    roots: dict[str, str] = dict()
    for fname in _flow_fnames(deriv_root=deriv_root, subject=subject):
        if not fname.is_file():  # also avoids creating lock files on pure reads
            continue
        with FileLock(f"{fname}.lock"):
            content = _parse_flow_file(fname)
        for entry in content.get("entries", dict()).values():
            if entry["session"] in (None, session):
                entries.append(entry)
        roots.update(content.get("roots", dict()))
    entries.sort(key=_flow_entry_key)
    return entries, roots


# -- Graph -------------------------------------------------------------------

# Color categories for the step groups; init/freesurfer share one (dataset setup)
_CATEGORIES = {
    "init": "init",
    "freesurfer": "init",
    "preprocessing": "preproc",
    "sensor": "sensor",
    "source": "source",
}
# Categorical hues (validated for CVD + the report's white surface); identity is
# never color-alone since every node names its category in text
_FLOW_CSS = """
svg.mbp-flow .mbp-flow-source .mbp-flow-box { stroke-dasharray: 4 3; }
svg.mbp-flow .mbp-flow-cat-init .mbp-flow-box { stroke: #2a78d6; fill: #2a78d6; }
svg.mbp-flow .mbp-flow-cat-preproc .mbp-flow-box { stroke: #eb6834; fill: #eb6834; }
svg.mbp-flow .mbp-flow-cat-sensor .mbp-flow-box { stroke: #1baf7a; fill: #1baf7a; }
svg.mbp-flow .mbp-flow-cat-source .mbp-flow-box { stroke: #eda100; fill: #eda100; }
svg.mbp-flow [class*=" mbp-flow-cat-"] .mbp-flow-box {
  fill-opacity: 0.12;
  stroke-width: 1.2;
}
"""


def _node_id(step: str) -> str:
    return f"{_ID_PREFIX}-node-{re.sub(r'[^A-Za-z0-9_-]+', '-', step)}"


def _step_lines(step: str) -> list[str]:
    group, _, name = step.partition("/")
    return [group, name] if name else [group]


def _fname_bits(path: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Get the (processing, suffix, extension, run) descriptors of a filename."""
    name = pathlib.Path(path).name
    stem, _, _ = name.partition(".")
    extension = name[len(stem) :] or None
    parts = stem.split("_")
    if not any("-" in part for part in parts):  # not a BIDS-style name
        return None, None, extension, None
    entities = get_entities_from_fname(name, on_error="ignore")
    suffix = parts[-1] if "-" not in parts[-1] else None
    return entities.get("processing"), suffix, extension, entities.get("run")


def _edge_lines(paths: Sequence[str]) -> list[str]:
    """Get a compact semantic descriptor for a set of files."""
    processings: set[str] = set()
    suffixes: set[str] = set()
    extensions: set[str] = set()
    runs: set[str] = set()
    for path in paths:
        processing, suffix, extension, run = _fname_bits(path)
        if processing is not None:
            processings.add(f"proc-{processing}")
        if suffix is not None:
            suffixes.add(suffix)
        elif extension is not None:
            extensions.add(extension)
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


def _node_tooltip(entries: Sequence[FlowEntryT], out_paths: Sequence[str]) -> list[str]:
    """Summarize a step's recorded calls for its hover tooltip."""
    lines: list[str] = list()
    titles = [title for entry in entries if (title := entry.get("title"))]
    if titles:
        lines.append(titles[0])
    timed = [entry for entry in entries if entry.get("duration") is not None]
    # cached=True survives only when no original computation was ever recorded
    # (e.g. the recording was deleted but the joblib cache kept), see
    # _write_flow_entry; normally timing describes the last real computation
    fresh = [entry for entry in timed if not entry.get("cached")]
    if fresh:
        total = sum(entry["duration"] or 0.0 for entry in fresh)
        took = f"{total:.1f} s" if total < 60 else f"{total / 60:.1f} min"
        calls = f" over {len(fresh)} calls" if len(fresh) > 1 else ""
        lines.append(f"took {took}{calls}")
        stamps = [stamp for entry in fresh if (stamp := entry.get("finished"))]
        if stamps:
            lines.append(f"completed {max(stamps)}")
    elif timed:
        lines.append("cached (original run not recorded)")
    if out_paths:
        lines.append("writes:")
        lines.extend(out_paths)
    return lines


def _build_flow_graph(
    entries: Sequence[FlowEntryT], roots: dict[str, str] | None = None
) -> _Graph:
    """Build the step graph implied by a set of recorded step calls."""
    roots = roots or dict()
    produced_by: dict[str, set[str]] = dict()
    step_paths: dict[str, set[str]] = dict()
    step_entries: dict[str, list[FlowEntryT]] = dict()
    for entry in entries:
        for path in entry["out_files"].values():
            produced_by.setdefault(path, set()).add(entry["step"])
        step_paths.setdefault(entry["step"], set()).update(entry["out_files"].values())
        step_entries.setdefault(entry["step"], []).append(entry)

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
            node = _Node(
                id=_node_id(step),
                lines=[_SOURCE_LABEL],
                paths=[f"<{name}> = {path}" for name, path in sorted(roots.items())],
                klass="mbp-flow-source",
            )
        else:
            category = _CATEGORIES.get(step.partition("/")[0], "")
            node = _Node(
                id=_node_id(step),
                lines=_step_lines(step),
                paths=_node_tooltip(
                    step_entries.get(step, []),
                    _shorten_paths(sorted(step_paths.get(step, set())), roots),
                ),
                klass=f"mbp-flow-cat-{category}" if category else "",
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
                paths=_shorten_paths(paths, roots),
            )
        )
    return graph


def _report_flow_html(
    *, deriv_root: pathlib.Path, subject: str, session: str | None
) -> str | None:
    """Get the flow diagram HTML for one report, or None if nothing was recorded."""
    entries, roots = _read_flow(deriv_root=deriv_root, subject=subject, session=session)
    roots["deriv_root"] = str(deriv_root)
    graph = _build_flow_graph(entries, roots)
    if not graph.edges:
        return None
    # No links on the nodes: mne.Report derives its anchors from content titles, and
    # nothing maps a pipeline step to the titles it happens to add to the report.
    return _graph_html(
        _layout_graph(graph), extra_css=_FLOW_CSS, label="Pipeline flow diagram"
    )
