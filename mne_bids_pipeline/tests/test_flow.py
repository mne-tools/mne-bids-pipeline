"""Test the pipeline flow diagram."""

import pathlib
import shutil
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any

import pytest
from mne_bids import BIDSPath

from mne_bids_pipeline._flow import (
    FlowEntryT,
    _build_flow_graph,
    _read_flow,
    _report_flow_html,
    _write_flow_entry,
)
from mne_bids_pipeline._graph import _Graph, _graph_html, _layout_graph, _Node
from mne_bids_pipeline._logging import _collapse_runs, _shorten_paths
from mne_bids_pipeline._run import (
    _flow_files,
    _prep_out_files_path,
    _ran_when,
    failsafe_run,
)
from mne_bids_pipeline.typing import InFilesPathT, InFilesT, OutFilesT

SVG_NS = "{http://www.w3.org/2000/svg}"
RUNS = ("01", "02", "03")
BIDS = "/bids/sub-01_task-av"
DERIV = "/deriv/sub-01_task-av"
EPO = f"{DERIV}_epo.fif"


def _filt_raw(run: str) -> str:
    return f"{DERIV}_run-{run}_proc-filt_raw.fif"


def _read01(deriv_root: pathlib.Path, session: str | None = None) -> list[FlowEntryT]:
    return _read_flow(deriv_root=deriv_root, subject="01", session=session)[0]


def _find_node(graph: _Graph, line: str) -> _Node:
    return next(node for node in graph.nodes if line in node.lines)


def _entry(
    step: str,
    *,
    func: str = "func",
    title: str | None = None,
    subject: str | None = "01",
    session: str | None = None,
    run: str | None = None,
    task: str | None = "av",
    duration: float | None = None,
    finished: str | None = None,
    cached: bool | None = None,
    in_files: dict[str, str] | None = None,
    out_files: dict[str, str] | None = None,
) -> FlowEntryT:
    return {
        "step": step,
        "func": func,
        "title": title,
        "subject": subject,
        "session": session,
        "run": run,
        "task": task,
        "duration": duration,
        "finished": finished,
        "cached": cached,
        "in_files": in_files or dict(),
        "out_files": out_files or dict(),
    }


def _pipeline_entries() -> list[FlowEntryT]:
    """Get a small but representative recording: 3 runs, a fan-out, a dead end."""
    entries = list()
    for run in RUNS:
        entries.append(
            _entry(
                "preprocessing/_04_frequency_filter",
                func="filter_data",
                title="Frequency filter",
                run=run,
                duration=2.0,
                finished=f"2026-08-12 09:0{run[-1]}:00",
                cached=run == "01",
                in_files={"raw": f"{BIDS}_run-{run}_meg.fif"},
                out_files={"raw": _filt_raw(run)},
            )
        )
    entries.append(
        _entry(
            "preprocessing/_07_make_epochs",
            func="make_epochs",
            in_files={f"raw_{run}": _filt_raw(run) for run in RUNS},
            out_files={"epo": EPO},
        )
    )
    for step in ("sensor/_01_make_evoked", "sensor/_06_make_cov"):
        entries.append(
            _entry(
                step,
                in_files={"epo": EPO},
                out_files={"out": f"{DERIV}_{step[-3:]}.fif"},
            )
        )
    # Produces something nobody reads, so it should not show up at all
    entries.append(
        _entry(
            "init/_01_init_derivatives_dir",
            func="init_dataset",
            subject=None,
            task=None,
            out_files={"json": "/deriv/dataset_description.json"},
        )
    )
    return entries


@pytest.mark.parametrize(
    "runs, want",
    [
        ([], ""),
        (["01"], "run 01"),
        (["01", "02", "03", "04", "05", "06"], "runs 01–06"),
        (["03", "01", "02", "07"], "runs 01–03, 07"),
        (["1", "2"], "runs 1–2"),
        (["rest", "noise"], "runs noise, rest"),
    ],
)
def test_collapse_runs(runs: list[str], want: str) -> None:
    """Test that run labels collapse into ranges."""
    assert _collapse_runs(runs) == want


def test_flow_storage(tmp_path: pathlib.Path) -> None:
    """Test the write/read roundtrip, roots, and missing/corrupt recordings."""
    flow_dir = tmp_path / ".pipeline_flow"
    assert _read_flow(deriv_root=tmp_path, subject="01", session=None) == ([], {})
    assert _report_flow_html(deriv_root=tmp_path, subject="01", session=None) is None
    roots = {"bids_root": "/bids"}
    entries = _pipeline_entries()
    for entry in entries:
        _write_flow_entry(deriv_root=tmp_path, entry=entry, roots=roots)
    # Re-running a step must overwrite its own entry rather than add another
    changed = _entry(
        entries[0]["step"],
        func=entries[0]["func"],
        run=entries[0]["run"],
        in_files=entries[0]["in_files"],
        out_files={"raw": "/deriv/other.fif"},
    )
    _write_flow_entry(deriv_root=tmp_path, entry=changed)

    got, got_roots = _read_flow(deriv_root=tmp_path, subject="01", session=None)
    assert got_roots == roots
    assert len(got) == len(entries)
    assert changed in got
    assert entries[0] not in got
    # The dataset-level entry lives in its own file but is relevant to every subject
    assert sum(entry["subject"] is None for entry in got) == 1
    assert (flow_dir / "dataset.json").is_file()
    assert (flow_dir / "sub-01.json").is_file()
    dataset_only = [entry for entry in got if entry["subject"] is None]
    assert _read_flow(deriv_root=tmp_path, subject="02", session=None)[0] == (
        dataset_only
    )
    # Entries of other sessions are filtered out
    other = _entry("sensor/_01_make_evoked", session="t2")
    _write_flow_entry(deriv_root=tmp_path, entry=other)
    assert other not in _read01(tmp_path, "t1")
    assert other in _read01(tmp_path, "t2")

    assert _shorten_paths(
        ["/bids/sub-01_meg.fif", f"{tmp_path}/a.fif", "/elsewhere/b.fif"],
        dict(roots, deriv_root=str(tmp_path)),
    ) == ["<bids_root>/sub-01_meg.fif", "<deriv_root>/a.fif", "/elsewhere/b.fif"]

    # A corrupt per-subject file is skipped rather than fatal
    (flow_dir / "sub-01.json").write_text("not json")
    assert _read01(tmp_path) == dataset_only


def test_flow_graph() -> None:
    """Test that edges follow produced/consumed and node tooltips summarize calls."""
    graph = _build_flow_graph(_pipeline_entries())
    labels = {node.id: " ".join(node.lines) for node in graph.nodes}
    edges = {(labels[edge.src], labels[edge.dst]): edge.lines for edge in graph.edges}
    assert edges == {
        ("BIDS raw data", "preprocessing _04_frequency_filter"): [
            "meg",
            "runs 01–03",
        ],
        ("preprocessing _04_frequency_filter", "preprocessing _07_make_epochs"): [
            "proc-filt raw",
            "runs 01–03",
        ],
        ("preprocessing _07_make_epochs", "sensor _01_make_evoked"): ["epo"],
        ("preprocessing _07_make_epochs", "sensor _06_make_cov"): ["epo"],
    }
    # init only writes a file nobody consumes, so it contributes no node
    assert not any("init" in label for label in labels.values())
    # The full paths are kept for the tooltips
    edge = next(edge for edge in graph.edges if edge.lines[0] == "proc-filt raw")
    assert len(edge.paths) == 3
    assert edge.paths[0] == _filt_raw("01")

    # Node tooltips: title, then timing (the cached=True run-01 call has no recorded
    # original so timing sums the other two), then the outputs
    node = _find_node(graph, "_04_frequency_filter")
    assert node.paths == [
        "Frequency filter",
        "took 4.0 s over 2 calls",
        "completed 2026-08-12 09:03:00",
        "writes:",
        *(_filt_raw(run) for run in RUNS),
    ]
    # All-cached with no recorded original: say so instead of showing check times
    entries = _pipeline_entries()
    for entry in entries:
        entry["cached"] = True
    graph = _build_flow_graph(entries)
    node = _find_node(graph, "_04_frequency_filter")
    assert node.paths[1] == "cached (original run not recorded)"


def test_flow_edge_logic() -> None:
    """Test self-edge suppression and the routing of layer-skipping edges."""
    # A step reading its own output (e.g. a reference run) adds no self-edge
    entries = [
        _entry(
            "preprocessing/_03_maxfilter",
            run="01",
            out_files={"raw": "/deriv/sub-01_run-01_proc-sss_raw.fif"},
        ),
        _entry(
            "preprocessing/_03_maxfilter",
            run="02",
            in_files={"ref": "/deriv/sub-01_run-01_proc-sss_raw.fif"},
            out_files={"raw": "/deriv/sub-01_run-02_proc-sss_raw.fif"},
        ),
    ]
    assert _build_flow_graph(entries).edges == []

    # Edges skipping a layer are routed through the layers in between
    entries = _pipeline_entries()
    entries.append(_entry("sensor/_06_make_cov", in_files={"raw": _filt_raw("01")}))
    graph = _layout_graph(_build_flow_graph(entries))
    nodes = {node.id: node for node in graph.nodes}
    long_edge = next(
        edge
        for edge in graph.edges
        if nodes[edge.dst].layer - nodes[edge.src].layer > 1
    )
    assert len(long_edge.points) == 3  # one routing point in the skipped layer
    assert nodes[long_edge.src].y < long_edge.points[1][1] < nodes[long_edge.dst].y


def test_flow_layout() -> None:
    """Test that the layout respects the topology and does not overlap."""
    graph = _layout_graph(_build_flow_graph(_pipeline_entries()))
    nodes = {node.id: node for node in graph.nodes}
    assert [node.layer for node in graph.nodes] == [0, 1, 2, 3, 3]
    for edge in graph.edges:
        src, dst = nodes[edge.src], nodes[edge.dst]
        assert src.layer < dst.layer
        assert src.y < dst.y
        assert edge.points[0][1] < edge.points[-1][1]
    for layer in range(4):
        spans = sorted(
            (node.x - node.width / 2, node.x + node.width / 2)
            for node in graph.nodes
            if node.layer == layer
        )
        for (_, right), (left, _) in zip(spans[:-1], spans[1:]):
            assert right <= left
        assert spans[0][0] >= 0 and spans[-1][1] <= graph.width
    assert all(0 < node.y < graph.height for node in graph.nodes)

    # Stages never interleave: steps depending only on FreeSurfer files still land
    # below every preprocessing/sensor step (banded ranks)
    entries = _pipeline_entries()
    entries += [
        _entry(
            "freesurfer/_01_recon_all",
            in_files={"t1": "/bids/sub-01_T1w.nii.gz"},
            out_files={"white": "/fs/sub-01/surf/lh.white"},
        ),
        _entry(
            "source/_01_make_bem_surfaces",
            in_files={"white": "/fs/sub-01/surf/lh.white"},
            out_files={"bem": "/fs/sub-01/bem/inner_skull.surf"},
        ),
    ]
    graph = _layout_graph(_build_flow_graph(entries))
    layer = {" ".join(node.lines): node.layer for node in graph.nodes}
    highest = max(
        value
        for key, value in layer.items()
        if key.startswith(("preprocessing", "sensor"))
    )
    assert layer["freesurfer _01_recon_all"] > highest
    assert layer["source _01_make_bem_surfaces"] > layer["freesurfer _01_recon_all"]
    # Distant bands get their own copy of the BIDS source node (here for the
    # freesurfer band's T1w input) instead of one edge spanning the diagram
    assert sum(node.klass == "mbp-flow-source" for node in graph.nodes) == 2


def test_flow_svg() -> None:
    """Test that the emitted SVG is well-formed and self-consistent."""
    graph = _layout_graph(_build_flow_graph(_pipeline_entries()))
    html = _graph_html(graph)
    assert "<script>" in html
    svg = ET.fromstring(html[html.index("<svg") : html.index("</svg>") + 6])
    assert svg.get("class") == "mbp-flow"

    groups = svg.findall(f".//{SVG_NS}g")
    node_els = [el for el in groups if "mbp-flow-node" in el.get("class", "").split()]
    edge_els = [el for el in groups if el.get("class") == "mbp-flow-edge"]
    assert len(node_els) == len(graph.nodes) == 5
    assert len(edge_els) == len(graph.edges) == 4
    assert sum("mbp-flow-source" in el.get("class", "") for el in node_els) == 1
    # Node links would need report anchors, which we cannot derive; see _flow.py
    assert svg.find(f".//{SVG_NS}a") is None

    ids = {el.get("id") for el in node_els + edge_els}
    assert None not in ids
    for el in node_els:
        for attr in ("ancestors", "descendants", "edges"):
            related = (el.get(f"data-flow-{attr}") or "").split()
            assert set(related) <= ids
    filt = next(
        el
        for el in node_els
        if "_04_frequency_filter" in "".join(el.itertext())  # tspans
    )
    assert len(filt.get("data-flow-ancestors", "").split()) == 1  # the raw data
    assert len(filt.get("data-flow-descendants", "").split()) == 3
    assert len(filt.get("data-flow-edges", "").split()) == 4

    titles = [el.text or "" for el in svg.findall(f".//{SVG_NS}title")]
    assert any(_filt_raw("01") in t for t in titles)

    classes = [el.get("class", "") for el in node_els]
    assert sum("mbp-flow-cat-preproc" in k for k in classes) == 2
    assert sum("mbp-flow-cat-sensor" in k for k in classes) == 2
    # The two identically-labeled fan edges share one rendered label
    label_els = [
        el
        for el in svg.findall(f".//{SVG_NS}text")
        if el.get("class") == "mbp-flow-edge-label"
    ]
    assert len(label_els) == len(edge_els) - 1 == 3


_N_CALLS: list[str] = list()


def _get_input_fnames_flow(*, cfg: SimpleNamespace, **kwargs: Any) -> InFilesT:
    return dict(raw=cfg.raw)


def _get_output_fnames_flow(*, cfg: SimpleNamespace, **kwargs: Any) -> InFilesPathT:
    return dict(filt=cfg.out)


def _flow_step_impl(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    _N_CALLS.append(subject)
    in_files.pop("raw")
    cfg.out.write_text("filtered")
    return _prep_out_files_path(exec_params=exec_params, out_files=dict(filt=cfg.out))


_flow_step = failsafe_run(get_input_fnames=_get_input_fnames_flow)(_flow_step_impl)
_flow_step_out = failsafe_run(
    get_input_fnames=_get_input_fnames_flow,
    get_output_fnames=_get_output_fnames_flow,
)(_flow_step_impl)


@pytest.fixture
def flow_kwargs(tmp_path: pathlib.Path) -> dict[str, Any]:
    """Get kwargs for a fake pipeline step writing into a fresh derivatives dir."""
    _N_CALLS.clear()
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir()
    raw = tmp_path / "sub-01_task-av_run-01_meg.fif"
    raw.write_text("raw")
    cfg = SimpleNamespace(
        raw=raw, out=deriv_root / "sub-01_task-av_run-01_filt_raw.fif"
    )
    exec_params = SimpleNamespace(
        on_error="abort",
        deriv_root=deriv_root,
        memory_location=True,
        memory_subdir="joblib",
        memory_verbose=0,
        memory_file_method="mtime",
        ignore_warnings=(),
    )
    return dict(
        cfg=cfg,
        exec_params=exec_params,
        subject="01",
        session=None,
        run="01",
        task="av",
    )


def _recorded(flow_kwargs: dict[str, Any]) -> list[FlowEntryT]:
    return _read01(flow_kwargs["exec_params"].deriv_root)


@pytest.mark.parametrize("memory_location", (True, False))
def test_flow_recorder(flow_kwargs: dict[str, Any], memory_location: bool) -> None:
    """Test that the step wrapper records files on fresh runs and on cache hits."""
    # The wrapper snapshots in_files/out_files through _flow_files normalization
    bids_path = BIDSPath(
        subject="01",
        root="/bids",
        datatype="meg",
        suffix="meg",
        extension=".fif",
        check=False,
    )
    assert _flow_files(
        {
            "path": pathlib.Path("/deriv/a.fif"),
            "bids": bids_path,
            "hashed": ("/deriv/b.fif", 1234.0),
            "__unknown_inputs__": "custom cov",
            "junk": None,
        }
    ) == {
        "path": "/deriv/a.fif",
        "bids": str(bids_path.fpath),
        "hashed": "/deriv/b.fif",
    }

    flow_kwargs["exec_params"].memory_location = memory_location
    cfg = flow_kwargs["cfg"]
    _flow_step(**flow_kwargs)
    want = {
        "step": "tests/test_flow",
        "func": "_flow_step_impl",
        "title": "Test the pipeline flow diagram",
        "subject": "01",
        "session": None,
        "run": "01",
        "task": "av",
        "in_files": {"raw": str(cfg.raw)},
        "out_files": {"filt": str(cfg.out)},
    }

    def _check(cached: bool) -> tuple[float, str]:
        (entry,) = (dict(e) for e in _recorded(flow_kwargs))
        duration = entry.pop("duration")
        finished = entry.pop("finished")
        assert isinstance(duration, float) and duration >= 0.0
        assert isinstance(finished, str)
        assert entry.pop("cached") is cached
        assert entry == want
        return duration, finished

    first = _check(cached=False)
    _flow_step(**flow_kwargs)
    assert len(_N_CALLS) == (1 if memory_location else 2)  # cache hit the 2nd time
    second = _check(cached=False)  # ... still recorded, as the original computation
    if memory_location:
        assert second == first  # the cache hit preserved the original timing
    (entry,) = _recorded(flow_kwargs)
    assert _ran_when(entry) == f", ran {second[1][:16]}"
    deriv_root = flow_kwargs["exec_params"].deriv_root
    if memory_location:
        # A cache hit whose original computation is on record skips the write
        fname = deriv_root / ".pipeline_flow" / "sub-01.json"
        mtime = fname.stat().st_mtime_ns
        _flow_step(**flow_kwargs)
        assert fname.stat().st_mtime_ns == mtime
        # With the recording gone but the joblib cache kept, the original run time
        # is unknowable: the entry marks that, and the log helper returns nothing
        shutil.rmtree(deriv_root / ".pipeline_flow")
        _flow_step(**flow_kwargs)
        (entry,) = _recorded(flow_kwargs)
        assert entry["cached"] is True
        assert _ran_when(entry) == ""


def test_flow_recorder_edge_cases(
    flow_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the output-exists short circuit and recording-failure tolerance."""
    # A step skipped because its outputs already exist is still recorded
    flow_kwargs["cfg"].out.write_text("stale")
    _flow_step_out(**flow_kwargs)
    assert _N_CALLS == []
    (entry,) = _recorded(flow_kwargs)
    assert entry["in_files"] == {"raw": str(flow_kwargs["cfg"].raw)}
    assert entry["out_files"] == {"filt": str(flow_kwargs["cfg"].out)}

    # A broken recording must not take the pipeline down with it
    shutil.rmtree(flow_kwargs["exec_params"].deriv_root / ".pipeline_flow")

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("no disk for you")

    monkeypatch.setattr("mne_bids_pipeline._run._write_flow_entry", _boom)
    _flow_step(**flow_kwargs)
    assert _N_CALLS == ["01"]
    assert _recorded(flow_kwargs) == []
