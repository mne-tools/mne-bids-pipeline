"""Test the pipeline flow diagram."""

import pathlib
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any

import mne
import pytest
from mne_bids import BIDSPath

from mne_bids_pipeline._flow import (
    FlowEntryT,
    _build_flow_graph,
    _collapse_runs,
    _flow_files,
    _read_flow_entries,
    _read_flow_roots,
    _report_flow_html,
    _shorten_paths,
    _write_flow_entry,
)
from mne_bids_pipeline._graph import _graph_html, _layout_graph
from mne_bids_pipeline._report import _add_flow_diagram
from mne_bids_pipeline._run import _prep_out_files_path, failsafe_run
from mne_bids_pipeline.typing import InFilesT, OutFilesT

SVG_NS = "{http://www.w3.org/2000/svg}"


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
    for run in ("01", "02", "03"):
        entries.append(
            _entry(
                "preprocessing/_04_frequency_filter",
                func="filter_data",
                run=run,
                in_files={"raw": f"/bids/sub-01_task-av_run-{run}_meg.fif"},
                out_files={"raw": f"/deriv/sub-01_task-av_run-{run}_proc-filt_raw.fif"},
            )
        )
    entries.append(
        _entry(
            "preprocessing/_07_make_epochs",
            func="make_epochs",
            in_files={
                f"raw_{run}": f"/deriv/sub-01_task-av_run-{run}_proc-filt_raw.fif"
                for run in ("01", "02", "03")
            },
            out_files={"epo": "/deriv/sub-01_task-av_epo.fif"},
        )
    )
    for step in ("sensor/_01_make_evoked", "sensor/_06_make_cov"):
        entries.append(
            _entry(
                step,
                in_files={"epo": "/deriv/sub-01_task-av_epo.fif"},
                out_files={"out": f"/deriv/sub-01_task-av_{step[-3:]}.fif"},
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


def test_flow_files() -> None:
    """Test normalization of the in_files/out_files mappings."""
    bids_path = BIDSPath(
        subject="01",
        root="/bids",
        datatype="meg",
        suffix="meg",
        extension=".fif",
        check=False,
    )
    got = _flow_files(
        {
            "path": pathlib.Path("/deriv/a.fif"),
            "bids": bids_path,
            "hashed": ("/deriv/b.fif", 1234.0),
            "__unknown_inputs__": "custom cov",
            "junk": None,
        }
    )
    assert got == {
        "path": "/deriv/a.fif",
        "bids": str(bids_path.fpath),
        "hashed": "/deriv/b.fif",
    }


def test_flow_recording_roundtrip(tmp_path: pathlib.Path) -> None:
    """Test that recorded entries survive a write/read cycle without duplicating."""
    entries = _pipeline_entries()
    for entry in entries:
        _write_flow_entry(deriv_root=tmp_path, entry=entry)
    # Re-running a step must overwrite its own entry rather than add another
    changed = _entry(
        entries[0]["step"],
        func=entries[0]["func"],
        run=entries[0]["run"],
        in_files=entries[0]["in_files"],
        out_files={"raw": "/deriv/other.fif"},
    )
    _write_flow_entry(deriv_root=tmp_path, entry=changed)

    got = _read_flow_entries(deriv_root=tmp_path, subject="01", session=None)
    assert len(got) == len(entries)
    assert changed in got
    assert entries[0] not in got
    # The dataset-level entry lives in its own file but is relevant to every subject
    assert sum(entry["subject"] is None for entry in got) == 1
    assert (tmp_path / ".pipeline_flow" / "dataset.json").is_file()
    assert (tmp_path / ".pipeline_flow" / "sub-01.json").is_file()

    dataset_only = [entry for entry in got if entry["subject"] is None]
    assert _read_flow_entries(deriv_root=tmp_path, subject="02", session=None) == (
        dataset_only
    )
    # Entries of other sessions are filtered out
    other = _entry("sensor/_01_make_evoked", session="t2")
    _write_flow_entry(deriv_root=tmp_path, entry=other)
    assert other not in _read_flow_entries(
        deriv_root=tmp_path, subject="01", session="t1"
    )
    assert other in _read_flow_entries(deriv_root=tmp_path, subject="01", session="t2")


def test_flow_recording_missing(tmp_path: pathlib.Path) -> None:
    """Test that an absent or unreadable recording is not fatal."""
    assert _read_flow_entries(deriv_root=tmp_path, subject="01", session=None) == []
    assert _report_flow_html(deriv_root=tmp_path, subject="01", session=None) is None
    fname = tmp_path / ".pipeline_flow" / "sub-01.json"
    fname.parent.mkdir()
    fname.write_text("not json")
    assert _read_flow_entries(deriv_root=tmp_path, subject="01", session=None) == []


def test_flow_roots(tmp_path: pathlib.Path) -> None:
    """Test that recorded roots shorten the tooltip paths."""
    roots = {"bids_root": "/bids"}
    for entry in _pipeline_entries():
        _write_flow_entry(deriv_root=tmp_path, entry=entry, roots=roots)
    assert _read_flow_roots(deriv_root=tmp_path, subject="01") == roots
    assert _shorten_paths(
        ["/bids/sub-01_meg.fif", f"{tmp_path}/a.fif", "/elsewhere/b.fif"],
        dict(roots, deriv_root=str(tmp_path)),
    ) == ["<bids_root>/sub-01_meg.fif", "<deriv_root>/a.fif", "/elsewhere/b.fif"]
    html = _report_flow_html(deriv_root=tmp_path, subject="01", session=None)
    assert html is not None
    assert "&lt;bids_root&gt;/sub-01_task-av_run-01_meg.fif" in html
    # The source node's tooltip lists the roots themselves
    assert "&lt;bids_root&gt; = /bids" in html
    assert f"&lt;deriv_root&gt; = {tmp_path}" in html


def test_flow_graph_build() -> None:
    """Test that edges follow the produced/consumed relationships."""
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
    assert edge.paths[0] == "/deriv/sub-01_task-av_run-01_proc-filt_raw.fif"


def test_flow_node_tooltip() -> None:
    """Test that node tooltips summarize title, timing, cache state, and outputs."""
    entries = [
        _entry(
            "preprocessing/_04_frequency_filter",
            title="Frequency filter",
            run=run,
            duration=2.0,
            finished=f"2026-08-12 09:0{run[-1]}:00",
            cached=run == "01",
            in_files={"raw": f"/bids/sub-01_run-{run}_meg.fif"},
            out_files={"raw": f"/deriv/sub-01_run-{run}_proc-filt_raw.fif"},
        )
        for run in ("01", "02", "03")
    ]
    graph = _build_flow_graph(entries)
    node = next(node for node in graph.nodes if "preprocessing" in node.lines)
    assert node.paths[:4] == [
        "Frequency filter",
        "took 6.0 s over 3 calls (1/3 from cache)",
        "finished 2026-08-12 09:03:00",
        "writes:",
    ]
    assert node.paths[4:] == [
        f"/deriv/sub-01_run-{run}_proc-filt_raw.fif" for run in ("01", "02", "03")
    ]


def test_flow_graph_no_self_edges() -> None:
    """Test that a step reading its own output does not add a self-edge."""
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


def test_flow_layout_long_edges() -> None:
    """Test that edges skipping a layer are routed through the layers in between."""
    entries = _pipeline_entries()
    entries.append(
        _entry(
            "sensor/_06_make_cov",
            in_files={"raw": "/deriv/sub-01_task-av_run-01_proc-filt_raw.fif"},
        )
    )
    graph = _layout_graph(_build_flow_graph(entries))
    nodes = {node.id: node for node in graph.nodes}
    long_edge = next(
        edge
        for edge in graph.edges
        if nodes[edge.dst].layer - nodes[edge.src].layer > 1
    )
    assert len(long_edge.points) == 3  # one routing point in the skipped layer
    assert nodes[long_edge.src].y < long_edge.points[1][1] < nodes[long_edge.dst].y


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
    assert any("/deriv/sub-01_task-av_run-01_proc-filt_raw.fif" in t for t in titles)

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


def test_flow_report(tmp_path: pathlib.Path) -> None:
    """Test that the diagram makes it into a saved report."""
    report = mne.Report(title="sub-01")
    exec_params = SimpleNamespace(deriv_root=tmp_path)
    _add_flow_diagram(
        report=report, exec_params=exec_params, subject="01", session=None
    )
    assert report.html == []  # nothing recorded yet
    for entry in _pipeline_entries():
        _write_flow_entry(deriv_root=tmp_path, entry=entry)
    for _ in range(2):  # must refresh in place as more steps run
        _add_flow_diagram(
            report=report, exec_params=exec_params, subject="01", session=None
        )
    assert len(report.html) == 1  # replaced in place rather than appended

    fname = tmp_path / "report.html"
    report.save(fname, open_browser=False)
    content = fname.read_text(encoding="utf-8")
    assert content.count('<svg id="mbp-flow-svg"') == 1
    assert "_04_frequency_filter" in content


_N_CALLS: list[str] = list()


def _get_input_fnames_flow(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> InFilesT:
    return dict(raw=cfg.raw)


def _get_output_fnames_flow(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> InFilesT:
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
    deriv_root = flow_kwargs["exec_params"].deriv_root
    return _read_flow_entries(deriv_root=deriv_root, subject="01", session=None)


@pytest.mark.parametrize("memory_location", (True, False))
def test_flow_recorder(flow_kwargs: dict[str, Any], memory_location: bool) -> None:
    """Test that the step wrapper records files on fresh runs and on cache hits."""
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

    def _check(cached: bool) -> None:
        (entry,) = (dict(e) for e in _recorded(flow_kwargs))
        assert entry.pop("duration") >= 0.0
        assert entry.pop("finished") is not None
        assert entry.pop("cached") is cached
        assert entry == want

    _check(cached=False)
    _flow_step(**flow_kwargs)
    assert len(_N_CALLS) == (1 if memory_location else 2)  # cache hit the 2nd time
    _check(cached=memory_location)  # ... and still recorded


def test_flow_recorder_short_circuit(flow_kwargs: dict[str, Any]) -> None:
    """Test that a step skipped because its outputs exist is still recorded."""
    flow_kwargs["cfg"].out.write_text("stale")
    _flow_step_out(**flow_kwargs)
    assert _N_CALLS == []
    (entry,) = _recorded(flow_kwargs)
    assert entry["in_files"] == {"raw": str(flow_kwargs["cfg"].raw)}
    assert entry["out_files"] == {"filt": str(flow_kwargs["cfg"].out)}


def test_flow_recorder_failure(
    flow_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a broken recording does not take the pipeline down with it."""

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("no disk for you")

    monkeypatch.setattr("mne_bids_pipeline._run._write_flow_entry", _boom)
    _flow_step(**flow_kwargs)
    assert _N_CALLS == ["01"]
    assert _recorded(flow_kwargs) == []
