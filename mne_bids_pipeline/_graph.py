"""Layered-DAG layout and self-contained SVG rendering.

Domain-agnostic: everything here operates on ``_Node``/``_Edge``/``_Graph`` and
could draw any small DAG. The pipeline-specific graph construction lives in
``_flow.py``.

The layout is a from-scratch implementation of the classical Sugiyama framework
(layering, dummy vertices routing long edges, iterative barycenter crossing
reduction) per Sugiyama, Tagawa & Toda (1981), IEEE Trans SMC 11(2):109-125,
https://doi.org/10.1109/TSMC.1981.4308636 — simplified to longest-path layering
and width-aware center alignment, which suffice at this graph size.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

_ID_PREFIX = "mbp-flow"
_SVG_ID = f"{_ID_PREFIX}-svg"


@dataclass
class _Node:
    id: str
    lines: list[str]
    paths: list[str]
    klass: str = ""  # extra CSS class(es), e.g. a color category
    rank: int = 0  # layout band; higher ranks sit strictly below lower ones
    layer: int = 0
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    dummy: bool = False


@dataclass
class _Edge:
    id: str
    src: str
    dst: str
    lines: list[str]
    paths: list[str]
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class _Graph:
    nodes: list[_Node] = field(default_factory=list)
    edges: list[_Edge] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


# -- Layout ------------------------------------------------------------------

_FONT_SIZE = 12.0
_CHAR_WIDTH = 6.6  # rough advance width of the report's body font at _FONT_SIZE
_LINE_HEIGHT = 15.0
_NODE_PAD_X = 12.0
_NODE_PAD_Y = 12.0
_NODE_GAP = 26.0
_ROW_HEIGHT = 100.0
_BOLD_FACTOR = 1.1  # node labels are bold, so wider than _CHAR_WIDTH estimates
_MARGIN = 10.0
_ROW_MID = 21.0  # where a routing point sits within its row


def _text_width(lines: Sequence[str]) -> float:
    return max((len(line) for line in lines), default=0) * _CHAR_WIDTH


def _node_height(node: _Node) -> float:
    return 0.0 if node.dummy else len(node.lines) * _LINE_HEIGHT + _NODE_PAD_Y


def _min_sep(left: _Node, right: _Node) -> float:
    return left.width / 2 + _NODE_GAP + right.width / 2


def _min_x(node: _Node) -> float:
    return _MARGIN + node.width / 2


def _max_x(width: float, node: _Node) -> float:
    return width - _MARGIN - node.width / 2


def _assign_layers(graph: _Graph) -> None:
    # Bands (ranks) settle top to bottom, each starting below the previous one, so
    # e.g. pipeline stages never interleave even when dependencies would allow it;
    # an edge from a still-unsettled (higher-rank) node is simply not a constraint
    layers: dict[str, int] = dict()
    base = 0
    for rank in sorted({node.rank for node in graph.nodes}):
        band = {node.id for node in graph.nodes if node.rank == rank}
        for node_id in band:
            layers[node_id] = base
        # Longest-path layering by relaxation; the pass limit also bounds the
        # (never expected, but possible if a step reads and writes a shared file)
        # cycle.
        for _ in range(len(band)):
            changed = False
            for edge in graph.edges:
                src = layers.get(edge.src, None)
                if edge.dst not in band or src is None:
                    continue
                if layers[edge.dst] < src + 1:
                    layers[edge.dst] = src + 1
                    changed = True
            if not changed:
                break
        base = max(layers[node_id] for node_id in band) + 1
    for node in graph.nodes:
        node.layer = layers[node.id]


def _order_layers(
    nodes: Sequence[_Node], neighbors: dict[str, list[str]]
) -> list[list[_Node]]:
    n_layers = max((node.layer for node in nodes), default=0) + 1
    layers: list[list[_Node]] = [[] for _ in range(n_layers)]
    for node in nodes:
        layers[node.layer].append(node)
    for _ in range(4):
        index = {
            node.id: ii for layer in layers for ii, node in enumerate(layer)
        }  # barycenter positions
        for layer in layers:
            layer.sort(
                key=lambda node: (
                    sum(index[other] for other in neighbors.get(node.id, []))
                    / max(len(neighbors.get(node.id, [])), 1),
                    node.id,
                )
            )
    return layers


def _layout_graph(graph: _Graph) -> _Graph:
    """Place nodes on a fixed-height grid of layers."""
    _assign_layers(graph)
    pos = {node.id: node for node in graph.nodes}
    for node in graph.nodes:
        node.width = _text_width(node.lines) * _BOLD_FACTOR + 2 * _NODE_PAD_X
    # Route edges that skip layers through dummy nodes, so they reserve horizontal
    # space and neither they nor their labels run over the nodes in between.
    routes: dict[str, list[_Node]] = dict()
    for edge in graph.edges:
        routes[edge.id] = [
            _Node(
                id=f"{edge.id}-p{layer}",
                lines=[],
                paths=[],
                layer=layer,
                width=max(_text_width(edge.lines) + 12, 30.0),
                dummy=True,
            )
            for layer in range(pos[edge.src].layer + 1, pos[edge.dst].layer)
        ]
    all_nodes = graph.nodes + [node for route in routes.values() for node in route]
    pos.update((node.id, node) for route in routes.values() for node in route)
    # Chain adjacency: edge endpoints link through their routing dummies
    neighbors: dict[str, list[str]] = dict()
    for edge in graph.edges:
        chain = [edge.src] + [node.id for node in routes[edge.id]] + [edge.dst]
        for prev_id, next_id in zip(chain[:-1], chain[1:]):
            neighbors.setdefault(prev_id, []).append(next_id)
            neighbors.setdefault(next_id, []).append(prev_id)
    layers = _order_layers(all_nodes, neighbors)
    widths = [
        sum(node.width for node in layer) + _NODE_GAP * max(len(layer) - 1, 0)
        for layer in layers
    ]
    graph.width = max(widths, default=0.0) + 2 * _MARGIN

    def _pack() -> None:
        for li, layer in enumerate(layers):
            x = (graph.width - widths[li]) / 2
            for node in layer:
                node.x = x + node.width / 2
                node.y = _MARGIN + li * _ROW_HEIGHT
                node.y += _ROW_MID if node.dummy else _node_height(node) / 2
                x += node.width + _NODE_GAP

    _pack()
    real_neighbors: dict[str, list[str]] = dict()
    for edge in graph.edges:
        real_neighbors.setdefault(edge.src, []).append(edge.dst)
        real_neighbors.setdefault(edge.dst, []).append(edge.src)
    # Long edges route around the side of the content in the layers they traverse,
    # so their chains never compete with the nodes themselves for the center:
    # values are (order key, desired x)
    margin: dict[str, tuple[float, float]] = dict()

    def _pick_sides() -> None:
        # Recomputed each refinement pass: the endpoints move as steps settle, and
        # a side chosen from stale positions sends a chain on a needless detour.
        # The order key keeps same-side chains in one consistent relative order
        # across layers, so they run parallel instead of braiding. The desired x
        # hugs the widest *traversed* layer rather than the graph margin — the
        # graph is as wide as its widest row, which may be far from the action.
        margin.clear()
        bounds: dict[int, tuple[float, float]] = dict()
        for node in graph.nodes:
            lo, hi = bounds.get(node.layer, (graph.width, 0.0))
            bounds[node.layer] = (
                min(lo, node.x - node.width / 2),
                max(hi, node.x + node.width / 2),
            )
        for edge in graph.edges:
            if len(routes[edge.id]) < 2:
                continue
            mid = (pos[edge.src].x + pos[edge.dst].x) / 2
            spanned = [
                bounds[node.layer] for node in routes[edge.id] if node.layer in bounds
            ]
            if mid <= graph.width / 2:
                hull = min((lo for lo, _ in spanned), default=graph.width / 2)
                for node in routes[edge.id]:
                    want = hull - _NODE_GAP - node.width / 2
                    margin[node.id] = (mid - graph.width, max(want, _min_x(node)))
            else:
                hull = max((hi for _, hi in spanned), default=graph.width / 2)
                for node in routes[edge.id]:
                    want = hull + _NODE_GAP + node.width / 2
                    margin[node.id] = (
                        mid + 2 * graph.width,
                        min(want, _max_x(graph.width, node)),
                    )

    def _neighbor_mean(node: _Node) -> float:
        source = neighbors if node.dummy else real_neighbors
        linked = source.get(node.id)
        if not linked:
            return node.x
        return sum(pos[other].x for other in linked) / len(linked)

    def _order_key(node: _Node) -> float:
        if node.id in margin:
            return margin[node.id][0]
        return _neighbor_mean(node)

    def _desired(node: _Node) -> float:
        if node.id in margin:
            return margin[node.id][1]
        # A lone routing point yields to the steps (stays put, so it never pushes
        # them off-center) and settles into the leftover slack afterwards
        if node.dummy:
            return node.x
        return _neighbor_mean(node)

    for _ in range(2):
        # The ordinal barycenter above ignores geometry (a 30px dummy and a 190px
        # node count the same), so refine the order by actual positions
        _pick_sides()
        for layer in layers:
            layer.sort(key=_order_key)
        _pack()
        _straighten_routes(graph, layers, _desired)
    for layer in layers:
        for ii, node in enumerate(layer):
            if not node.dummy or node.id in margin:
                continue
            lo = _min_x(node)
            if ii > 0:
                lo = max(lo, layer[ii - 1].x + _min_sep(layer[ii - 1], node))
            hi = _max_x(graph.width, node)
            if ii < len(layer) - 1:
                hi = min(hi, layer[ii + 1].x - _min_sep(node, layer[ii + 1]))
            if lo <= hi:
                node.x = min(max(_neighbor_mean(node), lo), hi)
    for edge in graph.edges:
        src, dst = pos[edge.src], pos[edge.dst]
        edge.points = (
            [(src.x, src.y + _node_height(src) / 2)]
            + [(node.x, node.y) for node in routes[edge.id]]
            + [(dst.x, dst.y - _node_height(dst) / 2)]
        )
    graph.height = (
        max((node.y + _node_height(node) / 2 for node in graph.nodes), default=0.0)
        + _MARGIN
    )
    return graph


def _straighten_routes(
    graph: _Graph,
    layers: Sequence[Sequence[_Node]],
    desired_fn: Callable[[_Node], float],
) -> None:
    """Move every node toward its desired x while preserving order and gaps.

    Forward/backward min-separation sweeps let a node push its whole run aside;
    each layer fits inside graph.width by construction, so the backward pass
    always finds room and nothing overlaps.
    """
    for _ in range(8):
        for layer in layers:
            xs = [desired_fn(node) for node in layer]
            for ii, node in enumerate(layer):
                lo = _min_x(node)
                if ii > 0:
                    lo = max(lo, xs[ii - 1] + _min_sep(layer[ii - 1], node))
                xs[ii] = max(xs[ii], lo)
            for ii in reversed(range(len(layer))):
                node = layer[ii]
                hi = _max_x(graph.width, node)
                if ii < len(layer) - 1:
                    hi = min(hi, xs[ii + 1] - _min_sep(node, layer[ii + 1]))
                xs[ii] = min(xs[ii], hi)
            for node, x in zip(layer, xs):
                node.x = x


# -- SVG ---------------------------------------------------------------------

_CSS = """
.mbp-flow-wrap { overflow-x: auto; }
svg.mbp-flow { font-size: 12px; color: inherit; }
svg.mbp-flow text { fill: currentColor; }
svg.mbp-flow .mbp-flow-box {
  fill: var(--bs-tertiary-bg, #f3f3f3);
  stroke: currentColor;
  stroke-width: 1;
}
svg.mbp-flow .mbp-flow-node text { font-weight: 600; }
svg.mbp-flow .mbp-flow-line {
  fill: none;
  stroke: currentColor;
  stroke-width: 1.2;
  opacity: 0.65;
}
svg.mbp-flow .mbp-flow-line-casing {
  fill: none;
  stroke: var(--bs-body-bg, #ffffff);
  stroke-width: 5;
  opacity: 0.5;
}
svg.mbp-flow .mbp-flow-arrow { fill: currentColor; opacity: 0.65; }
svg.mbp-flow .mbp-flow-label-bg { fill: var(--bs-body-bg, #ffffff); stroke: none; }
svg.mbp-flow .mbp-flow-edge text { font-size: 10.5px; opacity: 0.85; }
svg.mbp-flow .mbp-flow-node, svg.mbp-flow .mbp-flow-edge { transition: opacity 0.1s; }
svg.mbp-flow.mbp-flow-hovering .mbp-flow-node,
svg.mbp-flow.mbp-flow-hovering .mbp-flow-edge { opacity: 0.15; }
svg.mbp-flow.mbp-flow-hovering .mbp-flow-hl { opacity: 1; }
"""

_JS = """
(function () {
  var root = document.getElementById("%(svg_id)s");
  if (!root) { return; }
  var clear = function () {
    root.classList.remove("mbp-flow-hovering");
    root.querySelectorAll(".mbp-flow-hl").forEach(function (el) {
      el.classList.remove("mbp-flow-hl");
    });
  };
  root.querySelectorAll(".mbp-flow-node").forEach(function (node) {
    node.addEventListener("mouseenter", function () {
      clear();
      var attrs = ["data-flow-ancestors", "data-flow-descendants", "data-flow-edges"];
      var ids = [node.id];
      attrs.forEach(function (attr) {
        ids = ids.concat((node.getAttribute(attr) || "").split(" "));
      });
      ids.forEach(function (id) {
        var el = id ? root.querySelector("#" + id) : null;
        if (el) { el.classList.add("mbp-flow-hl"); }
      });
      root.classList.add("mbp-flow-hovering");
    });
    node.addEventListener("mouseleave", clear);
    node.addEventListener("focus", clear);
  });
})();
"""


def _reachable(
    graph: _Graph, *, reverse: bool
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Get the node and edge ids reachable from every node."""
    adjacent: dict[str, list[tuple[str, str]]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        src, dst = (edge.dst, edge.src) if reverse else (edge.src, edge.dst)
        adjacent[src].append((dst, edge.id))
    nodes: dict[str, set[str]] = dict()
    edges: dict[str, set[str]] = dict()
    for node in graph.nodes:
        seen_nodes: set[str] = set()
        seen_edges: set[str] = set()
        stack = [node.id]
        while stack:
            for other, edge_id in adjacent[stack.pop()]:
                seen_edges.add(edge_id)
                if other not in seen_nodes:
                    seen_nodes.add(other)
                    stack.append(other)
        nodes[node.id] = seen_nodes
        edges[node.id] = seen_edges
    return nodes, edges


def _edge_path(
    points: Sequence[tuple[float, float]], *, frac: float
) -> tuple[str, tuple[float, float]]:
    """Get the SVG path through an edge's routing points and its label anchor.

    The anchor is the point on the curve ``frac`` of the way down the gap it spans.
    """
    d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        # Vertical control handles, so consecutive segments join smoothly
        ym = (y0 + y1) / 2
        d += f" C {x0:.1f} {ym:.1f}, {x1:.1f} {ym:.1f}, {x1:.1f} {y1:.1f}"
    if len(points) > 2:  # the routing points sit clear of the nodes, so label there
        return d, points[len(points) // 2]
    (x0, y0), (x1, y1) = points
    ym = (y0 + y1) / 2

    def at(t: float) -> tuple[float, float]:
        weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3)
        return (
            sum(w * x for w, x in zip(weights, (x0, x0, x1, x1))),
            sum(w * y for w, y in zip(weights, (y0, ym, ym, y1))),
        )

    target = y0 + frac * (y1 - y0)
    lo, hi = 0.0, 1.0
    for _ in range(24):  # the curve is monotonic in y, so just bisect for the height
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at(mid)[1] < target else (lo, mid)
    return d, at((lo + hi) / 2)


def _svg_title(paths: Sequence[str]) -> str:
    limit = 20
    if not paths:
        return ""
    lines = list(paths[:limit])
    if len(paths) > limit:
        lines.append(f"… and {len(paths) - limit} more")
    return f"<title>{escape(chr(10).join(lines))}</title>"


def _svg_text(lines: Sequence[str], *, x: float, y: float, klass: str) -> str:
    top = y - (len(lines) - 1) * _LINE_HEIGHT / 2 + _FONT_SIZE / 3
    spans = "".join(
        f'<tspan x="{x:.1f}" y="{top + ii * _LINE_HEIGHT:.1f}">{escape(line)}</tspan>'
        for ii, line in enumerate(lines)
    )
    return f'<text class="{klass}" text-anchor="middle">{spans}</text>'


def _graph_svg(graph: _Graph, *, extra_css: str = "", label: str = "Graph") -> str:
    """Render a laid-out graph as a single self-contained SVG element."""
    ancestors, ancestor_edges = _reachable(graph, reverse=True)
    descendants, descendant_edges = _reachable(graph, reverse=False)

    # Fanned edges get one label per distinct text (on the middlemost edge of each
    # cluster), staggered in height — every copy of an identical label is redundant
    # ink and they collide near the shared end, where the curves converge.
    src_fan = Counter(edge.src for edge in graph.edges)
    groups: dict[tuple[str, str], list[_Edge]] = dict()
    for edge in graph.edges:
        key = ("src", edge.src) if src_fan[edge.src] > 1 else ("dst", edge.dst)
        groups.setdefault(key, []).append(edge)
    fracs: dict[str, float] = dict()
    for key, group in groups.items():
        far = -1 if key[0] == "src" else 0  # the end that is not shared
        clusters: dict[tuple[str, ...], list[_Edge]] = dict()
        for edge in sorted(group, key=lambda e: e.points[far][0]):
            clusters.setdefault(tuple(edge.lines), []).append(edge)
        reps = []
        for cluster in clusters.values():
            # Routed edges label at a distant routing point, so a direct edge keeps
            # the cluster's one label visually inside the fan
            direct = [edge for edge in cluster if len(edge.points) == 2] or cluster
            reps.append(direct[len(direct) // 2])
        reps.sort(key=lambda e: e.points[far][0])
        n_levels = min(len(reps), 6)
        for ii, edge in enumerate(reps):
            level = ii % n_levels
            fracs[edge.id] = (
                0.5 if n_levels == 1 else 0.2 + 0.6 * level / (n_levels - 1)
            )

    edge_html: list[str] = list()
    # Long chains draw first so short edges sit on top; each line carries a
    # translucent body-background casing that breaks the lines beneath it at
    # crossings, making the z-order legible
    for edge in sorted(graph.edges, key=lambda edge: -len(edge.points)):
        d, (mx, my) = _edge_path(edge.points, frac=fracs.get(edge.id, 0.5))
        width = _text_width(edge.lines) + 10
        height = len(edge.lines) * _LINE_HEIGHT
        if edge.id in fracs:
            label_html = (
                f'<rect class="mbp-flow-label-bg" rx="3" x="{mx - width / 2:.1f}" '
                f'y="{my - height / 2:.1f}" width="{width:.1f}" '
                f'height="{height:.1f}" />'
                f"{_svg_text(edge.lines, x=mx, y=my, klass='mbp-flow-edge-label')}"
            )
        else:
            label_html = ""
        edge_html.append(
            f'<g id="{edge.id}" class="mbp-flow-edge">'
            f'<path class="mbp-flow-line-casing" d="{d}" />'
            f'<path class="mbp-flow-line" d="{d}" '
            f'marker-end="url(#{_ID_PREFIX}-arrow)" />'
            f"{label_html}"
            f"{_svg_title(edge.paths)}</g>"
        )

    node_html: list[str] = list()
    for node in graph.nodes:
        height = _node_height(node)
        klass = f"mbp-flow-node {node.klass}".strip()
        related = {
            "ancestors": ancestors[node.id],
            "descendants": descendants[node.id],
            "edges": ancestor_edges[node.id] | descendant_edges[node.id],
        }
        attrs = " ".join(
            f"data-flow-{key}={quoteattr(' '.join(sorted(value)))}"
            for key, value in related.items()
        )
        node_html.append(
            f'<g id="{node.id}" class="{klass}" {attrs}>'
            f'<rect class="mbp-flow-box" rx="4" x="{node.x - node.width / 2:.1f}" '
            f'y="{node.y - height / 2:.1f}" width="{node.width:.1f}" '
            f'height="{height:.1f}" />'
            f"{_svg_text(node.lines, x=node.x, y=node.y, klass='mbp-flow-node-label')}"
            f"{_svg_title(node.paths)}</g>"
        )

    return (
        f'<svg id="{_SVG_ID}" class="mbp-flow" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {graph.width:.0f} {graph.height:.0f}" '
        f'width="{graph.width:.0f}" height="{graph.height:.0f}" '
        f'role="img" aria-label={quoteattr(label)}>'
        f"<style>{_CSS}{extra_css}</style>"
        f'<defs><marker id="{_ID_PREFIX}-arrow" viewBox="0 0 8 8" refX="7" refY="4" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path class="mbp-flow-arrow" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs>'
        f'<g class="mbp-flow-edges">{"".join(edge_html)}</g>'
        f'<g class="mbp-flow-nodes">{"".join(node_html)}</g>'
        f"</svg>"
    )


def _graph_html(graph: _Graph, *, extra_css: str = "", label: str = "Graph") -> str:
    """Render a laid-out graph plus its hover-highlighting behavior."""
    svg = _graph_svg(graph, extra_css=extra_css, label=label)
    return (
        f'<div class="mbp-flow-wrap">{svg}</div>'
        f"<script>{_JS % dict(svg_id=_SVG_ID)}</script>"
    )
