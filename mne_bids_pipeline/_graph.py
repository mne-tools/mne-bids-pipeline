"""Layered-DAG layout and self-contained SVG rendering.

Domain-agnostic: everything here operates on ``_Node``/``_Edge``/``_Graph`` and
could draw any small DAG. The pipeline-specific graph construction lives in
``_flow.py``.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

_ID_PREFIX = "mbp-flow"


@dataclass
class _Node:
    id: str
    lines: list[str]
    paths: list[str]
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
_NODE_GAP = 26.0
_ROW_HEIGHT = 150.0
_MARGIN = 10.0
_ROW_MID = 21.0  # where a routing point sits within its row


def _text_width(lines: Sequence[str]) -> float:
    return max((len(line) for line in lines), default=0) * _CHAR_WIDTH


def _node_height(node: _Node) -> float:
    return 0.0 if node.dummy else len(node.lines) * _LINE_HEIGHT + 12.0


def _assign_layers(graph: _Graph) -> None:
    layers = {node.id: 0 for node in graph.nodes}
    # Longest-path layering by relaxation; the pass limit also bounds the (never
    # expected, but possible if a step both reads and writes a shared file) cycle.
    for _ in range(len(graph.nodes)):
        changed = False
        for edge in graph.edges:
            if layers[edge.dst] < layers[edge.src] + 1:
                layers[edge.dst] = layers[edge.src] + 1
                changed = True
        if not changed:
            break
    for node in graph.nodes:
        node.layer = layers[node.id]


def _order_layers(
    nodes: Sequence[_Node], chains: Sequence[Sequence[str]]
) -> list[list[_Node]]:
    n_layers = max((node.layer for node in nodes), default=0) + 1
    layers: list[list[_Node]] = [[] for _ in range(n_layers)]
    for node in nodes:
        layers[node.layer].append(node)
    neighbors: dict[str, list[str]] = {node.id: [] for node in nodes}
    for chain in chains:
        for src, dst in zip(chain[:-1], chain[1:]):
            neighbors[src].append(dst)
            neighbors[dst].append(src)
    for _ in range(4):
        index = {
            node.id: ii for layer in layers for ii, node in enumerate(layer)
        }  # barycenter positions
        for layer in layers:
            layer.sort(
                key=lambda node: (
                    sum(index[other] for other in neighbors[node.id])
                    / max(len(neighbors[node.id]), 1),
                    node.id,
                )
            )
    return layers


def _layout_graph(graph: _Graph) -> _Graph:
    """Place nodes on a fixed-height grid of layers."""
    _assign_layers(graph)
    nodes = {node.id: node for node in graph.nodes}
    for node in graph.nodes:
        node.width = _text_width(node.lines) + 2 * _NODE_PAD_X
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
            for layer in range(nodes[edge.src].layer + 1, nodes[edge.dst].layer)
        ]
    all_nodes = graph.nodes + [node for route in routes.values() for node in route]
    chains = [
        [edge.src] + [node.id for node in routes[edge.id]] + [edge.dst]
        for edge in graph.edges
    ]
    layers = _order_layers(all_nodes, chains)
    widths = [
        sum(node.width for node in layer) + _NODE_GAP * max(len(layer) - 1, 0)
        for layer in layers
    ]
    graph.width = max(widths, default=0.0) + 2 * _MARGIN
    for li, layer in enumerate(layers):
        x = (graph.width - widths[li]) / 2
        for node in layer:
            node.x = x + node.width / 2
            node.y = _MARGIN + li * _ROW_HEIGHT
            node.y += _ROW_MID if node.dummy else _node_height(node) / 2
            x += node.width + _NODE_GAP
    for edge in graph.edges:
        src, dst = nodes[edge.src], nodes[edge.dst]
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
svg.mbp-flow .mbp-flow-source .mbp-flow-box { stroke-dasharray: 4 3; }
svg.mbp-flow .mbp-flow-line {
  fill: none;
  stroke: currentColor;
  stroke-width: 1.2;
  opacity: 0.65;
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
    points: Sequence[tuple[float, float]], *, frac: float = 0.5
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


def _svg_title(paths: Sequence[str], *, limit: int = 20) -> str:
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


def _graph_svg(
    graph: _Graph, *, source_id: str | None = None, svg_id: str = f"{_ID_PREFIX}-svg"
) -> str:
    """Render a laid-out graph as a single self-contained SVG element."""
    ancestors, ancestor_edges = _reachable(graph, reverse=True)
    descendants, descendant_edges = _reachable(graph, reverse=False)

    # Stagger the labels of edges that fan out of (or into) a common node, otherwise
    # they all land at the same height and overlap.
    fan: dict[str, int] = dict()
    for edge in graph.edges:
        for key in (f"src-{edge.src}", f"dst-{edge.dst}"):
            fan[key] = fan.get(key, 0) + 1
    groups: dict[str, list[_Edge]] = dict()
    for edge in graph.edges:
        key = f"src-{edge.src}" if fan[f"src-{edge.src}"] > 1 else f"dst-{edge.dst}"
        groups.setdefault(key, []).append(edge)
    fracs: dict[str, float] = dict()
    for key, group in groups.items():
        far = -1 if key.startswith("src-") else 0  # the end that is not shared
        # Curves converge at the shared end, so edges that wrap onto a reused level
        # collide there; only wrap for fans too big to give every edge its own level
        n_levels = min(len(group), 6)
        for ii, edge in enumerate(sorted(group, key=lambda e: e.points[far][0])):
            level = ii % n_levels
            fracs[edge.id] = (
                0.5 if n_levels == 1 else 0.15 + 0.7 * level / (n_levels - 1)
            )

    edge_html: list[str] = list()
    for edge in graph.edges:
        d, (mx, my) = _edge_path(edge.points, frac=fracs[edge.id])
        width = _text_width(edge.lines) + 10
        height = len(edge.lines) * _LINE_HEIGHT
        edge_html.append(
            f'<g id="{edge.id}" class="mbp-flow-edge">'
            f'<path class="mbp-flow-line" d="{d}" '
            f'marker-end="url(#{_ID_PREFIX}-arrow)" />'
            f'<rect class="mbp-flow-label-bg" rx="3" x="{mx - width / 2:.1f}" '
            f'y="{my - height / 2:.1f}" width="{width:.1f}" height="{height:.1f}" />'
            f"{_svg_text(edge.lines, x=mx, y=my, klass='mbp-flow-edge-label')}"
            f"{_svg_title(edge.paths)}</g>"
        )

    node_html: list[str] = list()
    for node in graph.nodes:
        height = _node_height(node)
        klass = "mbp-flow-node"
        if node.id == source_id:
            klass += " mbp-flow-source"
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
        f'<svg id="{svg_id}" class="mbp-flow" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {graph.width:.0f} {graph.height:.0f}" '
        f'width="{graph.width:.0f}" height="{graph.height:.0f}" '
        f'role="img" aria-label="Pipeline flow diagram">'
        f"<style>{_CSS}</style>"
        f'<defs><marker id="{_ID_PREFIX}-arrow" viewBox="0 0 8 8" refX="7" refY="4" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path class="mbp-flow-arrow" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs>'
        f'<g class="mbp-flow-edges">{"".join(edge_html)}</g>'
        f'<g class="mbp-flow-nodes">{"".join(node_html)}</g>'
        f"</svg>"
    )


def _graph_html(
    graph: _Graph,
    *,
    source_id: str | None = None,
    svg_id: str = f"{_ID_PREFIX}-svg",
) -> str:
    """Render a laid-out graph plus its hover-highlighting behavior."""
    svg = _graph_svg(graph, source_id=source_id, svg_id=svg_id)
    return (
        f'<div class="mbp-flow-wrap">{svg}</div>'
        f"<script>{_JS % dict(svg_id=svg_id)}</script>"
    )
