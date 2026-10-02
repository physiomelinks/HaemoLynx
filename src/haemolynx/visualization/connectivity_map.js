/* The 2D connectivity map page: draws a connectivity CSV in the browser.
 *
 * write_connectivity_map (connectivity_map.py) puts this script in the page
 * with the CSV's text and a config (branch-order sort groups, colours). All
 * the layout happens here, so "Load CSV..." can draw any other connectivity
 * CSV without Python. Node's require() gets the same functions for the tests
 * (tests/test_connectivity_map.py).
 *
 * Two views, picked from the page's "View" dropdown:
 *   inlet         columns count vessels from an inlet; inlets left, outlets
 *                 right, separate pieces of the network stacked
 *   branch_order  one column per branch order, inlets to outlets; each vessel
 *                 a short line in its order's column, thin grey lines joining
 *                 a vessel to the vessels it feeds
 * Inlet and outlet vessels are drawn bolder. "Branch order stats" opens a
 * table per branch order (orderStats). Hovering is done here rather
 * than by plotly, which only answers near a data point and so misses most of
 * a line once zoomed in: the vessel or node nearest the mouse, in pixels,
 * gets the box at any zoom.
 */
(function (root) {
  "use strict";

  /** Vertical gap, in rows, between separate pieces of the network. */
  const PIECE_GAP = 2.0;
  /** Line widths: inlet and outlet vessels are drawn bolder than the rest. */
  const LINE_WIDTH = 2;
  const BOLD_LINE_WIDTH = 6;
  /** Half the width of a vessel's line in the branch-order view, in columns. */
  const HALF_SPAN = 0.35;
  /** The column the branch-order view gives a vessel with no branch order. */
  const NO_ORDER = "(none)";
  /** How near the mouse, in pixels, a vessel's line must be to get the box. */
  const HOVER_PIXELS = 8;
  const VIEWS = ["inlet", "branch_order"];
  const REQUIRED_COLUMNS = ["Branch ID", "From node ID", "To node ID"];

  const NODE_KINDS = [
    // label, symbol, colour, size
    ["Node", "circle", "#7f7f7f", 4],
    ["Inlet", "triangle-right", "#2ca02c", 11],
    ["Outlet", "square", "#d62728", 10],
    ["Dead end", "x", "#ff7f0e", 8],
  ];

  // ------------------------------------------------------------------ CSV

  /** The rows of a CSV's text as {column: value} objects (quotes as Excel writes them). */
  function parseCsv(text) {
    const table = [];
    let row = [];
    let field = "";
    let quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"') {
          if (text[i + 1] === '"') {
            field += '"';
            i++;
          } else {
            quoted = false;
          }
        } else {
          field += c;
        }
      } else if (c === '"') {
        quoted = true;
      } else if (c === ",") {
        row.push(field);
        field = "";
      } else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(field);
        table.push(row);
        row = [];
        field = "";
      } else {
        field += c;
      }
    }
    if (field !== "" || row.length) {
      row.push(field);
      table.push(row);
    }
    const lines = table.filter((r) => r.some((v) => v !== ""));
    if (!lines.length) return [];
    const header = lines[0].map((h) => h.replace(/^﻿/, "").trim());
    const missing = REQUIRED_COLUMNS.filter((c) => !header.includes(c));
    if (missing.length) {
      throw new Error(
        `not a connectivity CSV (no ${missing.map((c) => `"${c}"`).join(", ")} column)`
      );
    }
    return lines.slice(1).map((r) => {
      const out = {};
      header.forEach((h, k) => {
        out[h] = r[k] === undefined ? "" : r[k];
      });
      return out;
    });
  }

  /** A row's [from, to] node IDs, filling a blank inlet/outlet tip from the edge key. */
  function ends(row) {
    let start = row["From node ID"] || "";
    let end = row["To node ID"] || "";
    const u = row["Edge u"] || "";
    const v = row["Edge v"] || "";
    if (!start && !end) return [u, v];
    if (!start) start = v === end ? u : v;
    if (!end) end = u === start ? v : u;
    return [start, end];
  }

  function notes(row) {
    return new Set(
      (row["Notes"] || "").split(";").map((n) => n.trim()).filter((n) => n)
    );
  }

  function isBold(row) {
    const n = notes(row);
    return n.has("Inlet") || n.has("Outlet");
  }

  function orderOf(row) {
    return row["Branch order"] || NO_ORDER;
  }

  function cmp(a, b) {
    return a < b ? -1 : a > b ? 1 : 0;
  }

  function cmpKeys(a, b) {
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      const c = cmp(a[i], b[i]);
      if (c) return c;
    }
    return 0;
  }

  /** Where *order*'s column sits, from the inlets (left) to the outlets (right).
   *
   * Large_Art, Art and B count up from the inlet side, so they run 1, 2, 3...;
   * Ven and Large_Ven count up from the outlets, so they run backwards.
   * config.orderGroups is visualization._helpers' sort-group table.
   */
  function orderSortKey(order, config) {
    const match = /(\d+)$/.exec(order);
    if (!match) return [99, 0, order];
    const groups = config.orderGroups;
    const prefix = order.slice(0, match.index).toLowerCase();
    const group = prefix in groups ? groups[prefix] : Object.keys(groups).length;
    const n = parseInt(match[1], 10);
    const venous = config.venousGroups.includes(group);
    return [group, venous ? -n : n, order];
  }

  function sortOrders(orders, config) {
    return [...orders].sort((a, b) => cmpKeys(orderSortKey(a, config), orderSortKey(b, config)));
  }

  function branchIdKey(row) {
    const id = row["Branch ID"] || "";
    return /^-?\d+$/.test(id) ? [parseInt(id, 10), ""] : [1 << 30, id];
  }

  function pushTo(map, key, value) {
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(value);
  }

  function uniquePush(map, key, value) {
    if (!map.has(key)) map.set(key, new Set());
    map.get(key).add(value);
  }

  // ------------------------------------------------------------- layouts

  /** View "inlet": each node in the column of how many vessels it is from an inlet. */
  function inletLayout(rows) {
    const succ = new Map();
    const pred = new Map();
    const und = new Map();
    const nodes = new Set();
    const inlets = new Set();
    const outlets = new Set();
    const deadEnds = new Set();
    const vessels = [];
    const selfLoops = [];
    for (const row of rows) {
      const [start, end] = ends(row);
      const n = notes(row);
      nodes.add(start);
      if (start === end) {
        selfLoops.push(row);
        continue;
      }
      nodes.add(end);
      uniquePush(succ, start, end);
      uniquePush(pred, end, start);
      uniquePush(und, start, end);
      uniquePush(und, end, start);
      vessels.push({ row, start, end });
      if (n.has("Inlet")) inlets.add(start);
      if (n.has("Outlet")) outlets.add(end);
      if (n.has("Dead end")) deadEnds.add(end);
    }
    const around = (map, node) => [...(map.get(node) || [])];

    const seen = new Set();
    const pieces = [];
    for (const node of nodes) {
      if (seen.has(node)) continue;
      const piece = [node];
      seen.add(node);
      for (let i = 0; i < piece.length; i++) {
        for (const m of around(und, piece[i])) {
          if (!seen.has(m)) {
            seen.add(m);
            piece.push(m);
          }
        }
      }
      pieces.push(piece);
    }
    const least = (piece) => piece.reduce((a, b) => (b < a ? b : a));
    pieces.sort((a, b) => b.length - a.length || cmp(least(a), least(b)));

    const positions = new Map();
    let top = 0.0;
    for (const piece of pieces) {
      const sorted = [...piece].sort(cmp);
      let sources = sorted.filter((n) => inlets.has(n));
      if (!sources.length) sources = sorted.filter((n) => !(pred.get(n) || new Set()).size);
      if (!sources.length) sources = [sorted[0]];
      const level = new Map(sources.map((n) => [n, 0]));
      const queue = [...sources];
      for (let i = 0; i < queue.length; i++) {
        for (const next of around(succ, queue[i])) {
          if (!level.has(next)) {
            level.set(next, level.get(queue[i]) + 1);
            queue.push(next);
          }
        }
      }
      // Whatever the arrows do not reach (a loop no inlet feeds) is placed
      // from its neighbours, ignoring direction.
      const order = [sources[0]];
      const visited = new Set(order);
      for (let i = 0; i < order.length; i++) {
        for (const m of around(und, order[i])) {
          if (!visited.has(m)) {
            visited.add(m);
            order.push(m);
          }
        }
      }
      for (const node of order) {
        if (level.has(node)) continue;
        const placed = around(und, node).filter((m) => level.has(m)).map((m) => level.get(m) + 1);
        level.set(node, placed.length ? Math.min(...placed) : 0);
      }
      for (const node of piece) if (!level.has(node)) level.set(node, 0);

      const columns = new Map();
      for (const [node, x] of level) pushTo(columns, x, node);
      const yOf = new Map();
      let height = 0;
      for (const x of [...columns.keys()].sort((a, b) => a - b)) {
        const barycentre = (node) => {
          let above = around(pred, node).filter((p) => yOf.has(p));
          if (!above.length) above = around(und, node).filter((p) => yOf.has(p));
          return above.length ? above.reduce((s, p) => s + yOf.get(p), 0) / above.length : 0.0;
        };
        const keyed = columns.get(x).map((node) => [barycentre(node), node]);
        keyed.sort(cmpKeys);
        keyed.forEach(([, node], i) => yOf.set(node, i - (keyed.length - 1) / 2));
        height = Math.max(height, keyed.length);
      }
      for (const [node, y] of yOf) positions.set(node, [level.get(node), top - y]);
      top -= height + PIECE_GAP;
    }
    return { positions, inlets, outlets, deadEnds, vessels, selfLoops };
  }

  /** View "branch_order": each vessel in its branch order's column.
   *
   * Inside a column, a vessel sits level with the vessels that feed it, so a
   * vessel's daughters stay near it; vessels pushed apart keep one row
   * between them.
   */
  function branchLayout(rows, config) {
    const items = rows.map((row) => [row, ...ends(row)]);
    const orders = items.map(([row]) => orderOf(row));
    const columns = sortOrders(new Set(orders), config);
    const columnOf = new Map(columns.map((o, i) => [o, i]));

    const endingAt = new Map();
    const touching = new Map();
    items.forEach(([, start, end], i) => {
      pushTo(endingAt, end, i);
      pushTo(touching, start, i);
      pushTo(touching, end, i);
    });

    const yOf = new Map();
    for (const order of columns) {
      const members = [];
      orders.forEach((o, i) => {
        if (o === order) members.push(i);
      });
      const wants = new Map();
      for (const i of members) {
        const [, start, end] = items[i];
        let feeders = (endingAt.get(start) || []).filter((j) => yOf.has(j));
        if (!feeders.length) {
          feeders = [...(touching.get(start) || []), ...(touching.get(end) || [])].filter((j) =>
            yOf.has(j)
          );
        }
        wants.set(
          i,
          feeders.length ? feeders.reduce((s, j) => s + yOf.get(j), 0) / feeders.length : null
        );
      }
      const byId = (a, b) => cmpKeys(branchIdKey(items[a][0]), branchIdKey(items[b][0]));
      const placed = members
        .filter((i) => wants.get(i) !== null)
        .sort((a, b) => wants.get(a) - wants.get(b) || byId(a, b));
      const unplaced = members.filter((i) => wants.get(i) === null).sort(byId);
      let ys = [];
      for (const i of placed) {
        ys.push(ys.length ? Math.max(wants.get(i), ys[ys.length - 1] + 1.0) : wants.get(i));
      }
      // Pushing vessels apart only moves them down; centre the column back on
      // where its vessels wanted to be.
      if (ys.length) {
        const shift =
          (ys.reduce((s, y) => s + y, 0) - placed.reduce((s, i) => s + wants.get(i), 0)) /
          ys.length;
        ys = ys.map((y) => y - shift);
      }
      placed.forEach((i, k) => yOf.set(i, ys[k]));
      const below = ys.length ? ys[ys.length - 1] + 1.0 : -(unplaced.length - 1) / 2.0;
      unplaced.forEach((i, k) => yOf.set(i, below + k));
    }

    const segments = items.map(([row, start, end], i) => {
      const x = columnOf.get(orders[i]);
      return { row, start, end, x0: x - HALF_SPAN, x1: x + HALF_SPAN, y: -yOf.get(i) };
    });
    return { columns, segments };
  }

  // ---------------------------------------------------------------- stats

  function idCount(cell) {
    return (cell || "").split(",").filter((id) => id.trim()).length;
  }

  function mean(values) {
    return values.length ? values.reduce((s, v) => s + v, 0) / values.length : null;
  }

  function numbers(rows, key) {
    return rows.map((row) => parseFloat(row[key])).filter((v) => Number.isFinite(v));
  }

  /** One summary per branch order, inlets to outlets, then "All".
   *
   * upstream / downstream: the mean number of vessels feeding a vessel and
   * fed by it (its "Upstream/Downstream branch IDs"; 0 at an inlet or outlet
   * vessel's open end). connections: the mean of those two, as asked for.
   * Lengths and diameters left blank in the CSV are left out of their means.
   */
  function orderStats(rows, config) {
    const byOrder = new Map();
    for (const row of rows) pushTo(byOrder, orderOf(row), row);
    const summary = (order, group) => {
      const up = group.map((row) => idCount(row["Upstream branch IDs"]));
      const down = group.map((row) => idCount(row["Downstream branch IDs"]));
      const upstream = mean(up);
      const downstream = mean(down);
      return {
        order,
        vessels: group.length,
        length: mean(numbers(group, "Length (um)")),
        diameter: mean(numbers(group, "Diameter (um)")),
        upstream,
        downstream,
        connections: upstream === null ? null : (upstream + downstream) / 2,
        inlets: group.filter((row) => notes(row).has("Inlet")).length,
        outlets: group.filter((row) => notes(row).has("Outlet")).length,
      };
    };
    const out = sortOrders(byOrder.keys(), config).map((o) => summary(o, byOrder.get(o)));
    if (rows.length) out.push(summary("All", rows));
    return out;
  }

  const STATS_COLUMNS = [
    // key, heading, decimals
    ["order", "Branch order", null],
    ["vessels", "Vessels", 0],
    ["length", "Mean length (µm)", 2],
    ["diameter", "Mean diameter (µm)", 2],
    ["upstream", "Mean vessels in", 2],
    ["downstream", "Mean vessels out", 2],
    ["connections", "Mean connections (in + out) / 2", 2],
    ["inlets", "Inlet vessels", 0],
    ["outlets", "Outlet vessels", 0],
  ];

  function statsCell(value, decimals) {
    if (value === null || value === undefined) return "";
    return decimals === null ? String(value) : Number(value).toFixed(decimals);
  }

  function statsCsv(stats) {
    const quote = (v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
    const lines = [STATS_COLUMNS.map(([, heading]) => quote(heading)).join(",")];
    for (const s of stats) {
      // The file keeps 4 decimals where the table shows 2.
      lines.push(STATS_COLUMNS.map(([key, , d]) => quote(statsCell(s[key], d ? 4 : d))).join(","));
    }
    return lines.join("\n") + "\n";
  }

  // ---------------------------------------------------------------- views

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function hoverText(row, start, end) {
    const lines = [
      `<b>Branch ${escapeHtml(row["Branch ID"] || "")}</b>`,
      `node ${escapeHtml(start)} → node ${escapeHtml(end)}`,
    ];
    for (const [label, key, unit] of [
      ["branch order", "Branch order", ""],
      ["length", "Length (um)", " µm"],
      ["diameter", "Diameter (um)", " µm"],
      ["upstream", "Upstream branch IDs", ""],
      ["downstream", "Downstream branch IDs", ""],
      ["notes", "Notes", ""],
    ]) {
      const value = row[key] || "";
      if (value) lines.push(`${label}: ${escapeHtml(value)}${unit}`);
    }
    return lines.join("<br>");
  }

  /** Line traces (one legend entry per branch order) and the hover items for them.
   *
   * A segment is {row, start, end, x0, y0, x1, y1}. Inlet and outlet vessels
   * go in a bolder trace of the same legend group.
   */
  function vesselTraces(segments, colourOf, config, traces, items) {
    const byOrder = new Map();
    for (const seg of segments) pushTo(byOrder, orderOf(seg.row), seg);
    for (const order of sortOrders(byOrder.keys(), config)) {
      const group = byOrder.get(order);
      let legendShown = false;
      for (const bold of [false, true]) {
        const chosen = group.filter((seg) => isBold(seg.row) === bold);
        if (!chosen.length) continue;
        const x = [];
        const y = [];
        const index = traces.length;
        const width = bold ? BOLD_LINE_WIDTH : LINE_WIDTH;
        for (const seg of chosen) {
          x.push(seg.x0, seg.x1, null);
          y.push(seg.y0, seg.y1, null);
          items.push({
            kind: "vessel", trace: index, width, x0: seg.x0, y0: seg.y0, x1: seg.x1, y1: seg.y1,
            text: hoverText(seg.row, seg.start, seg.end),
          });
        }
        traces.push({
          type: "scatter", mode: "lines", x, y, hoverinfo: "skip",
          line: { color: colourOf.get(order), width },
          name: `${order} (${group.length})`, legendgroup: order, showlegend: !legendShown,
        });
        legendShown = true;
      }
    }
  }

  /** Marker traces for junctions, inlets, outlets and dead ends at *points*. */
  function nodeTraces(points, layout, traces, items) {
    const marked = new Set([...layout.inlets, ...layout.outlets, ...layout.deadEnds]);
    const kinds = {
      Node: [...points.keys()].filter((n) => !marked.has(n)),
      Inlet: [...layout.inlets],
      Outlet: [...layout.outlets],
      "Dead end": [...layout.deadEnds].filter((n) => !layout.outlets.has(n)),
    };
    for (const [label, symbol, colour, size] of NODE_KINDS) {
      const nodes = kinds[label].filter((n) => points.has(n)).sort(cmp);
      const x = [];
      const y = [];
      const index = traces.length;
      for (const node of nodes) {
        for (const [px, py] of points.get(node)) {
          x.push(px);
          y.push(py);
          items.push({
            kind: "node", trace: index, size, x0: px, y0: py,
            text: `${label}: node ${escapeHtml(node)}`,
          });
        }
      }
      traces.push({
        type: "scatter", mode: "markers", x, y, hoverinfo: "skip",
        marker: { symbol, color: colour, size }, name: `${label}s (${nodes.length})`,
      });
    }
  }

  /** Everything one view draws: {traces, items (for hovering), xaxis, counts}. */
  function buildView(rows, view, config) {
    if (!VIEWS.includes(view)) throw new Error(`view must be one of ${VIEWS}, not ${view}`);
    const layout = inletLayout(rows);
    const orders = sortOrders(new Set(rows.map(orderOf)), config);
    const colourOf = new Map(orders.map((o, i) => [o, config.palette[i % config.palette.length]]));
    const traces = [];
    const items = [];
    let xaxis;

    if (view === "inlet") {
      const pos = layout.positions;
      vesselTraces(
        layout.vessels.map(({ row, start, end }) => ({
          row, start, end,
          x0: pos.get(start)[0], y0: pos.get(start)[1], x1: pos.get(end)[0], y1: pos.get(end)[1],
        })),
        colourOf, config, traces, items
      );
      nodeTraces(new Map([...pos].map(([n, xy]) => [n, [xy]])), layout, traces, items);
      const loops = layout.selfLoops.filter((row) => pos.has(ends(row)[0]));
      if (loops.length) {
        const index = traces.length;
        const xy = loops.map((row) => pos.get(ends(row)[0]));
        loops.forEach((row, k) => {
          items.push({
            kind: "node", trace: index, size: 14, x0: xy[k][0], y0: xy[k][1],
            text: hoverText(row, ...ends(row)),
          });
        });
        traces.push({
          type: "scatter", mode: "markers", hoverinfo: "skip",
          x: xy.map((p) => p[0]), y: xy.map((p) => p[1]),
          marker: { symbol: "circle-open", color: "#9467bd", size: 14 },
          name: `Self-loops (${loops.length})`,
        });
      }
      xaxis = { title: { text: "vessels from an inlet" }, tickmode: "auto" };
    } else {
      const byOrder = branchLayout(rows, config);
      const starts = new Map();
      const endsAt = new Map();
      for (const seg of byOrder.segments) {
        pushTo(starts, seg.start, [seg.x0, seg.y]);
        pushTo(endsAt, seg.end, [seg.x1, seg.y]);
      }
      const lx = [];
      const ly = [];
      for (const [node, arrivals] of endsAt) {
        for (const [x1, y1] of arrivals) {
          for (const [x0, y0] of starts.get(node) || []) {
            if (x1 !== x0 || y1 !== y0) {
              lx.push(x1, x0, null);
              ly.push(y1, y0, null);
            }
          }
        }
      }
      traces.push({
        type: "scatter", mode: "lines", x: lx, y: ly, hoverinfo: "skip",
        line: { color: "#c7c7c7", width: 1 }, name: "Joins between vessels",
      });
      vesselTraces(
        byOrder.segments.map((s) => ({ ...s, y0: s.y, y1: s.y })), colourOf, config, traces, items
      );
      // A node is drawn where each vessel leaving it starts, or, for an outlet
      // or dead end, where the vessel reaching it ends.
      const points = new Map();
      for (const node of new Set([...starts.keys(), ...endsAt.keys()])) {
        const terminal = layout.outlets.has(node) || layout.deadEnds.has(node);
        points.set(node, (terminal && endsAt.get(node)) || starts.get(node) || endsAt.get(node));
      }
      nodeTraces(points, layout, traces, items);
      xaxis = {
        title: { text: "branch order (inlets → outlets)" }, tickmode: "array",
        tickvals: byOrder.columns.map((_o, i) => i), ticktext: byOrder.columns,
      };
    }
    return {
      traces, items, xaxis,
      counts: { vessels: rows.length, inlets: layout.inlets.size, outlets: layout.outlets.size },
    };
  }

  // -------------------------------------------------------------- hovering

  function distanceToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const span = dx * dx + dy * dy;
    const t = span ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / span)) : 0;
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  /** The hover item nearest pixel (px, py), or null if none is near enough.
   *
   * toPixel maps data (x, y) to pixels; shown(item) says whether its trace is
   * on (not hidden from the legend). A node beats the vessels it sits on.
   */
  function nearestItem(items, px, py, toPixel, shown) {
    let best = null;
    let bestScore = Infinity;
    for (const item of items) {
      if (!shown(item)) continue;
      const [ax, ay] = toPixel(item.x0, item.y0);
      let distance;
      let reach;
      if (item.kind === "node") {
        distance = Math.hypot(px - ax, py - ay);
        reach = item.size / 2 + 3;
      } else {
        const [bx, by] = toPixel(item.x1, item.y1);
        distance = distanceToSegment(px, py, ax, ay, bx, by);
        reach = Math.max(HOVER_PIXELS, item.width / 2 + 4);
      }
      if (distance > reach) continue;
      const score = item.kind === "node" ? distance - reach : distance;
      if (score < bestScore) {
        bestScore = score;
        best = item;
      }
    }
    return best;
  }

  // ------------------------------------------------------------------ page

  const STYLE = `
    html, body { margin: 0; height: 100%; font-family: "Open Sans", Arial, sans-serif;
      background: #fff; color: #222; }
    #hl-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px;
      padding: 8px 16px; border-bottom: 1px solid #ddd; font-size: 14px; }
    #hl-title { font-weight: 600; font-size: 16px; }
    #hl-counts { color: #555; }
    #hl-status { color: #555; }
    #hl-status.error { color: #c62828; }
    #hl-bar select, #hl-bar button { font: inherit; padding: 3px 8px; }
    #hl-plot { position: absolute; left: 0; right: 0; bottom: 0; }
    #hl-tip { position: fixed; display: none; pointer-events: none; z-index: 10;
      background: rgba(255, 255, 255, 0.97); border: 1px solid #888; border-radius: 4px;
      padding: 6px 8px; font-size: 12px; line-height: 1.45; max-width: 360px;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
    #hl-stats { position: absolute; right: 16px; z-index: 5; display: none;
      max-height: calc(100% - 120px); overflow: auto; background: #fff;
      border: 1px solid #bbb; border-radius: 4px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
      font-size: 12px; }
    #hl-stats.open { display: block; }
    #hl-stats-head { display: flex; align-items: center; gap: 8px; padding: 6px 8px;
      border-bottom: 1px solid #ddd; position: sticky; top: 0; background: #fff; }
    #hl-stats-head b { flex: 1; }
    #hl-stats table { border-collapse: collapse; }
    #hl-stats th, #hl-stats td { padding: 3px 8px; text-align: right; white-space: nowrap; }
    #hl-stats th { position: sticky; top: 33px; background: #f4f4f4; font-weight: 600;
      white-space: normal; max-width: 90px; vertical-align: bottom; }
    #hl-stats td:first-child, #hl-stats th:first-child { text-align: left; }
    #hl-stats tr:nth-child(even) td { background: #fafafa; }
    #hl-stats tr.hl-all td { border-top: 2px solid #999; font-weight: 600; background: #fff; }
    #hl-stats .hl-swatch { display: inline-block; width: 10px; height: 10px;
      margin-right: 6px; border-radius: 2px; vertical-align: -1px; }
    body.hl-dragover #hl-plot { outline: 3px dashed #1f77b4; outline-offset: -6px; }
  `;

  function start(Plotly, document, data) {
    const config = data.config;
    const state = { rows: [], name: "", view: VIEWS.includes(data.view) ? data.view : "inlet" };
    let built = null;
    let hovered = null;

    const style = document.createElement("style");
    style.textContent = STYLE;
    document.head.appendChild(style);
    const page = document.createElement("div");
    page.id = "hl-page";
    page.innerHTML = `
      <div id="hl-bar">
        <span id="hl-title"></span><span id="hl-counts"></span>
        <label>View <select id="hl-view">
          <option value="inlet">Vessels from an inlet</option>
          <option value="branch_order">By branch order</option>
        </select></label>
        <button id="hl-load" type="button">Load CSV…</button>
        <button id="hl-stats-button" type="button" aria-expanded="false">Branch order stats</button>
        <input id="hl-file" type="file" accept=".csv,text/csv" hidden>
        <span id="hl-status">Drop a connectivity CSV here to draw it.</span>
      </div>
      <div id="hl-plot"></div>
      <div id="hl-stats">
        <div id="hl-stats-head"><b>Per branch order</b>
          <button id="hl-stats-save" type="button">Save table CSV</button>
          <button id="hl-stats-close" type="button" aria-label="Close">✕</button></div>
        <table id="hl-stats-table"></table>
      </div>
      <div id="hl-tip"></div>`;
    document.body.prepend(page);
    const $ = (id) => document.getElementById(id);
    const plot = $("hl-plot");
    const tip = $("hl-tip");
    const select = $("hl-view");
    select.value = state.view;

    function fitPlot() {
      plot.style.top = `${$("hl-bar").offsetHeight}px`;
      $("hl-stats").style.top = `${$("hl-bar").offsetHeight + 8}px`;
    }

    function renderStats() {
      const orders = sortOrders(new Set(state.rows.map(orderOf)), config);
      const colourOf = new Map(orders.map((o, i) => [o, config.palette[i % config.palette.length]]));
      state.stats = orderStats(state.rows, config);
      const head = `<tr>${STATS_COLUMNS.map(([, h]) => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
      const body = state.stats.map((st) => {
        const cells = STATS_COLUMNS.map(([key, , d]) => {
          const text = escapeHtml(statsCell(st[key], d));
          if (key !== "order" || st.order === "All") return `<td>${text}</td>`;
          return `<td><span class="hl-swatch" style="background:${colourOf.get(st.order)}"></span>${text}</td>`;
        });
        return `<tr${st.order === "All" ? ' class="hl-all"' : ""}>${cells.join("")}</tr>`;
      });
      $("hl-stats-table").innerHTML = head + body.join("");
    }

    function toggleStats(open) {
      $("hl-stats").classList.toggle("open", open);
      $("hl-stats-button").setAttribute("aria-expanded", String(open));
    }

    function setStatus(text, error) {
      $("hl-status").textContent = text;
      $("hl-status").className = error ? "error" : "";
    }

    function hideHover() {
      tip.style.display = "none";
      if (hovered && hovered.kind === "vessel") {
        Plotly.restyle(plot, { x: [[]], y: [[]] }, [built.traces.length]);
      }
      hovered = null;
    }

    function draw() {
      hideHover();
      built = buildView(state.rows, state.view, config);
      const c = built.counts;
      $("hl-title").textContent = state.name || "Network connectivity";
      $("hl-counts").textContent =
        `${c.vessels} vessels, ${c.inlets} inlets, ${c.outlets} outlets`;
      document.title = `${state.name || "Network connectivity"} — 2D connectivity map`;
      renderStats();
      fitPlot();
      const highlight = {
        type: "scatter", mode: "lines", x: [], y: [], hoverinfo: "skip", showlegend: false,
        line: { color: "rgba(255, 193, 7, 0.75)", width: 10 },
      };
      Plotly.react(plot, [...built.traces, highlight], {
        xaxis: { ...built.xaxis, zeroline: false, showgrid: false },
        yaxis: { visible: false },
        hovermode: false,
        dragmode: "pan",
        plot_bgcolor: "white",
        legend: { itemsizing: "constant" },
        margin: { t: 20, r: 20, b: 50, l: 20 },
        uirevision: `${state.name}|${state.view}|${state.rows.length}`,
      }, { responsive: true, scrollZoom: true, displaylogo: false });
    }

    function onMouseMove(event) {
      if (!built || event.buttons) return hideHover();
      const full = plot._fullLayout;
      if (!full || !full.xaxis || !full.yaxis) return hideHover();
      const xa = full.xaxis;
      const ya = full.yaxis;
      const box = plot.getBoundingClientRect();
      const px = event.clientX - box.left - xa._offset;
      const py = event.clientY - box.top - ya._offset;
      if (px < 0 || py < 0 || px > xa._length || py > ya._length) return hideHover();
      const item = nearestItem(
        built.items, px, py,
        (x, y) => [xa.l2p(x), ya.l2p(y)],
        (it) => (plot.data[it.trace] || {}).visible !== "legendonly"
      );
      if (!item) return hideHover();
      if (item !== hovered) {
        if (hovered && hovered.kind === "vessel" && item.kind !== "vessel") {
          Plotly.restyle(plot, { x: [[]], y: [[]] }, [built.traces.length]);
        }
        if (item.kind === "vessel") {
          Plotly.restyle(
            plot,
            { x: [[item.x0, item.x1]], y: [[item.y0, item.y1]], "line.width": item.width + 8 },
            [built.traces.length]
          );
        }
        hovered = item;
        tip.innerHTML = item.text;
      }
      tip.style.display = "block";
      const gap = 14;
      const left = event.clientX + gap + tip.offsetWidth > window.innerWidth
        ? event.clientX - gap - tip.offsetWidth : event.clientX + gap;
      const topPx = event.clientY + gap + tip.offsetHeight > window.innerHeight
        ? event.clientY - gap - tip.offsetHeight : event.clientY + gap;
      tip.style.left = `${Math.max(0, left)}px`;
      tip.style.top = `${Math.max(0, topPx)}px`;
    }

    function load(file) {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const rows = parseCsv(String(reader.result));
          if (!rows.length) throw new Error("it has no vessel rows");
          state.rows = rows;
          state.name = file.name.replace(/\.csv$/i, "");
          draw();
          setStatus(`Loaded ${file.name}.`, false);
        } catch (error) {
          setStatus(`Could not draw ${file.name}: ${error.message}`, true);
        }
      };
      reader.onerror = () => setStatus(`Could not read ${file.name}.`, true);
      reader.readAsText(file);
    }

    select.addEventListener("change", () => {
      state.view = select.value;
      draw();
    });
    $("hl-load").addEventListener("click", () => $("hl-file").click());
    $("hl-stats-button").addEventListener("click", () =>
      toggleStats(!$("hl-stats").classList.contains("open"))
    );
    $("hl-stats-close").addEventListener("click", () => toggleStats(false));
    $("hl-stats-save").addEventListener("click", () => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([statsCsv(state.stats || [])], { type: "text/csv" }));
      link.download = `${state.name || "network"}_branch_order_stats.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });
    $("hl-file").addEventListener("change", (event) => {
      load(event.target.files[0]);
      event.target.value = "";
    });
    document.addEventListener("dragover", (event) => {
      event.preventDefault();
      document.body.classList.add("hl-dragover");
    });
    document.addEventListener("dragleave", (event) => {
      if (!event.relatedTarget) document.body.classList.remove("hl-dragover");
    });
    document.addEventListener("drop", (event) => {
      event.preventDefault();
      document.body.classList.remove("hl-dragover");
      load(event.dataTransfer.files[0]);
    });
    plot.addEventListener("mousemove", onMouseMove);
    plot.addEventListener("mouseleave", hideHover);
    plot.addEventListener("wheel", hideHover, { passive: true });
    window.addEventListener("resize", fitPlot);

    try {
      state.rows = parseCsv(data.csv || "");
      state.name = data.name || "";
      draw();
    } catch (error) {
      state.rows = [];
      draw();
      setStatus(`Could not draw ${data.name || "the CSV"}: ${error.message}`, true);
    }
  }

  const api = {
    VIEWS, parseCsv, ends, orderSortKey, inletLayout, branchLayout, buildView, nearestItem,
    orderStats, statsCsv, start,
  };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.HaemoLynxConnectivityMap = api;
  }
})(typeof window !== "undefined" ? window : this);
