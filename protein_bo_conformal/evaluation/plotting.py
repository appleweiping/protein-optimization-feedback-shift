"""Simple plotting helpers for baseline evaluation artifacts."""

from __future__ import annotations

from pathlib import Path


def _svg_shell(title: str, width: int, height: int, body: str) -> str:
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>
{body}
</svg>"""


def write_curve_svg(
    path: Path,
    title: str,
    method_curves: dict[str, list[dict[str, float]]],
    y_label: str,
) -> None:
    """Write a mean ± std curve comparison figure in pure SVG."""
    width = 860
    height = 460
    margin = 58
    colors = {
        "random": "#64748b",
        "greedy": "#2563eb",
        "ucb": "#dc2626",
        "ei": "#059669",
        "conformal_ucb": "#7c3aed",
    }
    all_points = [point for curve in method_curves.values() for point in curve]
    if not all_points:
        path.write_text(_svg_shell(title, width, height, ""), encoding="utf-8")
        return

    xs = [float(point["step"]) for point in all_points]
    ys = [
        value
        for point in all_points
        for value in (float(point["mean"] - point["std"]), float(point["mean"] + point["std"]))
    ]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1e-6)

    def map_x(value: float) -> float:
        return margin + ((value - min_x) / span_x) * (width - 2 * margin)

    def map_y(value: float) -> float:
        return height - margin - ((value - min_y) / span_y) * (height - 2 * margin)

    layers: list[str] = [
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />",
    ]
    legend: list[str] = []
    for index, (method, curve) in enumerate(method_curves.items()):
        color = colors.get(method, "#0f172a")
        lower_points = [
            (map_x(float(point["step"])), map_y(float(point["mean"] - point["std"])))
            for point in curve
        ]
        upper_points = [
            (map_x(float(point["step"])), map_y(float(point["mean"] + point["std"])))
            for point in reversed(curve)
        ]
        band_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in lower_points + upper_points)
        line_points = " ".join(
            f"{map_x(float(point['step'])):.1f},{map_y(float(point['mean'])):.1f}"
            for point in curve
        )
        layers.append(f"<polygon points='{band_points}' fill='{color}' fill-opacity='0.12' stroke='none' />")
        layers.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.6' points='{line_points}' />")
        legend_y = 46 + index * 18
        legend.append(f"<rect x='{width - 210}' y='{legend_y}' width='12' height='12' fill='{color}' />")
        legend.append(
            f"<text x='{width - 190}' y='{legend_y + 10}' font-size='12' font-family='Arial'>{method}</text>"
        )

    body = (
        "".join(layers)
        + "".join(legend)
        + f"<text x='{width - margin}' y='{height - margin + 24}' text-anchor='end' font-size='11'>budget step</text>"
        + f"<text x='{margin - 10}' y='{margin - 12}' text-anchor='start' font-size='11'>{y_label}</text>"
    )
    path.write_text(_svg_shell(title, width, height, body), encoding="utf-8")


def write_bar_svg(
    path: Path,
    title: str,
    method_values: dict[str, float],
    y_label: str,
) -> None:
    """Write a simple bar chart for scalar summary metrics."""
    width = 760
    height = 420
    margin = 58
    colors = ["#64748b", "#2563eb", "#dc2626", "#059669", "#7c3aed"]
    methods = list(method_values.keys())
    if not methods:
        path.write_text(_svg_shell(title, width, height, ""), encoding="utf-8")
        return

    max_value = max(method_values.values()) if method_values else 1.0
    min_value = min(0.0, min(method_values.values()))
    span = max(max_value - min_value, 1e-6)
    band_width = (width - 2 * margin) / max(len(methods), 1)
    bar_width = band_width * 0.55

    bars: list[str] = [
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />",
    ]
    for index, method in enumerate(methods):
        value = float(method_values[method])
        normalized = (value - min_value) / span
        bar_height = normalized * (height - 2 * margin)
        x = margin + index * band_width + (band_width - bar_width) / 2
        y = height - margin - bar_height
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' fill='{colors[index % len(colors)]}' />"
        )
        bars.append(
            f"<text x='{x + bar_width / 2:.1f}' y='{height - margin + 18}' text-anchor='middle' font-size='11'>{method}</text>"
        )
        bars.append(
            f"<text x='{x + bar_width / 2:.1f}' y='{y - 6:.1f}' text-anchor='middle' font-size='11'>{value:.3f}</text>"
        )

    body = "".join(bars) + f"<text x='{margin - 10}' y='{margin - 12}' text-anchor='start' font-size='11'>{y_label}</text>"
    path.write_text(_svg_shell(title, width, height, body), encoding="utf-8")


def write_grouped_bar_svg(
    path: Path,
    title: str,
    grouped_values: dict[str, dict[str, float]],
    y_label: str,
) -> None:
    """Write grouped bars, for example early-stage vs late-stage metrics per method."""
    width = 860
    height = 440
    margin = 58
    methods = list(grouped_values.keys())
    series = sorted({name for values in grouped_values.values() for name in values.keys()})
    if not methods or not series:
        path.write_text(_svg_shell(title, width, height, ""), encoding="utf-8")
        return

    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    max_value = max(value for values in grouped_values.values() for value in values.values())
    min_value = min(0.0, min(value for values in grouped_values.values() for value in values.values()))
    span = max(max_value - min_value, 1e-6)
    group_width = (width - 2 * margin) / max(len(methods), 1)
    inner_width = group_width * 0.75
    bar_width = inner_width / max(len(series), 1)

    layers: list[str] = [
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />",
    ]
    legend: list[str] = []
    for method_index, method in enumerate(methods):
        group_start = margin + method_index * group_width + (group_width - inner_width) / 2
        for series_index, series_name in enumerate(series):
            value = float(grouped_values.get(method, {}).get(series_name, 0.0))
            normalized = (value - min_value) / span
            bar_height = normalized * (height - 2 * margin)
            x = group_start + series_index * bar_width
            y = height - margin - bar_height
            color = palette[series_index % len(palette)]
            layers.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width * 0.8:.1f}' height='{bar_height:.1f}' fill='{color}' />"
            )
        layers.append(
            f"<text x='{group_start + inner_width / 2:.1f}' y='{height - margin + 18}' text-anchor='middle' font-size='11'>{method}</text>"
        )

    for index, series_name in enumerate(series):
        legend_y = 44 + index * 18
        color = palette[index % len(palette)]
        legend.append(f"<rect x='{width - 220}' y='{legend_y}' width='12' height='12' fill='{color}' />")
        legend.append(
            f"<text x='{width - 200}' y='{legend_y + 10}' font-size='12' font-family='Arial'>{series_name}</text>"
        )

    body = "".join(layers) + "".join(legend) + f"<text x='{margin - 10}' y='{margin - 12}' text-anchor='start' font-size='11'>{y_label}</text>"
    path.write_text(_svg_shell(title, width, height, body), encoding="utf-8")
