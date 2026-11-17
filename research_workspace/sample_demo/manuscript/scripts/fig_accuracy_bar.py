#!/usr/bin/env python3
"""Generate publication-style SVG bar chart from verified experimental_log.md values."""

from pathlib import Path

DATA = [
    ("TF-IDF + LR", 84.2, "#4D4D4D"),
    ("Encoder (Unmasked)", 88.5, "#0072B2"),
    ("Encoder + SMC (Ours)", 91.4, "#009E73"),
]

OUT_PATH = Path(__file__).resolve().parent.parent / "figures" / "fig_accuracy_bar.svg"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

bars = []
for idx, (label, val, color) in enumerate(DATA):
    x = 90 + idx * 170
    height = (val - 75.0) * 10.0
    y = 220 - height
    bars.append(
        f'<rect x="{x}" y="{y:.1f}" width="90" height="{height:.1f}" fill="{color}" rx="3"/>'
        f'<text x="{x+45}" y="{y-8:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">{val:.1f}%</text>'
        f'<text x="{x+45}" y="242" text-anchor="middle" font-size="11" font-family="sans-serif">{label}</text>'
    )

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="600" height="270" viewBox="0 0 600 270">
  <rect width="600" height="270" fill="#ffffff"/>
  <line x1="65" y1="30" x2="65" y2="220" stroke="#333333" stroke-width="1.2"/>
  <line x1="65" y1="220" x2="570" y2="220" stroke="#333333" stroke-width="1.2"/>
  <text x="22" y="130" transform="rotate(-90 22,130)" text-anchor="middle" font-size="12" font-family="sans-serif">Macro-F1 (%)</text>
  {''.join(bars)}
</svg>
"""

OUT_PATH.write_text(svg, encoding="utf-8")
