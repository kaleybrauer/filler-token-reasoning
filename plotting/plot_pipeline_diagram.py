"""
Recreates the 4-stage pipeline figure:
  Stage 1: Hidden-state extraction (cube)
  Stage 2: Residual logit lens (bar chart)
  Stage 3: Aggregation across (ℓ, p) (ranked list)
  Stage 4: LLM-as-judge decode (judge + output box)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import numpy as np

# ---------- canvas ----------
fig, ax = plt.subplots(figsize=(19, 7))
ax.set_xlim(0, 88)
ax.set_ylim(0, 36)
ax.set_aspect('auto')
ax.axis('off')
ax.patch.set_visible(False)
fig.patch.set_visible(False)

# Column centers for the 4 stages (tighter spacing)
X = [11, 31, 51, 70]
TITLES = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
SUBS = ['Hidden-state extraction',
        'Residual logit lens',
        'Aggregation across (ℓ, p)',
        'LLM-as-judge decode']

# Footer captions (dimensions + description)
DIMS = ['L × P × d', 'L × P × |V|', '50 tokens', 'decoded value']
DESCS = ['hidden states', 'residual lens distributions',
         'aggregated per example', '+ confidence + backups']

# ---------- titles ----------
for x, t, s in zip(X, TITLES, SUBS):
    ax.text(x, 34.2, t, ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(x, 32.4, s, ha='center', va='center', fontsize=12)

# ---------- footer captions ----------
for x, d, desc in zip(X, DIMS, DESCS):
    ax.text(x, 7.3, d, ha='center', va='center', fontsize=10, style='italic', color='#888888')
    ax.text(x, 6.3, desc, ha='center', va='center', fontsize=10, style='italic', color='#888888')

# ============================================================
# STAGE 1 — Question stub + 3D cube
# ============================================================

# Question stub box
ax.add_patch(FancyBboxPatch((3, 26), 16, 4, boxstyle="round,pad=0.25",
                            facecolor='#faf0e6', edgecolor='#c8b8a8', linewidth=0.8))
ax.text(11, 28.6, 'Question: ...fact A plus X?', ha='center', va='center',
        fontsize=11, style='italic')
ax.text(9.2, 27.0, 'Filler: . . . . . . . . . .', ha='center', va='center',
        fontsize=11, style='italic', color='#c0392b')
ax.text(13.2, 27.0, 'Answer:', ha='left', va='center',
        fontsize=11, style='italic', color='black')

# 3D cube (lowered by 2)
def draw_cube(ax, x0, y0, w, h, depth):
    # Front face
    ax.add_patch(Polygon([(x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h)],
                         facecolor='#f0c8a8', edgecolor='#a07050', linewidth=1.2))
    # Top face
    ax.add_patch(Polygon([(x0, y0+h), (x0+w, y0+h),
                          (x0+w+depth, y0+h+depth), (x0+depth, y0+h+depth)],
                         facecolor='#f5dcc8', edgecolor='#a07050', linewidth=1.2))
    # Right face
    ax.add_patch(Polygon([(x0+w, y0), (x0+w+depth, y0+depth),
                          (x0+w+depth, y0+h+depth), (x0+w, y0+h)],
                         facecolor='#d4a882', edgecolor='#a07050', linewidth=1.2))
    # Grid lines on the front
    n = 6
    for i in range(1, n):
        ax.plot([x0 + i*w/n, x0 + i*w/n], [y0, y0+h],
                color='#a07050', linewidth=0.4, alpha=0.7)
        ax.plot([x0, x0+w], [y0 + i*h/n, y0 + i*h/n],
                color='#a07050', linewidth=0.4, alpha=0.7)
    # Grid on top
    for i in range(1, n):
        ax.plot([x0 + i*w/n, x0 + i*w/n + depth],
                [y0+h, y0+h+depth], color='#a07050', linewidth=0.4, alpha=0.7)

draw_cube(ax, 6.5, 12.5, 8, 9, 1.8)

# Cube axis labels
# ax.annotate('', xy=(5.8, 21.5), xytext=(5.8, 12.5),
#             arrowprops=dict(arrowstyle='-', color='#555555', lw=0.8))
ax.text(5.7, 17, 'layer ℓ', ha='center', va='center', fontsize=10, rotation=90)

# ax.annotate('', xy=(14.5, 11.8), xytext=(6.5, 11.8),
#             arrowprops=dict(arrowstyle='-', color='#555555', lw=0.8))
ax.text(10.5, 11.4, 'filler position p', ha='center', va='center', fontsize=10)

ax.text(16.8, 17.5, 'd_model', ha='left', va='center', fontsize=10, rotation=90)

# Stage-1 caption
ax.text(11, 9.2, 'hidden state h(ℓ, p) per example',
        ha='center', va='center', fontsize=10.5)

# ============================================================
# STAGE 2 — Residual logit lens
# ============================================================

# Formula box
cx2 = X[1]  # 31
ax.add_patch(FancyBboxPatch((cx2 - 9, 26), 18, 4, boxstyle="round,pad=0.3",
                            facecolor='#fde8c8', edgecolor='#d08a3a', linewidth=0.9))
ax.text(cx2, 28.6, r'$p = \mathrm{softmax}(W_U \cdot \mathrm{RMS}(h))$',
        ha='center', va='center', fontsize=12)
ax.text(cx2, 26.9, r'$r = p - \langle p \rangle$',
        ha='center', va='center', fontsize=12)

# # Caption above bars
# ax.text(cx2, 24.6, 'top tokens by residual score',
#         ha='center', va='center', fontsize=9, color='#666666')

# Bar chart — stacked residual bars with bounding box
bar_x0, bar_x1 = cx2 - 7.5, cx2 + 7.5
bar_y0, bar_y_top = 14.5, 23.8
ax.add_patch(Rectangle((bar_x0 - 0.4, bar_y0 - 0.05),
                       (bar_x1 - bar_x0) + 0.8,
                       (bar_y_top - bar_y0),
                       facecolor='#fafaf6', edgecolor='#b0b0a0', linewidth=0.7,
                       zorder=0))

# Gray bars (cross-example mean) decrease left-to-right; highlighted tokens
# have a short gray base with a tall rust residual stacked on top.
gray_h = np.array([4.4, 3.4, 1.3, 2.5, 1.3, 1.8, 0.9, 1.3, 0.9, 0.7, 0.5, 0.45, 0.4])
red_h = np.zeros_like(gray_h)
red_h[2] = 5.2
red_h[4] = 4.3
red_h[6] = 3.3

n_bars = len(gray_h)
xs = np.linspace(bar_x0 + 0.4, bar_x1 - 0.4, n_bars, endpoint=False)
bw = (xs[1] - xs[0]) * 0.55
for i, xb in enumerate(xs):
    ax.add_patch(Rectangle((xb, bar_y0), bw, gray_h[i],
                           facecolor='#bbbbbb', edgecolor='none', zorder=2))
    if red_h[i] > 0:
        ax.add_patch(Rectangle((xb, bar_y0 + gray_h[i]), bw, red_h[i],
                               facecolor='#c0492a', edgecolor='none', zorder=2))

# Bar chart axes
ax.plot([bar_x0 - 0.2, bar_x1 + 0.2], [bar_y0, bar_y0], color='#444444', linewidth=0.7)
ax.text(bar_x0 - 0.4, bar_y0, '0', ha='right', va='center', fontsize=9, color='#444444')
ax.text(bar_x1 + 0.2, bar_y0 - 0.1, 'vocab', ha='right', va='top',
        fontsize=9, color='#444444')

# Caption below bars
ax.text(cx2, 13, 'keep top-T tokens by residual',
        ha='center', va='center', fontsize=10.5)
ax.text(cx2, 11.8, '(T = 30, layers 30–60)',
        ha='center', va='center', fontsize=9.5, style='italic', color='#888888')

# Legend
lx = cx2 - 7
ax.add_patch(Rectangle((lx, 9.6), 0.7, 0.55, facecolor='#bbbbbb'))
ax.text(lx + 1, 10, r'cross-example mean $\langle p \rangle$',
        ha='left', va='center', fontsize=9.5, color='#555555')
ax.add_patch(Rectangle((lx, 8.4), 0.7, 0.55, facecolor='#c0492a'))
ax.text(lx + 1, 8.8, 'example-specific residual',
        ha='left', va='center', fontsize=9.5, color='#555555')

# ============================================================
# STAGE 3 — Aggregation
# ============================================================

cx3 = X[2]  # 51

# Formula / description box
ax.add_patch(FancyBboxPatch((cx3 - 8, 26), 16, 4, boxstyle="round,pad=0.3",
                            facecolor='#fef3c8', edgecolor='#c8a030', linewidth=0.9))
ax.text(cx3, 28.6, r'$s(t) = \sum_{(\ell,\,p)} r(t)$',
        ha='center', va='center', fontsize=12)
ax.text(cx3, 26.9, '→ rank tokens, keep top 50',
        ha='center', va='center', fontsize=10.5)

# Bounding box around ranked token list
ax.add_patch(FancyBboxPatch((cx3 - 7, 11.0), 14, 11.8, boxstyle="round,pad=0.3",
                            facecolor='#fafaf6', edgecolor='#b0b0a0', linewidth=0.9))

# Ranked token list
tokens = [
    ("1.", "37.3", "' boron'"),
    ("2.", "13.5", "' Neptune'"),
    ("3.", " 8.7", "'93'"),
    ("4.", " 7.0", "' five'"),
    ("5.", " 6.1", "'5'"),
    ("6.", " 4.8", "'Np'"),
    ("7.", " 4.4", "' Calif…'"),
    ("8.", " 4.0", "' uranium'"),
    ("...", "...", "..."),
]
list_top = 22
for i, (n, sc, tk) in enumerate(tokens):
    y = list_top - i * 1.25
    ax.text(cx3 - 6, y, n,  ha='left', va='center', fontsize=9.5, family='monospace')
    ax.text(cx3 - 3.8, y, sc, ha='left', va='center', fontsize=9.5, family='monospace')
    ax.text(cx3 - 0.5, y, tk, ha='left', va='center', fontsize=9.5, family='monospace')

ax.text(cx3, 9.5, 'one ranked list per example',
        ha='center', va='center', fontsize=10.5)

# ============================================================
# STAGE 4 — LLM-as-judge decode
# ============================================================

cx4 = X[3]  # 71

# Judge box
ax.add_patch(FancyBboxPatch((cx4 - 8, 25), 16, 5, boxstyle="round,pad=0.3",
                            facecolor='#f0d0c0', edgecolor='#b06040', linewidth=0.9))
ax.text(cx4, 29, 'LLM judge', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax.text(cx4, 27.5, '"What number(s) or concept(s)', ha='center', va='center',
        fontsize=10, style='italic', color='#444444')
ax.text(cx4, 26.2, 'is the model thinking about?"', ha='center', va='center',
        fontsize=10, style='italic', color='#444444')

# Output box
ax.add_patch(FancyBboxPatch((cx4 - 6.5, 12), 13, 11, boxstyle="round,pad=0.3",
                            facecolor='#fafaf6', edgecolor='#b0b0a0', linewidth=0.9))
ax.text(cx4, 21.7, 'Output', ha='center', va='center',
        fontsize=11, fontweight='bold')
ax.text(cx4 - 5.5, 19.7, 'primary:    {93, 5}',
        ha='left', va='center', fontsize=10.5, family='monospace')
ax.text(cx4 - 5.5, 18.0, f'conf:    {0.89, 0.71}',
        ha='left', va='center', fontsize=10.5, family='monospace')
ax.text(cx4 - 5.5, 16.3, 'backups:',
        ha='left', va='center', fontsize=10.5, family='monospace')
ax.text(cx4 - 4.5, 14.6, '[6, 92, 7, 94, …]',
        ha='left', va='center', fontsize=10.5, family='monospace')

ax.text(cx4, 10.3, 'guess at hidden intermediate',
        ha='center', va='center', fontsize=10.5)

# ============================================================
# Arrows between stages (equal length, short)
# ============================================================
arrow_props = dict(arrowstyle='->', color='#333333', lw=1.6,
                   shrinkA=0, shrinkB=0, mutation_scale=18)
for x_start, x_end in [(19.5, 21.5), (40.5, 42.5), (59.5, 61.5)]:
    ax.annotate('', xy=(x_end, 18), xytext=(x_start, 18),
                arrowprops=arrow_props)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('plotting/pipeline_figure.pdf', bbox_inches='tight', pad_inches=0.05, facecolor='white')
plt.savefig('plotting/pipeline_figure.png', bbox_inches='tight', pad_inches=0.05, dpi=150, facecolor='white')
print("Saved pipeline_figure.pdf and pipeline_figure.png")
