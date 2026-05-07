"""
Banner Testing: Coefficient of Drag (CD) vs Reynolds Number (Re)
-----------------------------------------------------------------
Reads raw force data from the Paper, Cloth, and Silk sheets of the
Banner_Testing.xlsx file and reproduces the CD vs Re scatter plot
described in the 'Data For DR Graph' tab.

Usage:
    python banner_cd_re_plot.py

Requirements:
    pip install openpyxl matplotlib numpy
"""

import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constants (from row 1 of 'Data For DR Graph') ─────────────────────────────
RHO   = 0.002377        # air density, slug/ft^3
MU    = 3.737e-7        # dynamic viscosity, slug/(ft·s)
L     = 5.0             # banner length, ft
AREA  = L * (L / 5)    # reference area = 5 ft^2  (formula: =I1*(I1/5))

# Speed columns in source sheets (0-indexed from row data tuple):
#   index 0 = label col (None), 1 = 0 mph, 2 = 10 mph, 3 = 20 mph, ...
SPEED_COLS = {
    2: 10,   # col C → 10 mph
    3: 20,   # col D → 20 mph
    4: 30,   # col E → 30 mph
    5: 40,   # col F → 40 mph
    6: 50,   # col G → 50 mph
    7: 60,   # col H → 60 mph
    8: 70,   # col I → 70 mph
    9: 80,   # col J → 80 mph
    10: 40,
    11: 50,
    12: 60,
}

MPH_TO_FPS = 5280 / 3600   # ≈ 1.46667 ft/s per mph


def load_force_data(ws):
    """
    Return a dict {mph_speed: [force_values]} from a source sheet.
    Skips the header row (row 1) and the 0 mph column.
    Ignores None values.
    """
    data = {mph: [] for mph in SPEED_COLS.values()}
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0:          # header row
            continue
        for col_idx, mph in SPEED_COLS.items():
            val = row[col_idx] if col_idx < len(row) else None
            if val is not None and isinstance(val, (int, float)):
                data[mph].append(val)
    return data


def compute_cd_re(force_data):
    """
    For each speed with data, compute:
        v   (ft/s)
        Re  = rho * v * L / mu
        CD  = F / (0.5 * rho * v^2 * Area)  for each force measurement F
    Returns lists of (Re, CD) pairs.
    """
    re_vals, cd_vals = [], []
    for mph, forces in force_data.items():
        if not forces:
            continue
        v   = mph * MPH_TO_FPS
        re  = RHO * v * L / MU
        q   = 0.5 * RHO * v**2 * AREA   # dynamic pressure × area
        for F in forces:
            cd = F / q
            re_vals.append(re)
            cd_vals.append(cd)
    return np.array(re_vals), np.array(cd_vals)


# ── Load workbook ──────────────────────────────────────────────────────────────
XLSX_FILE = "/Users/gabrielkern/Downloads/Banner_Testing.xlsx"   # ← update path if needed

wb = openpyxl.load_workbook(XLSX_FILE)

materials = {
    "Polyester Taffeta": "Paper",
    "Cloth":             "Cloth",
    "Silk":              "Silk",
}

colors  = ["steelblue", "darkorange", "seagreen"]
markers = ["o", "s", "^"]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for (mat_name, sheet_name), color, marker in zip(materials.items(), colors, markers):
    ws             = wb[sheet_name]
    force_data     = load_force_data(ws)
    re_arr, cd_arr = compute_cd_re(force_data)

    # Remove non-physical / zero-speed noise (CD can be meaninglessly large near v≈0)
    mask = cd_arr > -0.5   # keep physically plausible values; tweak if needed

    ax.scatter(re_arr[mask], cd_arr[mask],
               label=mat_name,
               color=color,
               marker=marker,
               s=30,
               alpha=0.65,
               edgecolors="none")

ax.set_xlabel("Reynolds Number  $Re_b = \\rho\\, v\\, L \\,/\\, \\mu$", fontsize=13)
ax.set_ylabel("Coefficient of Drag  $C_D$", fontsize=13)
ax.set_title("Banner Material Drag Coefficient vs. Reynolds Number", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax.set_ylim(bottom=None)   # auto-scale; tighten if needed

# Format x-axis with scientific notation
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

plt.tight_layout()
plt.savefig("banner_cd_re_plot.png", dpi=150)
plt.show()
print("Plot saved to banner_cd_re_plot.png")