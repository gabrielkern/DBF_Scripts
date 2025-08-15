import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import glob
import os

plt.style.use("seaborn-v0_8-whitegrid")

# USER SETTINGS
aoades = 3
howtofind = "*2412.csv"
pic = 1  # 0 = no picture, 1 = background picture
pic_filename = "background.jpg"  # only used if pic = 1


################# FUNCTIONS
def read(filename):
    df = pd.read_csv(filename, skiprows=6, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def analyze(df, aoades):
    idx = (df['alpha'] - aoades).abs().idxmin()
    cl = df.loc[idx, 'CL']
    cd = df.loc[idx, 'CD']
    cdi = df.loc[idx, 'CDi']
    cdv = df.loc[idx, 'CDv']
    ld = cl/cd
    ld_max = (df['CL'] / df['CD']).max()
    return cd, cdi, cdv,  cl, ld, ld_max
#################


# Initialize 
taper_ratios, cl_list, cd_list, cdi_list, cdv_list, ld_list, ld_max_list = [], [], [], [], [], [], []
plt.close('all')

# Grab Files
script_dir = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(script_dir, howtofind)))

# Meaty Loop
for filename in files:
    df = read(filename)
    cd, cdi, cdv, cl, ld, ld_max = analyze(df, aoades)
    basename = os.path.basename(filename)    # gets the file name
    taper_ratio = float(basename.split("_")[0]) / 10 # pulls the first two digits for taper
    taper_ratios.append(taper_ratio)
    cl_list.append(cl)
    cd_list.append(cd)
    cdi_list.append(cdi)
    cdv_list.append(cdv)
    ld_list.append(ld)
    ld_max_list.append(ld_max)

# Sort data
sort_idx = np.argsort(taper_ratios)
taper_ratios = np.array(taper_ratios)[sort_idx]
cl_list = np.array(cl_list)[sort_idx]
cd_list = np.array(cd_list)[sort_idx]
cdi_list = np.array(cdi_list)[sort_idx]
cdv_list = np.array(cdv_list)[sort_idx]
ld_list = np.array(ld_list)[sort_idx]
ld_max_list = np.array(ld_max_list)[sort_idx]




########### PLOTTING ############

# --- Load and brighten background image if needed ---
if pic == 1:
    bg_img = mpimg.imread(os.path.join(script_dir, pic_filename)).astype(float)
    if bg_img.max() > 1.0:  # normalize if needed
        bg_img /= 255.0
    bg_img = np.clip(bg_img * 1.0, 0, 1)  # change the decimal of the 1.0 (+0.3 means 30% brighter)

# --- Create figure ---
fig, axes = plt.subplots(3, 2, figsize=(14, 8))

# Add background image if pic == 1
if pic == 1:
    fig.patch.set_alpha(0)
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg_ax.imshow(bg_img, aspect='auto', extent=[0, 1, 0, 1], transform=bg_ax.transAxes, alpha=0.5)
    bg_ax.axis('off')

fig.subplots_adjust(wspace=0.3, hspace=0.3)

# Helper to style subplot with white boxes
def style_ax(ax):
    if pic == 1:
        ax.set_facecolor((1, 1, 1, 0.4))
    for spine in ax.spines.values():
        spine.set_alpha(0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.xaxis.label.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    ax.yaxis.label.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    if ax.get_title():
        ax.title.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

# Top-left: CL
axes[0, 0].plot(taper_ratios, cl_list, marker='o', linewidth=2, alpha=0.8)
axes[0, 0].set_xlabel("Taper Ratio")
axes[0, 0].set_ylabel(f"CL at AoA = {aoades}°")
style_ax(axes[0, 0])

# Top Right: CD Total
axes[0, 1].plot(taper_ratios, cd_list, marker='o', color='r', linewidth=2, alpha=0.8)
axes[0, 1].set_xlabel("Taper Ratio")
axes[0, 1].set_ylabel(f"CD at AoA = {aoades}°")
style_ax(axes[0,1])

# Middle Left: CD viscous
axes[1, 0].plot(taper_ratios, cdv_list, marker='o', color='g', linewidth=2, alpha=0.8)
axes[1, 0].set_xlabel("Taper Ratio")
axes[1, 0].set_ylabel(f"CDv at AoA = {aoades}°")
style_ax(axes[1, 0])

# Middle Right: CD induced
axes[1, 1].plot(taper_ratios, cdi_list, marker='o', color='b', linewidth=2, alpha=0.8)
axes[1, 1].set_xlabel("Taper Ratio")
axes[1, 1].set_ylabel(f"CDi at AoA = {aoades}°")
style_ax(axes[1, 1])


# Bottom Left: L/D 
axes[2,0].plot(taper_ratios, ld_list, marker='o', color='m', linewidth=2, alpha=0.8)
axes[2,0].set_xlabel("Taper Ratio")
axes[2,0].set_ylabel(f"L/D at AoA = {aoades}°")
style_ax(axes[2,0])

# Bottom-right: Max L/D
axes[2, 1].plot(taper_ratios, ld_max_list, marker='o', color='c', linewidth=2, alpha=0.8)
axes[2, 1].set_xlabel("Taper Ratio")
axes[2, 1].set_ylabel("Max L/D")
style_ax(axes[2, 1])

# Figure title with white box if pic == 1
fig_title = f"Airfoil Performance vs Taper Ratio (NACA 2412, AoA = {aoades}°)"
st = fig.suptitle(fig_title, fontsize=14, fontweight='bold')
if pic == 1:
    st.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.4'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()