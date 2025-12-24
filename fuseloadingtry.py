import numpy as np
import matplotlib.pyplot as plt


########################################################################

Acc = 606 # number of points do not touch causes problems with moment
WT = 5.5 # total weight lbs

#Fuelselage
Wb = 2.5 # body weight lbs
nw = 20  # g load
qload = [0.6,.035] # pounds per inch
qplacement = [22,32,13,58] # start,end first is pucks second ducks
placedlocations = [3,8,10] # location of motor,esc,battery
placedweights = [0.75,0.325,1] # weights corresponding to above
# x=0 at nose
cg = 27 # location of cg
w = 23.6 # quarter chord positions
L = 65 # total length

# U section properties
tw = 0.125 # inches this is t for the bottom
tf = 0.125 # inches this is for the sides
width = 6.5 #inches is the width h
height = 6.5 # inches is the height b
SigmaT_plywood = 4000 #psi
SigmaC_plywood = 4500 #psi




# Fuselage

# distributed load segments (pairs)
q_pairs = [(qplacement[i], qplacement[i+1]) for i in range(0, len(qplacement), 2)]

# distributed load totals & centroids
dist_totals = [q * (end - start) for q, (start, end) in zip(qload, q_pairs)]
dist_centroids = [(start + end) / 2.0 for (start, end) in q_pairs]

# point loads (downward = negative, scaled by g-load)
point_forces = [-wt * nw for wt in placedweights]   # placed weights
point_positions = placedlocations.copy()
point_forces.append(-Wb * nw)   # body weight
point_positions.append(cg)

# distributed loads as equivalent point loads
dist_forces_signed = [-f * nw for f in dist_totals]

# solve reactions at wing (x=w) and tail (x=L)
A = np.array([[1.0, 1.0],
              [w,   L  ]], dtype=float)
b = np.array([
    - (sum(point_forces) + sum(dist_forces_signed)),
    - (sum([f * p for f, p in zip(point_forces, point_positions)]) +
       sum([f * c for f, c in zip(dist_forces_signed, dist_centroids)]))
], dtype=float)
R_w, R_L = np.linalg.solve(A, b)

# collect all forces
force_positions = point_positions + [w, L]
force_values = point_forces + [R_w, R_L]
sorted_idx = sorted(range(len(force_positions)), key=lambda i: force_positions[i])
force_positions = [force_positions[i] for i in sorted_idx]
force_values = [force_values[i] for i in sorted_idx]

# x grid
x_vals = np.linspace(0.0, L, Acc)
dx = x_vals[1] - x_vals[0]

# shear (upward positive)
shear = np.zeros_like(x_vals)
for pos, val in zip(force_positions, force_values):
    shear += np.where(x_vals >= pos, val, 0.0)

for q_val, (start, end) in zip(qload, q_pairs):
    seg_len = end - start
    shear += -q_val * np.clip(x_vals - start, 0.0, seg_len) * nw

# integrate shear to get moment (lb*ft)
moment_in = np.zeros_like(x_vals)
acc = 0.0
for j in range(len(x_vals) - 1):
    acc += 0.5 * (shear[j] + shear[j+1]) * dx
    moment_in[j+1] = acc



FOS = np.zeros_like(x_vals)
passes = np.zeros_like(x_vals)
A = 2*height*tf+(width-2*tf)*tw
xc = 1/A * ( (width-2*tf)*tw / 2  + tf*height**2 )
Iy0 = (width-2*tf)*tw**3 / 3 + 2*tf*height**3 / 3
Iy = (Iy0 - A*xc**2) # in^4
for i in range(len(x_vals)):
    sigma_max_top = moment_in[i]*(height-xc) / Iy
    if sigma_max_top <= SigmaT_plywood:
        FOS[i] = SigmaT_plywood/sigma_max_top
        passes[i] = 1



print(min(abs(FOS)))



# plotting
fig, axs = plt.subplots(2,1, figsize=(12,8))
axs[0].plot(x_vals/12.0, shear, label='Shear Force', color='royalblue')
axs[0].plot(x_vals/12.0, moment_in/12.0, label='Moment', color='hotpink')
axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
axs[0].set_title("Fuselage Shear / Moment Diagram")
axs[0].set_xlabel("Fuselage Length (ft)")
axs[0].set_ylabel("Shear Force (lb) and Moment (lb/ft)")
axs[0].grid(True)
axs[0].legend()
axs[1].plot(x_vals/12.0, FOS)
#)
#axs[1].plot(x_vals/12.0, passes)
plt.show()