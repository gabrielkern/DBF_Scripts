# Note: All credit to Dr. Kyle Greiner, PhD
# This is just his stupendous matlab code converted to python

import numpy as np
import matplotlib.pyplot as plt

#######################################
MTOF = 18 # lbs
CD_takeoff = 0.15
CL_takeoff = 1.0
CL_approach = 1.0
CL_max = 1.25
Ground_Friction = 0.04
Ground_Run = 50  # ft
Lift_Off_Speed = 40  # intakes mph turns it to ft/s
g = 32.174  # ft/s^2
rho = 0.0022  # slugs/ft^3
Cruise_Speed = 82  # ft/s
Vert_Speed = 30  # ft/s straight vertical velocity
Approach_Speed = 50  # intakes mph turns it to ft/s
Stall_Speed = 40
CD_min = 0.025
max_L_D = 14
# e = 0.8
# span = 60 / 12  # span in inches converted to feet
k_guess = 0.1
Turn_Bank = 60  # degrees
#######################################

q_ground = 0.5 * rho * (Lift_Off_Speed)**2
W_S = np.linspace(0.001, 9, 1000)

TW_takeoff = (Lift_Off_Speed**2 / (2*g*Ground_Run) + 
              q_ground*CD_takeoff / W_S + 
              Ground_Friction * (1 - q_ground*CL_takeoff/W_S))

q_cruise = 0.5 * rho * Cruise_Speed**2

TW_climb = (Vert_Speed/Cruise_Speed + 
            q_cruise / W_S * CD_min + 
            k_guess/q_cruise * W_S)

n = 1 / np.cos(np.radians(Turn_Bank))

TW_turn = q_cruise * (CD_min/W_S + k_guess*(n/q_cruise)**2 * W_S)

TW_cruise = q_cruise*CD_min/W_S + k_guess/q_cruise * W_S

q_approach = 0.5 * rho * (Approach_Speed)**2

q_stall = 0.5 * rho * (Stall_Speed)**2

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(W_S, TW_takeoff, label='Takeoff')
plt.plot(W_S, TW_climb, label='Climb')
plt.plot(W_S, TW_turn, label='Turn')
plt.plot(W_S, TW_cruise, label='Cruise')

# Add vertical line for approach
#plt.axvline(x=q_approach*CL_approach, color='black', linestyle='--', label='Approach')

# Add vertical line for stall
plt.axvline(x=q_stall*CL_max, color='purple',linestyle='--',label='Stall')

# Add horizontal line for desired L/D (efficiency)
plt.axhline(y=1/max_L_D, color='black',linestyle='--',label='L/D')

# Uncommented intersection finding code (converted to Python)
"""
vec1 = TW_takeoff
vec2 = TW_climb
v = vec1 - vec2  # choose the two vectors to grab the intersection of
for i in range(len(W_S)):
    if np.sign(v[i]) <= 0:
        index = i
        break

print(f"\nYour S value is: {MTOF/W_S[index-1]:.6f} ft^2")
print(f"Your T value is: {MTOF*vec1[index-1]:.6f} lbs\n")
plt.scatter(W_S[index-1], vec1[index-1], s=100, c='red', marker='o', zorder=5)
"""

plt.legend()
plt.xlabel('Wing Loading lbs/ft^2 (W/S)')
plt.ylabel('Thrust to Weight Ratio (T/W)')
plt.ylim([0, 2])
plt.xlim([0, 9])
plt.grid(True, alpha=0.3)
plt.title('Aircraft Performance Constraints')
plt.show()