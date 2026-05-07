# OHHHHHHH YEAHHHHHHHHHHH
# Run with: python VECTOOOOORRRRRRRRR.py

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline,interp1d
from typing import Dict

from LapSimulator import apc_data

# SET STATIC PARAMETERS
LAP_ALTITUDE = 50 #ft
CLIMB_ANGLE = 15 #deg
WING_AREA = 4.5833 #ft^2
AR = 5.455 #unitless
E = 0.9 #unitless
HORIZ_TAIL_AREA = 1 #ft^2
CD_TAIL = 0.02 #unitless
CD_PARA = 0.02 #unitless
CL_TURN = 0.8 #unitless
CL_STALL = 1.0 #unitless, max lift coefficient before stall
WEIGHT = 10 #lbs
GRAVITY = 32.174 #ft/s^2
RHO = 0.0023769 #slugs/ft^3 at sea level
DT = 0.05 #seconds
BATTERY_CAPACITY = 4500 #mAh
BATTERY_CELLS = 6 #number of cells in series
LAP_COUNT = 0 # Number of laps that the program should run, set to 0 to attempt max laps
PROPELLER_DIAMETER = 16 #inches
PROPELLER_PITCH = 10 #inches
MOTOR_KV = 520 #RPM/Volt
BANNER_LENGTH = 13 #ft, set to 0 to disable banner drag
PROPULSIVE_EFFICIENCY = 0.75

"""
    To get thrust, call calculate_thrust(vel_fps, thrust_interp), where
    vel_fps is the velocity in feet per second and thrust_interp is a constant. 
    bring thrust_interp into functions

    To get battery charge, call calculate_charge(current_charge, vel_fps, dt, batt_interp),
    where current_charge is the current battery charge in mAh, vel_fps is the
    velocity in feet per second, dt is the timestep in seconds, and batt_interp
    is a constant. bring battery_interp into functions

    To access different constants or states, use the key-value pairs in the dictionaries:
    constants['key_name'] or state['key_name'] where key_name is the parameter you want,
    like 'mass' or 'velocity'.
"""

# Helper functions

class LowThrustException(Exception):
    pass


def calculate_charge(current_charge:float, dt:float, current_pull) -> float:
    """
    Calculates the new battery charge given usage at speed vel_mph and the data
    interpolated and passed through batt_interp. Inputs the old charge, the
    velocity in mph, the timestep in seconds, and the batt_interp function.
    Returns the new charge in milliamp-hours.
    """
    charge_used = current_pull * (dt / 3600) * 1000 / PROPULSIVE_EFFICIENCY # Convert to mAh of charge used
    new_charge = current_charge - charge_used
    return new_charge  # Ensure change in charge isn't negative, also in mAh

def get_banner_drag(length, velocity):
    """Estimate banner drag based on data."""
    CD_banner = -1.43E-04 * velocity + 0.0633
    width = length / 5
    rho = 0.002378
    drag = 0.5 * CD_banner * rho * velocity**2 * (length * width)
    return drag


# FLIGHT LEG CALCULATORS

def climb(state:dict, constants:dict):
    """
    Function that calculates the climb of the plane, occurs only once per simulation,
    at takeoff.
    """

    # Convert constants for easier use:
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    dt = constants['dt']
    alt = constants['altitude']
    theta = np.radians(constants['theta'])  # Convert to radians
    cell_count = constants['battery_cells']
    prop_diam = constants['propeller_diameter']
    prop_pitch = constants['propeller_pitch']
    motor_kv = constants['motor_kv']

    i = state['i']

    while state['position'][i,1] <= alt and state['time'][-1] < 300:

        v = (state['velocity'][i]) # Current speed in ft/s

        if v < constants['stall_speed']:
            raise LowThrustException

        q = 0.5 * rho * v**2 # Dynamic pressure in slugs/ft/s^2

        # Coeff calculation
        CL_Climb = (W * np.cos(theta)) / (q * S) # CL needed to climb at angle theta
        CD_i = (CL_Climb**2) / (np.pi * AR * E) # Induced drag coefficient

        # Drag component buildup
        D_i = CD_i * q * constants['S'] # Induced drag
        D_p = constants['CD_p'] * q * constants['S'] # Parasite drag
        D_ht = constants['CD_t'] * q * constants['S_h'] # Horizontal tail drag
        drag = D_i + D_p + D_ht # Total drag
        CD_Climb = drag / (q * S) # Total drag coefficient

        thrust, current_amps = apc_data.get_propeller_performance(
                    diameter=prop_diam,
                    pitch=prop_pitch,
                    motor_kv=motor_kv,
                    battery_cell_count=cell_count,
                    airspeed_mph=v*0.681818
                )

        # Calculate the acceleration, velocity, positions
        new_acceleration = ( (thrust) - (drag) - (W*np.sin(theta)) ) / m # Acceleration in ft/s^2
        new_velocity = v + new_acceleration*dt
        new_position = np.add(state['position'][i], [v*dt*np.cos(theta), v*dt*np.sin(theta)])

        # Append to all fields in the state dict:
        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],dt,current_amps))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],0) # No turning (hopefully)
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Climb)
        state['Cd'] = np.append(state['Cd'],CD_Climb)
        state['lift'] = np.append(state['lift'], W * np.cos(theta))
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], new_acceleration * m)
        state['F_lat'] = np.append(state['F_lat'],0)
        i += 1
        state['i'] = i

def straight(state:dict, constants:dict, distance_needed):
    """
    The function to calculate the straightaways, uses the assumption of constant altitude.
    """

    # Convert constants for easier use:
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    dt = constants['dt']
    CD_p = constants['CD_p']
    AR = constants['AR']
    E = constants['E']
    CD_t = constants['CD_t']
    S_h = constants['S_h']
    cell_count = constants['battery_cells']
    prop_diam = constants['propeller_diameter']
    prop_pitch = constants['propeller_pitch']
    motor_kv = constants['motor_kv']
    banner_length = constants['banner_length']

    i = state['i']

    distance_traveled = 0

    while distance_needed >= distance_traveled:

        v = state['velocity'][i] # Current speed in ft/s

        if v < constants['stall_speed']:
            raise LowThrustException

        q = 0.5 * rho * v**2 # Dynamic pressure in slugs/ft/s^2

        CL_Straight = W / q / S # CL needed to stay level altitude
        CD_i = (CL_Straight**2) / (np.pi * AR * E) # Induced drag coefficient
        D_banner = get_banner_drag(banner_length, v) if banner_length > 0 else 0 # banner drag
        D_i = CD_i * q * S # Induced drag
        D_p = CD_p * q * S # Parasite drag
        D_ht = CD_t * q * S_h # Horizontal tail drag
        drag = D_i + D_p + D_ht + D_banner # Total drag
        CD_Straight = drag / (q * S) # Total drag coefficient

        thrust, current_amps = apc_data.get_propeller_performance(
                    diameter=prop_diam,
                    pitch=prop_pitch,
                    motor_kv=motor_kv,
                    battery_cell_count=cell_count,
                    airspeed_mph=v*0.681818
                )

        # Calculate the acceleration, velocity, positions
        new_acceleration = ( (thrust) - (drag) ) / m # Acceleration in ft/s^2
        new_velocity = v + new_acceleration*dt
        new_position = np.add(state['position'][i], [v*dt, 0.0])

        # Append to all fields in the state dict:
        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],dt,current_amps))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],0) # No turning (hopefully)
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Straight)
        state['Cd'] = np.append(state['Cd'],CD_Straight)
        state['lift'] = np.append(state['lift'], W)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'],new_acceleration*m)
        state['F_lat'] = np.append(state['F_lat'],0)
        distance_traveled = distance_traveled + (v*dt)
        i += 1
        state['i'] = i

def turn(state:dict, constants:dict, turn_needed):
    """
    Calculate through a turn with constant altitude
    """

    # Convert constants for easier use:
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    dt = constants['dt']
    CL_Turn = constants['CL_turn']
    AR = constants['AR']
    E = constants['E']
    CD_t = constants['CD_t']
    CD_p = constants['CD_p']
    S_h = constants['S_h']
    cell_count = constants['battery_cells']
    prop_diam = constants['propeller_diameter']
    prop_pitch = constants['propeller_pitch']
    motor_kv = constants['motor_kv']
    banner_length = constants['banner_length']

    i = state['i']

    turn_traveled = 0

    while turn_needed >= turn_traveled:

        v = state['velocity'][i] # Current speed in ft/s

        if v < constants['stall_speed']:
            raise LowThrustException

        q = 0.5 * rho * v**2 # Dynamic pressure in slugs/ft/s^2

        CD_i = (CL_Turn**2) / (np.pi * AR * E) # Induced drag coefficient
        D_banner = get_banner_drag(banner_length, v) if banner_length > 0 else 0 # banner drag
        D_i = CD_i * q * S # Induced drag
        D_p = CD_p * q * S # Parasite drag
        D_ht = CD_t * q * S_h # Horizontal tail drag
        drag = D_i + D_p + D_ht + D_banner # Total drag
        CD_Turn = drag / (q * S) # Total drag coefficient
        lift = CL_Turn * q * S  # Lift in lbs

        if lift > W:
            F_lat = np.sqrt(lift**2 - W**2)  # Lateral force in lbs
            omega = F_lat / (m * v)
        else:
            F_lat = 0
            omega = 0

        thrust, current_amps = apc_data.get_propeller_performance(
                    diameter=prop_diam,
                    pitch=prop_pitch,
                    motor_kv=motor_kv,
                    battery_cell_count=cell_count,
                    airspeed_mph=v*0.681818
                )

        # Calculate the acceleration, velocity, positions
        new_acceleration = ( (thrust) - (drag) ) / m # Acceleration in ft/s^2
        new_velocity = v + new_acceleration*dt
        new_position = np.add(state['position'][i], [v*dt, 0.0])
        turn_traveled = turn_traveled + np.degrees(omega*dt)  # Track turn in degrees

        # Append to all fields in the state dict:
        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],dt,current_amps))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],np.degrees(omega*dt)) # Now turning, convert to degrees
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Turn)
        state['Cd'] = np.append(state['Cd'],CD_Turn)
        state['lift'] = np.append(state['lift'], lift)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'],thrust-drag)
        state['F_lat'] = np.append(state['F_lat'],F_lat)
        i += 1
        state['i'] = i

def calculate_m2_reward(lap_count: int, battery_energy: int, cargo_units: int = 1) -> float:
    """
    Mission 2 financial reward.
    Income: (ducks x (6 + 2xlaps)) + (pucks x (10 + 8xlaps))
    Cost: laps x (10 + (0.5 x ducks) + (2 x pucks)) x EF
    where EF = battery_energy / 100
    """
    ducks = cargo_units * 3
    pucks = cargo_units
    income = (ducks * (6 + (2 * lap_count))) + (pucks * (10 + (8 * lap_count)))
    EF = battery_energy / 100.0
    cost = lap_count * (10 + (0.5 * ducks) + (2 * pucks)) * EF
    return income - cost if lap_count > 0 else 0

def calculate_m3_reward(lap_count: int, banner_length: int, wing_span: float = 5.0) -> float:
    """
    Mission 3 financial reward.
    Score = laps * banner length / RAC
    """
    RAC = 0.05 * (wing_span/12) + 0.75
    return lap_count * 12 * banner_length / RAC if lap_count > 0 else 0

def plot_battery_results(results: Dict[str, list], output_file: str = 'battery_optimization.png'):
    """
    Plot battery energy vs M2 score with secondary axis for lap count.

    Args:
        results: Dictionary with battery_energies, lap_counts, scores
        output_file: Output filename for plot
    """
    battery_energies = results['battery_energies']
    lap_counts = results['lap_counts']
    scores = results['scores']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: M2 Score
    color1 = 'tab:blue'
    ax1.set_xlabel('Battery Energy (Wh)', fontsize=12)
    ax1.set_ylabel('M2 Score', color=color1, fontsize=12)
    ax1.plot(battery_energies, scores, 'o-', color=color1, linewidth=2, markersize=6, label='M2 Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Lap Count
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Lap Count', color=color2, fontsize=12)
    ax2.plot(battery_energies, lap_counts, 's--', color=color2, linewidth=2, markersize=5, label='Lap Count')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Find and mark optimal
    optimal_idx = np.argmax(scores)
    optimal_battery = battery_energies[optimal_idx]
    optimal_score = scores[optimal_idx]
    optimal_laps = lap_counts[optimal_idx]

    ax1.axvline(x=optimal_battery, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax1.scatter([optimal_battery], [optimal_score], color='green', s=150, zorder=5, marker='*')

    # Title with optimal info
    plt.title(f'Battery Size Optimization for M2\nOptimal: {optimal_battery:.1f} Wh ({optimal_laps} laps, score={optimal_score:.1f})',
              fontsize=14)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()

    plt.show()

def plot_banner_results(results: Dict[str, list], output_file: str = 'banner_optimization.png'):
    """
    Plot banner length vs M# score with secondary axis for lap count.

    Args:
        results: Dictionary with banner_lengths, lap_counts, scores
        output_file: Output filename for plot
    """
    banner_lengths = results['banner_lengths']
    lap_counts = results['lap_counts']
    scores = results['scores']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: M2 Score
    color1 = 'tab:blue'
    ax1.set_xlabel('Banner Lengths (ft)', fontsize=12)
    ax1.set_ylabel('M3 Score', color=color1, fontsize=12)
    ax1.plot(banner_lengths, scores, 'o-', color=color1, linewidth=2, markersize=6, label='M3 Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Lap Count
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Lap Count', color=color2, fontsize=12)
    ax2.plot(banner_lengths, lap_counts, 's--', color=color2, linewidth=2, markersize=5, label='Lap Count')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Find and mark optimal
    optimal_idx = np.argmax(scores)
    optimal_banner = banner_lengths[optimal_idx]
    optimal_score = scores[optimal_idx]
    optimal_laps = lap_counts[optimal_idx]

    ax1.axvline(x=optimal_banner, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax1.scatter([optimal_banner], [optimal_score], color='green', s=150, zorder=5, marker='*')

    # Title with optimal info
    plt.title(f'Banner Length Optimization for M3\nOptimal: {optimal_banner:.1f} feet ({optimal_laps} laps, score={optimal_score:.1f})',
              fontsize=14)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    plt.show()

# Running the file:
def main():
    # First get all the constant parameters into a structured dictionary
    stall_speed = np.sqrt((2 * WEIGHT) / (RHO * WING_AREA * CL_STALL))
    constants = {
        'altitude': LAP_ALTITUDE,
        'theta': CLIMB_ANGLE,
        'S': WING_AREA,
        'W': WEIGHT,
        'AR': AR,
        'E': E,
        'S_h': HORIZ_TAIL_AREA,
        'CD_t': CD_TAIL,
        'CD_p': CD_PARA,
        'CL_turn': CL_TURN,
        'CL_stall': CL_STALL,
        'stall_speed': stall_speed,
        'm': WEIGHT/GRAVITY,
        'g': GRAVITY,
        'rho': RHO,
        'dt': DT,
        'battery_capacity': BATTERY_CAPACITY,
        'battery_cells': BATTERY_CELLS,
        'lap_count': LAP_COUNT,
        'propeller_diameter': PROPELLER_DIAMETER, # inches
        'propeller_pitch': PROPELLER_PITCH, # inches
        'motor_kv': MOTOR_KV, # RPM/Volt
        'banner_length': BANNER_LENGTH, # ft
    }

    # Set up initial state as well as the state directory
    state = {
        'velocity': np.array([stall_speed * 1.05]), #ft/s
        'position': np.array([[0.0,0.0]]), #ft
        'acceleration': np.array([0.0]), #ft/s^2
        'battery_charge': np.array([constants['battery_capacity']]), #mAh
        'time': np.array([0.0]), #seconds
        'turn_angle': np.array([0.0]), #degrees
        'thrust': np.array([0.0]), #lbs
        'Cl': np.array([0.0]),
        'Cd': np.array([0.0]),
        'lift': np.array([0.0]), #lbs
        'drag': np.array([0.0]), #lbs
        'F_long': np.array([0.0]), #lbs
        'F_lat': np.array([0.0]), #lbs
        'i': 0
    }
    print('State initial conditions completed successfully.')

    # Fourth, model the track that will be run and execute the functions

    # Takeoff and getting to altitude:
    print('Takeoff complete')
    try:
        climb(state,constants)
    except LowThrustException:
        print('Stall detected during climb — insufficient thrust.')
        return
    print('Climb complete')
    # Complete lap simulation
    lap_counter = 0

    try:
        if constants['lap_count'] > 0:
            for _ in range(constants['lap_count']):
                straight(state,constants,500)
                turn(state,constants,180)
                straight(state,constants,500)
                print(state['time'][-1])
                turn(state,constants,360)
                print(state['time'][-1])
                straight(state,constants,500)
                turn(state,constants,180)
                straight(state,constants,500)
                lap_counter = lap_counter + 1
                print(f'Lap {lap_counter} complete')
        elif constants['lap_count'] == 0:
            while state['battery_charge'][-1] > constants['battery_capacity']*0.3 and state['time'][-1] < 300:
                straight(state,constants,500)
                turn(state,constants,180)
                straight(state,constants,500)
                turn(state,constants,360)
                straight(state,constants,500)
                turn(state,constants,180)
                straight(state,constants,500)
                lap_counter = lap_counter + 1
                print(f'Lap {lap_counter} complete')
    except LowThrustException:
        print(f'Stall detected during lap {lap_counter + 1} — stopping simulation.')
    # Comprehensive plotting of all flight parameters
    print(f"Battery starting capacity: {constants['battery_capacity']} mAh")
    print(f"Remaining battery charge: {state['battery_charge'][-1]:.1f} mAh after {lap_counter} laps.")
    print(f"Time remaining: {300 - state['time'][-1]:.1f} seconds")
    print(f"Time elapsed: {state['time'][-1]:.1f} seconds")
    print(f"Maximum velocity achieved: {np.max(state['velocity'])*0.681818:.1f} mph")

    turn_angle_rate = []
    for index in range(len(state['turn_angle'])-1):
        if state['turn_angle'][index] != 0 or state['turn_angle'][index+1] != 0:
            turn_angle_rate = np.append(turn_angle_rate, (state['turn_angle'][index]-state['turn_angle'][index+1])/(state['time'][index+1]-state['time'][index]))

    turn_angle_rate_average = np.mean(np.abs(turn_angle_rate))
    turn_angle_max = np.max(np.abs(turn_angle_rate))
    print(f"Average turn rate during turns: {turn_angle_rate_average:.2f} deg/s and Max turn rate: {turn_angle_max:.2f} deg/s")

    fig = plt.figure(figsize=[15, 8])
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f"Aircraft Lap Simulation Results", fontsize=18)

    # Row 1: Velocities, Battery, Position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(state['time'], state['velocity'], 'b-', linewidth=2)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Velocity (ft/s)")
    ax1.set_title("Velocity vs Time"); ax1.grid(True); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(state['time'], state['battery_charge'], 'g-', linewidth=2)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Battery Charge (mAh)")
    ax2.set_title("Battery Charge vs Time"); ax2.grid(True)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(state['time'], state['position'][:,0], 'b-', linewidth=2, label='Altitude')
    ax3.plot(state['time'], state['position'][:,1], 'r-', linewidth=2, label='Distance Traveled')
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Position (ft)")
    ax3.set_title("Position vs Time"); ax3.grid(True); ax3.legend()

    # Row 2: Accelerations, Thrust, Turn Angle
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(state['time'], state['acceleration'], 'b-', linewidth=2)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Acceleration (ft/s²)")
    ax4.set_title("Acceleration vs Time"); ax4.grid(True); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(state['time'], state['thrust'], 'orange', linewidth=2)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Thrust (lbs)")
    ax5.set_title("Thrust vs Time"); ax5.grid(True)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(state['time'], state['turn_angle'], 'm-', linewidth=2)
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Turn Angle (deg/step)")
    ax6.set_title("Turn Angle vs Time"); ax6.grid(True)

    # Row 3: Aerodynamic coefficients with L/D, Combined Forces, Longitudinal/Lateral Forces
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(state['time'], state['Cl'], 'c-', linewidth=2, label='Cl')
    ax7.plot(state['time'], state['Cd'], 'y-', linewidth=2, label='Cd')
    # Calculate L/D ratio, avoiding division by zero
    # ld_ratio = np.divide(state['Cl'], state['Cd'], out=np.zeros_like(state['Cl']), where=state['Cd']!=0)
    # ax7.plot(state['time'], ld_ratio, 'purple', linewidth=2, label='L/D')
    ax7.set_xlabel("Time (s)"); ax7.set_ylabel("Coefficients / L/D Ratio")
    ax7.set_title("Cl, Cd, and L/D vs Time"); ax7.grid(True); ax7.legend()

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(state['time'], state['lift'], 'g-', linewidth=2, label='Lift')
    ax8.plot(state['time'], state['drag'], 'brown', linewidth=2, label='Drag')
    ax8.set_xlabel("Time (s)"); ax8.set_ylabel("Force (lbs)")
    ax8.set_title("Lift and Drag vs Time"); ax8.grid(True); ax8.legend()

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(state['time'], state['F_long'], 'k-', linewidth=2, label='Longitudinal')
    ax9.plot(state['time'], state['F_lat'], 'gray', linewidth=2, label='Lateral')
    ax9.set_xlabel("Time (s)"); ax9.set_ylabel("Force (lbs)")
    ax9.set_title("Longitudinal and Lateral Forces vs Time"); ax9.grid(True); ax9.legend()

    plt.tight_layout()
    plt.show()

    print(f"Simulation completed with {len(state['time'])} data points over {state['time'][-1]:.1f} seconds")

def vary_battery(battery_list, mission: int = 2):
    # First get all the constant parameters into a structured dictionary
    laps = np.zeros(len(battery_list))
    score = np.zeros(len(battery_list))

    if mission == 2:
        banner = 0
    elif mission == 3:
        banner = BANNER_LENGTH

    for idx,battery in enumerate(battery_list):
        stall_speed = np.sqrt((2 * WEIGHT) / (RHO * WING_AREA * CL_STALL))
        constants = {
            'altitude': LAP_ALTITUDE,
            'theta': CLIMB_ANGLE,
            'S': WING_AREA,
            'W': WEIGHT,
            'AR': AR,
            'E': E,
            'S_h': HORIZ_TAIL_AREA,
            'CD_t': CD_TAIL,
            'CD_p': CD_PARA,
            'CL_turn': CL_TURN,
            'CL_stall': CL_STALL,
            'stall_speed': stall_speed,
            'm': WEIGHT/GRAVITY,
            'g': GRAVITY,
            'rho': RHO,
            'dt': DT,
            'battery_capacity': battery,
            'battery_cells': BATTERY_CELLS,
            'lap_count': LAP_COUNT,
            'propeller_diameter': PROPELLER_DIAMETER, # inches
            'propeller_pitch': PROPELLER_PITCH, # inches
            'motor_kv': MOTOR_KV, # RPM/Volt
            'banner_length': banner, # ft
        }

        # Set up initial state as well as the state directory
        state = {
            'velocity': np.array([stall_speed * 1.05]), #ft/s
            'position': np.array([[0.0,0.0]]), #ft
            'acceleration': np.array([0.0]), #ft/s^2
            'battery_charge': np.array([constants['battery_capacity']]), #mAh
            'time': np.array([0.0]), #seconds
            'turn_angle': np.array([0.0]), #degrees
            'thrust': np.array([0.0]), #lbs
            'Cl': np.array([0.0]),
            'Cd': np.array([0.0]),
            'lift': np.array([0.0]), #lbs
            'drag': np.array([0.0]), #lbs
            'F_long': np.array([0.0]), #lbs
            'F_lat': np.array([0.0]), #lbs
            'i': 0
        }

        # Fourth, model the track that will be run and execute the functions
        
        # Takeoff and getting to altitude:
        lap_counter = 0
        try:
            climb(state,constants)
            # Complete lap simulation

            if constants['lap_count'] > 0:
                for lap in range(constants['lap_count']):
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    turn(state,constants,360)
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    lap_counter = lap_counter + 1
                    print(f'Lap {lap_counter} complete')
            elif constants['lap_count'] == 0:
                while state['battery_charge'][-1] > constants['battery_capacity']*0.3 and state['time'][-1] < 300:
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    turn(state,constants,360)
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    lap_counter = lap_counter + 1
                    print(f'Lap {lap_counter} complete')
        except LowThrustException:
            print(f'Stall detected — stopping simulation at lap {lap_counter}.')
        # Comprehensive plotting of all flight parameters
        print(f"Battery starting capacity: {constants['battery_capacity']} mAh")
        print(f"Remaining battery charge: {state['battery_charge'][-1]:.1f} mAh after {lap_counter} laps.")
        print(f"Time remaining: {300 - state['time'][-1]:.1f} seconds")
        print(f"Time elapsed: {state['time'][-1]:.1f} seconds")

        turn_angle_rate = []
        for index in range(len(state['turn_angle'])-1):
            if state['turn_angle'][index] != 0 or state['turn_angle'][index+1] != 0:
                turn_angle_rate = np.append(turn_angle_rate, (state['turn_angle'][index]-state['turn_angle'][index+1])/(state['time'][index+1]-state['time'][index]))

        turn_angle_rate_average = np.mean(np.abs(turn_angle_rate))
        turn_angle_max = np.max(np.abs(turn_angle_rate))
        print(f"Average turn rate during turns: {turn_angle_rate_average:.2f} deg/s and Max turn rate: {turn_angle_max:.2f} deg/s")

        print(f"Simulation completed with {len(state['time'])} data points over {state['time'][-1]:.1f} seconds")

        laps[idx] = lap_counter
        battery_energy = battery * constants['battery_cells'] * 3.7 / 1000
        score[idx] = calculate_m2_reward(lap_counter, battery_energy)

    results = {'battery_energies': battery_list * constants['battery_cells'] * 3.7 / 1000, 'lap_counts': laps, 'scores': score}
    plot_battery_results(results)

def vary_banner(banner_list):
    # First get all the constant parameters into a structured dictionary
    laps = np.zeros(len(banner_list))
    score = np.zeros(len(banner_list))

    for idx,banner in enumerate(banner_list):
        stall_speed = np.sqrt((2 * WEIGHT) / (RHO * WING_AREA * CL_STALL))
        constants = {
            'altitude': LAP_ALTITUDE,
            'theta': CLIMB_ANGLE,
            'S': WING_AREA,
            'W': WEIGHT,
            'AR': AR,
            'E': E,
            'S_h': HORIZ_TAIL_AREA,
            'CD_t': CD_TAIL,
            'CD_p': CD_PARA,
            'CL_turn': CL_TURN,
            'CL_stall': CL_STALL,
            'stall_speed': stall_speed,
            'm': WEIGHT/GRAVITY,
            'g': GRAVITY,
            'rho': RHO,
            'dt': DT,
            'battery_capacity': BATTERY_CAPACITY,
            'battery_cells': BATTERY_CELLS,
            'lap_count': LAP_COUNT,
            'propeller_diameter': PROPELLER_DIAMETER, # inches
            'propeller_pitch': PROPELLER_PITCH, # inches
            'motor_kv': MOTOR_KV, # RPM/Volt
            'banner_length': banner, # ft
        }

        # Set up initial state as well as the state directory
        state = {
            'velocity': np.array([stall_speed * 1.05]), #ft/s
            'position': np.array([[0.0,0.0]]), #ft
            'acceleration': np.array([0.0]), #ft/s^2
            'battery_charge': np.array([constants['battery_capacity']]), #mAh
            'time': np.array([0.0]), #seconds
            'turn_angle': np.array([0.0]), #degrees
            'thrust': np.array([0.0]), #lbs
            'Cl': np.array([0.0]),
            'Cd': np.array([0.0]),
            'lift': np.array([0.0]), #lbs
            'drag': np.array([0.0]), #lbs
            'F_long': np.array([0.0]), #lbs
            'F_lat': np.array([0.0]), #lbs
            'i': 0
        }
        print('State initial conditions completed successfully.')

        # Fourth, model the track that will be run and execute the functions
        
        # Takeoff and getting to altitude:
        print('Takeoff complete')
        lap_counter = 0
        try:
            climb(state,constants)
            print('Climb complete')
            # Complete lap simulation

            if constants['lap_count'] > 0:
                for lap in range(constants['lap_count']):
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    turn(state,constants,360)
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    lap_counter = lap_counter + 1
                    print(f'Lap {lap_counter} complete')
            elif constants['lap_count'] == 0:
                while state['battery_charge'][-1] > constants['battery_capacity']*0.3 and state['time'][-1] < 300:
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    turn(state,constants,360)
                    straight(state,constants,500)
                    turn(state,constants,180)
                    straight(state,constants,500)
                    lap_counter = lap_counter + 1
                    print(f'Lap {lap_counter} complete')
        except LowThrustException:
            print(f'Stall detected — stopping simulation at lap {lap_counter}.')
        # Comprehensive plotting of all flight parameters
        print(f"Banner length: {banner} ft")
        print(f"Battery starting capacity: {constants['battery_capacity']} mAh")
        print(f"Remaining battery charge: {state['battery_charge'][-1]:.1f} mAh after {lap_counter} laps.")
        print(f"Time remaining: {300 - state['time'][-1]:.1f} seconds")
        print(f"Time elapsed: {state['time'][-1]:.1f} seconds")

        turn_angle_rate = []
        for index in range(len(state['turn_angle'])-1):
            if state['turn_angle'][index] != 0 or state['turn_angle'][index+1] != 0:
                turn_angle_rate = np.append(turn_angle_rate, (state['turn_angle'][index]-state['turn_angle'][index+1])/(state['time'][index+1]-state['time'][index]))

        turn_angle_rate_average = np.mean(np.abs(turn_angle_rate))
        turn_angle_max = np.max(np.abs(turn_angle_rate))
        print(f"Average turn rate during turns: {turn_angle_rate_average:.2f} deg/s and Max turn rate: {turn_angle_max:.2f} deg/s")

        print(f"Simulation completed with {len(state['time'])} data points over {state['time'][-1]:.1f} seconds")

        laps[idx] = lap_counter
        score[idx] = calculate_m3_reward(lap_counter, banner, np.sqrt(constants['AR']*constants['S']))

    results = {'banner_lengths': banner_list, 'lap_counts': laps, 'scores': score}
    plot_banner_results(results)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Parameterized lap simulator.'
    )
    parser.add_argument('--banner', nargs=3, type=int,
                       help='Enter minimum, maximum, and number of banner lengths in feet.')
    parser.add_argument('--battery', nargs=3, type=int,
                       help='Enter minimum, maximum, and number of battery capacities in mAh.')
    parser.add_argument('--mission', nargs=1, type=int,
                       help='Enter mission (only for battery optimization, 2 or 3)')
    
    args = parser.parse_args()

    if args.banner:
        min_banner, max_banner, num_banners = args.banner
        banner_lengths = np.linspace(min_banner, max_banner, num_banners)
        vary_banner(banner_lengths)
    elif args.battery:
        min_battery, max_battery, num_batteries = args.battery
        mission = args.mission[0] if args.mission else 2
        battery_capacities = np.linspace(min_battery, max_battery, num_batteries)
        vary_battery(battery_capacities, mission)
    else:
        main()
