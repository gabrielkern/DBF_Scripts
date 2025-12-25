# OHHHHHHH YEAHHHHHHHHHHH
# Run with: python VECTOOOOORRRRRRRRR.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline,interp1d

# SET STATIC PARAMETERS
LAP_ALTITUDE = 200 #ft
CLIMB_ANGLE = 20 #deg
WING_AREA = 4.75 #ft^2
AR = 5.26 #unitless
E = 0.9 #unitless
HORIZ_TAIL_AREA = 0.776 #ft^2
CD_TAIL = 0.05 #unitless
CD_PARA = 0.015 #unitless
CL_TURN = 1.2 #unitless
WEIGHT = 10 #lbs
GRAVITY = 32.174 #ft/s^2
RHO = 0.0023769 #slugs/ft^3 at sea level
DT = 0.05 #seconds
BATTERY_CAPACITY = 3300 #mAh
BATTERY_CELLS = 6 #number of cells in series
LAP_COUNT = 3 # Number of laps that the program should run

MOTOCALC_FILEPATH = 'MC_4025_520kV_17x8e.csv'


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

# IMPORT FUNCTIONS

def imported_data(csv_filepath:str) -> pd.DataFrame:
    """
    Function that pulls data from a MotoCalc csv file and returns thrust
    as a function of airspeed
    """

    df = pd.read_csv(
        csv_filepath,
        skiprows=10,
        skipfooter=4,
        engine='python',
        encoding='latin-1'
    )

    # Define the column names in the correct order
    column_names = [
        'airspeed_mph', 'drag_oz', 'lift_oz', 'batt_amps', 'motor_amps',
        'motor_volts', 'input_watts', 'loss_watts', 'mgbout_watts',
        'motgb_eff_pct', 'shaft_eff_pct', 'prop_rpm', 'thrust_oz',
        'prop_speed_mph', 'prop_eff_pct', 'total_eff_pct', 'time_min_sec'
    ]

    # Assign the new names to the DataFrame's columns
    df.columns = column_names

    # Replace infinite values with NaN (Not a Number), which pandas handles well
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop any row that contains at least one NaN value
    df.dropna(inplace=True)

    return df

def create_batt_interpolator(prop_data: pd.DataFrame):
    """
    Creates an interpolation helper function for battery amps vs. airspeed.
    The purpose of this is so that interpolation can be done once, outside
    of individual calls to the main function.
    """
    sorted_data = prop_data.sort_values(by='airspeed_mph')
    return make_interp_spline(
        sorted_data['airspeed_mph'],
        sorted_data['batt_amps']
    )

def create_thrust_interpolator(prop_data: pd.DataFrame):
    """
    Creates an interpolation helper function for thrust vs. airspeed.
    The purpose of this is so that interpolation can be done once, outside
    """
    sorted_data = prop_data.sort_values(by='airspeed_mph')
    return make_interp_spline(
        sorted_data['airspeed_mph'],
        sorted_data['thrust_oz']
    )

def calculate_charge(current_charge:float,vel_fps:float, dt:float, batt_interp) -> float:
    """
    Calculates the new battery charge given usage at speed vel_mph and the data
    interpolated and passed through batt_interp. Inputs the old charge, the
    velocity in mph, the timestep in seconds, and the batt_interp function.
    Returns the new charge in milliamp-hours.
    """
    vel_mph = vel_fps * 0.681818 # convert ft/s to mph
    amps = batt_interp(vel_mph)
    charge_used = amps * (dt / 3600) * 1000  # Convert to mAh of charge used
    new_charge = current_charge - charge_used
    return new_charge  # Ensure change in charge isn't negative, also in mAh

def calculate_thrust(vel_fps:float, thrust_interp) -> float:
    """
    Calculates the thrust at a given velocity using the thrust_interp function.
    Inputs the velocity in mph and the thrust_interp function.
    Returns the thrust in ounces.
    """
    vel_mph = vel_fps * 0.681818 # convert ft/s to mph
    return ((thrust_interp(vel_mph)) * 0.0625) # convert oz to lbs


# FLIGHT LEG CALCULATORS

def climb(state:dict, constants:dict, batt_interp, thrust_interp):
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

    i = state['i']

    while state['position'][i,1] <= alt:

        v = (state['velocity'][i]) # Current speed in ft/s

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

        thrust = calculate_thrust(v, thrust_interp) # Thrust in lbs

        # Calculate the acceleration, velocity, positions
        new_acceleration = ( (thrust) - (drag) - (W*np.sin(theta)) ) / m # Acceleration in ft/s^2
        new_velocity = v + new_acceleration*dt
        new_position = np.add(state['position'][i], [v*dt*np.cos(theta), v*dt*np.sin(theta)])

        # Append to all fields in the state dict:
        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],v,dt,batt_interp))
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

def straight(state:dict, constants:dict, distance_needed, batt_interp, thrust_interp):
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

    i = state['i']

    distance_traveled = 0

    while distance_needed >= distance_traveled:

        v = state['velocity'][i] # Current speed in ft/s

        q = 0.5 * rho * v**2 # Dynamic pressure in slugs/ft/s^2

        CL_Straight = W / q / S # CL needed to stay level altitude
        CD_i = (CL_Straight**2) / (np.pi * AR * E) # Induced drag coefficient
        D_i = CD_i * q * S # Induced drag
        D_p = CD_p * q * S # Parasite drag
        D_ht = CD_t * q * S_h # Horizontal tail drag
        drag = D_i + D_p + D_ht # Total drag
        CD_Straight = drag / (q * S) # Total drag coefficient
        
        thrust = calculate_thrust(v, thrust_interp) # Thrust in lbs

        # Calculate the acceleration, velocity, positions
        new_acceleration = ( (thrust) - (drag) ) / m # Acceleration in ft/s^2
        new_velocity = v + new_acceleration*dt
        new_position = np.add(state['position'][i], [v*dt, 0.0])

        # Append to all fields in the state dict:
        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],v,dt,batt_interp))
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

def turn(state:dict, constants:dict, turn_needed, batt_interp, thrust_interp):
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

    i = state['i']

    turn_traveled = 0

    while turn_needed >= turn_traveled:

        v = state['velocity'][i] # Current speed in ft/s

        q = 0.5 * rho * v**2 # Dynamic pressure in slugs/ft/s^2

        CD_i = (CL_Turn**2) / (np.pi * AR * E) # Induced drag coefficient
        D_i = CD_i * q * S # Induced drag
        D_p = CD_p * q * S # Parasite drag
        D_ht = CD_t * q * S_h # Horizontal tail drag
        drag = D_i + D_p + D_ht # Total drag
        CD_Turn = drag / (q * S) # Total drag coefficient
        lift = CL_Turn * q * S  # Lift in lbs

        F_lat = np.sqrt(lift**2 - W**2)  # Lateral force in lbs
        omega = F_lat / (m * v)

        thrust = calculate_thrust(v, thrust_interp) # Find thrust

        # Calculate the acceleration, velocity, positions
        new_acceleration = ( (thrust) - (drag) ) / m # Acceleration in ft/s^2
        new_velocity = v + new_acceleration*dt
        new_position = np.add(state['position'][i], [v*dt, 0.0])
        turn_traveled = turn_traveled + np.degrees(omega*dt)  # Track turn in degrees

        # Append to all fields in the state dict:
        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],v,dt,batt_interp))
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


# Running the file:
if __name__ == '__main__':

    # First get all the constant parameters into a structured dictionary
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
        'm': WEIGHT/GRAVITY,
        'g': GRAVITY,
        'rho': RHO,
        'dt': DT, 
        'battery_capacity': BATTERY_CAPACITY,
        'battery_cells': BATTERY_CELLS,
        'lap_count': LAP_COUNT
    }

    # Second create the interpolators with the imported MotoCalc data to be parsed
    imported_dataframe = imported_data(MOTOCALC_FILEPATH)
    batt_interp = create_batt_interpolator(imported_dataframe)
    thrust_interp = create_thrust_interpolator(imported_dataframe)
    print(f"Interpolators created successfully.")

    # Third set up initial state as well as the state directory
    state = {
        'velocity': np.array([30.0]), #ft/s
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
    climb(state,constants,batt_interp,thrust_interp)
    print('Climb complete')
    # Complete lap simulation
    lap_counter = 0

    for lap in range(constants['lap_count']):
        straight(state,constants,500,batt_interp,thrust_interp)
        turn(state,constants,180,batt_interp,thrust_interp)
        straight(state,constants,500,batt_interp,thrust_interp)
        turn(state,constants,360,batt_interp,thrust_interp)
        straight(state,constants,500,batt_interp,thrust_interp)
        turn(state,constants,180,batt_interp,thrust_interp)
        straight(state,constants,500,batt_interp,thrust_interp)
        lap_counter = lap_counter + 1
        print(f'Lap {lap_counter} complete')
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
    ld_ratio = np.divide(state['Cl'], state['Cd'], out=np.zeros_like(state['Cl']), where=state['Cd']!=0)
    ax7.plot(state['time'], ld_ratio, 'purple', linewidth=2, label='L/D')
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