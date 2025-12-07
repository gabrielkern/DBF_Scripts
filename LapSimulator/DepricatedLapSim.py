# OHHHHHHH YEAHHHHHHHHHHH
# Run with: python VECTOOOOORRRRRRRRR.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline,interp1d

# SET STATIC PARAMETERS
LAP_ALTITUDE = 200 #ft
CLIMB_ANGLE = 30 #deg
WING_AREA = 4.44 #ft^2
WEIGHT = 4.5 #lbs
COEFF_FRICTION = 0.04 #unitless
GRAVITY = 32.174 #ft/s^2
RHO = 0.0023769 #slugs/ft^3 at sea level
DT = 0.05 #seconds
BATTERY_CAPACITY = 3300 #mAh
BATTERY_CELLS = 4 #number of cells in series
LAP_COUNT = 3 # Number of laps that the program should run
ANGLE_OF_ATTACK = 3 #deg
AOA_AT_MAX_LIFT = 10 #deg

MOTOCALC_FILEPATH = 'Lark8lb10x6.csv'
XFLR5_FILEPATH = 'Lark45lbfull04m.csv'


"""
    To get thrust, call calculate_thrust(vel_fps, thrust_interp), where
    vel_fps is the velocity in feet per second and thrust_interp is a constant. 
    bring thrust_interp into functions

    To get battery charge, call calculate_charge(current_charge, vel_fps, dt, batt_interp),
    where current_charge is the current battery charge in mAh, vel_fps is the
    velocity in feet per second, dt is the timestep in seconds, and batt_interp
    is a constant. bring battery_interp into functions

    For xflr data you need to run the interp once then bring in coeff_interp into functions
    results = xflr_results(coeff_interp, "Cl", 0.3778)
    this will ouput correspoding other two values of CL, CD, alpha

    To access different constants or states, use the key-value pairs in the dictionaries:
    constants['key_name'] or state['key_name'] where key_name is the parameter you want,
    like 'mass' or 'velocity'.
"""

# These are the functions that do the initial data processing

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
    return max(new_charge, 0)  # Ensure change in charge isn't negative, also in mAh

def calculate_thrust(vel_fps:float, thrust_interp) -> float:
    """
    Calculates the thrust at a given velocity using the thrust_interp function.
    Inputs the velocity in mph and the thrust_interp function.
    Returns the thrust in ounces.
    """
    vel_mph = vel_fps * 0.681818 # convert ft/s to mph
    return ((thrust_interp(vel_mph)) * 0.0625) # convert oz to lbs

def xflr_interp(filename):
    # --- read the polar -------------------------------------------------
    df = pd.read_csv(filename, skiprows=6, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")

    coeff_interp = {
    'cl_of_alpha': interp1d(df["alpha"], df["CL"], kind='cubic', fill_value='extrapolate'),
    'cd_of_alpha': interp1d(df["alpha"], df["CD"], kind='cubic', fill_value='extrapolate'),
    'cd_of_cl':    interp1d(df["CL"], df["CD"], kind='cubic', fill_value='extrapolate'),
    'alpha_of_cl': interp1d(df["CL"], df["alpha"], kind='cubic', fill_value='extrapolate')
}
    return  coeff_interp

def xflr_results(interpolators,name,value):
    name = name.lower()  # Normalize to lowercase for easier matching

    if name == "cl":
        CD = interpolators['cd_of_cl'](value)
        alpha = interpolators['alpha_of_cl'](value)
        return float(CD), float(alpha)

    elif name == "cd":
        # You need to define these interpolators first
        CL = interpolators['cl_of_cd'](value)
        alpha = interpolators['alpha_of_cd'](value)
        return float(CL), float(alpha)

    elif name == "alpha":
        CL = interpolators['cl_of_alpha'](value)
        CD = interpolators['cd_of_alpha'](value)

        return float(CL), float(CD)

    else:
        raise ValueError("name must be one of: 'CL', 'CD', 'alpha'")

# These are the functions that calculate each leg of the flight

def takeoff(state:dict, constants:dict, batt_interp, thrust_interp, coeff_interp):
    """
    Function that simulates the takeoff sequence, starting from rest and accelerating
    until the zero angle of attack speed is enough such that lift equals weight.
    """

    # Convert constants for easier use:
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    Cf = constants['Cf']
    dt = constants['dt']
    i = state['i']

    CL_Takeoff,CD_Takeoff = xflr_results(coeff_interp, "alpha", 0) # Get CL and CD at 0 deg alpha

    # Calculate the takeoff sequence until the lift is greater than weight
    while state['lift'][i] <= W:

        v = np.linalg.norm(state['velocity'][i]) # Current speed in ft/s

        q = 0.5 * rho * state['velocity'][i,0]**2 # Dynamic pressure in slugs/ft/s^2
        drag = CD_Takeoff * q * S # Drag in lbs
        lift = CL_Takeoff * q * S # Lift in lbs

        f_ground = Cf * (W - lift) # Friction force in lbs

        thrust = calculate_thrust(state['velocity'][i,0], thrust_interp) # Thrust in lbs

        new_acceleration = np.array([(thrust - drag - f_ground) / m,0.0]) # Acceleration in ft/s^2
        new_velocity = np.add(state['velocity'][i], new_acceleration*dt)
        new_position = np.add(state['position'][i], state['velocity'][i]*dt)

        # Append to all fields in the state dict:
        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], new_acceleration))
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],v,dt,batt_interp))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],0) # No turning (hopefully)
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Takeoff)
        state['Cd'] = np.append(state['Cd'],CD_Takeoff)
        state['alpha'] = np.append(state['alpha'],0) # Hold at 0 aoa
        state['lift'] = np.append(state['lift'], lift)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'],thrust-drag-f_ground)
        state['F_lat'] = np.append(state['F_lat'],0)
        i += 1
        state['i'] = i

def climb(state:dict, constants:dict, batt_interp, thrust_interp, coeff_interp):
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
    aoa = constants['cruise_aoa']

    i = state['i']

    # Calculate the lift and drag coefficients at the cruise aoa
    CL_Cruise, CD_Cruise = xflr_results(coeff_interp, "alpha", aoa)

    while state['position'][i,1] <= alt:

        v = np.linalg.norm(state['velocity'][i]) # Current speed in ft/s

        q = 0.5 * rho * state['velocity'][i,0]**2 # Dynamic pressure in slugs/ft/s^2
        drag = CD_Cruise * q * S # Drag in lbs
        lift = CL_Cruise * q * S # Lift in lbs

        thrust = calculate_thrust(state['velocity'][i,0], thrust_interp) # Thrust in lbs

        # Calculate the accelerations acting in each direction
        ax = ( (thrust) - (drag) - (W*np.sin(theta)) ) / m
        ay = ( (lift * np.cos(theta)) + ( thrust * np.sin(theta) ) - (W) ) / m

        new_acceleration = np.array([ax,ay]) # Acceleration in ft/s^2
        new_velocity = np.add(state['velocity'][i], new_acceleration*dt)
        new_position = np.add(state['position'][i], state['velocity'][i]*dt)

        # Append to all fields in the state dict:
        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], new_acceleration))
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],v,dt,batt_interp))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],0) # No turning (hopefully)
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Cruise)
        state['Cd'] = np.append(state['Cd'],CD_Cruise)
        state['alpha'] = np.append(state['alpha'],aoa) # Hold at cruise aoa
        state['lift'] = np.append(state['lift'], lift)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], ax * m)
        state['F_lat'] = np.append(state['F_lat'],0)
        i += 1
        state['i'] = i

def straight(state:dict, constants:dict, distance_needed, batt_interp, thrust_interp, coeff_interp):
    """
    The function to calculate the straightaways, uses the assumption of constant altitude.
    """

    # Convert constants for easier use:
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    dt = constants['dt']

    i = state['i']

    distance_traveled = 0

    while distance_needed >= distance_traveled:

        v = np.linalg.norm(state['velocity'][i]) # Current speed in ft/s

        q = 0.5 * rho * state['velocity'][i,0]**2 # Dynamic pressure in slugs/ft/s^2

        CL_Needed = W / q / S # CL needed to stay level altitude

        CD_Needed,alpha = xflr_results(coeff_interp, "cl", CL_Needed) # Get CL and CD at 0 deg alpha

        drag = CD_Needed * q * S # Calculate drag on the plane
        thrust = calculate_thrust(state['velocity'][i,0], thrust_interp) # Thrust in lbs

        new_acceleration = np.array([(thrust - drag) / m,0.0]) # Acceleration in ft/s^2
        old_velocity = np.array([state['velocity'][i,0],0.0])
        new_velocity = np.add(old_velocity, new_acceleration*dt)
        new_position = np.add(state['position'][i], old_velocity*dt)

        # Append to all fields in the state dict:
        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], new_acceleration))
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],v,dt,batt_interp))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],0) # No turning (hopefully)
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Needed)
        state['Cd'] = np.append(state['Cd'],CD_Needed)
        state['alpha'] = np.append(state['alpha'],alpha) # Alpha needed to maintain level flight
        state['lift'] = np.append(state['lift'], W)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'],thrust-drag)
        state['F_lat'] = np.append(state['F_lat'],0)
        distance_traveled = distance_traveled + (state['velocity'][i,0]*dt)
        i += 1
        state['i'] = i

def turn(state:dict, constants:dict, turn_needed, batt_interp, thrust_interp, coeff_interp):
    """
    Calculate through a turn with constant altitude
    """

    # Convert constants for easier use:
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    dt = constants['dt']
    aoa_max_lift = constants['aoa_max_lift']

    i = state['i']

    CL_Turn, CD_Turn = xflr_results(coeff_interp, "alpha", aoa_max_lift) # Get lift/drag coeffs at point of max lift

    turn_traveled = 0

    while turn_needed >= turn_traveled:

        v = np.linalg.norm(state['velocity'][i]) # Current speed in ft/s

        q = 0.5 * rho * state['velocity'][i,0]**2 # Dynamic pressure in slugs/ft/s^2

        lift = CL_Turn * q * S # Calculate the actual lift considering the lift max

        if lift < W:
            beta = 0
            F_lat = 0
            omega = 0
        else:
            beta = np.arccos(W/lift)
            F_lat = lift * np.sin(beta)
            omega = F_lat / (m * v)

        drag = CD_Turn * q * S # Calculate the actual drag considering the lift max
        thrust = calculate_thrust(state['velocity'][i,0],thrust_interp) # Find thrust

        new_acceleration = np.array([(thrust - drag) / m,0.0]) # Acceleration in ft/s^2
        old_velocity = np.array([state['velocity'][i,0],0.0])
        new_velocity = np.add(old_velocity, new_acceleration*dt)
        new_position = np.add(state['position'][i], old_velocity*dt)
        turn_traveled = turn_traveled + np.degrees(omega*dt)  # Track turn in degrees

        # Append to all fields in the state dict:
        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], new_acceleration))
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'][i],v,dt,batt_interp))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],np.degrees(omega*dt)) # Now turning, convert to degrees
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Turn)
        state['Cd'] = np.append(state['Cd'],CD_Turn)
        state['alpha'] = np.append(state['alpha'],aoa_max_lift) # Alpha needed to maintain level flight
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
        'Cf': COEFF_FRICTION,
        'm': WEIGHT/GRAVITY,
        'g': GRAVITY,
        'rho': RHO,
        'dt': DT, 
        'battery_capacity': BATTERY_CAPACITY,
        'battery_cells': BATTERY_CELLS,
        'lap_count': LAP_COUNT,
        'cruise_aoa': ANGLE_OF_ATTACK,
        'aoa_max_lift': AOA_AT_MAX_LIFT
    }

    # Second create the interpolators with the imported MotoCalc data to be parsed
    imported_dataframe = imported_data(MOTOCALC_FILEPATH)
    batt_interp = create_batt_interpolator(imported_dataframe)
    thrust_interp = create_thrust_interpolator(imported_dataframe)
    coeff_interp = xflr_interp(XFLR5_FILEPATH)
    print(f"Interpolators created successfully.")

    # Third set up initial state as well as the state directory
    state = {
        'velocity': np.array([[0.0,0.0]]), #ft/s
        'position': np.array([[0.0,0.0]]), #ft
        'acceleration': np.array([[0.0,0.0]]), #ft/s^2
        'battery_charge': np.array([constants['battery_capacity']]), #mAh
        'time': np.array([0.0]), #seconds
        'turn_angle': np.array([0.0]), #degrees
        'thrust': np.array([0.0]), #lbs
        'Cl': np.array([0.0]),
        'Cd': np.array([0.0]),
        'alpha': np.array([0.0]),
        'lift': np.array([0.0]), #lbs
        'drag': np.array([0.0]), #lbs
        'F_long': np.array([0.0]), #lbs
        'F_lat': np.array([0.0]), #lbs
        'i': 0
    }
    print('State initial conditions completed successfully.')

    # Fourth, model the track that will be run and execute the functions
    
    # Takeoff and getting to altitude:
    takeoff(state,constants,batt_interp,thrust_interp,coeff_interp)
    print('Takeoff complete')
    climb(state,constants,batt_interp,thrust_interp,coeff_interp)
    print('Climb complete')
    # Complete lap simulation
    lap_counter = 0

    while (state['battery_charge'][-1] > constants['battery_capacity']*0.3) and (state['time'][-1] < 300): # Stop if battery below 30% or time exceeds 5 minutes
        straight(state,constants,500,batt_interp,thrust_interp,coeff_interp)
        turn(state,constants,180,batt_interp,thrust_interp,coeff_interp)
        straight(state,constants,500,batt_interp,thrust_interp,coeff_interp)
        turn(state,constants,360,batt_interp,thrust_interp,coeff_interp)
        straight(state,constants,500,batt_interp,thrust_interp,coeff_interp)
        turn(state,constants,180,batt_interp,thrust_interp,coeff_interp)
        straight(state,constants,500,batt_interp,thrust_interp,coeff_interp)
        lap_counter = lap_counter + 1
        print(f'Lap {lap_counter} complete')
    # Comprehensive plotting of all flight parameters
    print(f"Battery starting capacity: {constants['battery_capacity']} mAh")
    print(f"Remaining battery charge: {state['battery_charge'][-1]:.1f} mAh after {lap_counter} laps.")
    print(f"Time remaining: {300 - state['time'][-1]:.1f} seconds")

    turn_angle_rate = []
    for index in range(len(state['turn_angle'])-1):
        if state['turn_angle'][index] != 0 or state['turn_angle'][index+1] != 0:
            turn_angle_rate = np.append(turn_angle_rate, (state['turn_angle'][index]-state['turn_angle'][index+1])/(state['time'][index+1]-state['time'][index]))

    turn_angle_rate_average = np.mean(turn_angle_rate)
    turn_angle_max = np.max(turn_angle_rate)
    print(f"Average turn rate during turns: {turn_angle_rate_average:.2f} deg/s and Max turn rate: {turn_angle_max:.2f} deg/s")

    fig = plt.figure(figsize=[15, 8])
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f"Aircraft Lap Simulation Results", fontsize=18)

    # Row 1: Velocities, Battery, Position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(state['time'], state['velocity'][:,0], 'b-', linewidth=2, label='Horizontal')
    ax1.plot(state['time'], state['velocity'][:,1], 'r-', linewidth=2, label='Vertical')
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Velocity (ft/s)")
    ax1.set_title("Velocity vs Time"); ax1.grid(True); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(state['time'], state['battery_charge'], 'g-', linewidth=2)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Battery Charge (mAh)")
    ax2.set_title("Battery Charge vs Time"); ax2.grid(True)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(state['time'], state['position'][:,0], 'b-', linewidth=2, label='Horizontal')
    ax3.plot(state['time'], state['position'][:,1], 'r-', linewidth=2, label='Vertical')
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Position (ft)")
    ax3.set_title("Position vs Time"); ax3.grid(True); ax3.legend()

    # Row 2: Accelerations, Thrust, Turn Angle
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(state['time'], state['acceleration'][:,0], 'b-', linewidth=2, label='Horizontal')
    ax4.plot(state['time'], state['acceleration'][:,1], 'r-', linewidth=2, label='Vertical')
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