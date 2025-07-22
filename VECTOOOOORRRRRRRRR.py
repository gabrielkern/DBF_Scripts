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
WEIGHT = 8.0 #lbs
COEFF_FRICTION = 0.04 #unitless
GRAVITY = 32.174 #ft/s^2
RHO = 0.0023769 #slugs/ft^3 at sea level
DT = 0.05 #seconds
BATTERY_CAPACITY = 3300 #mAh
BATTERY_CELLS = 4 #number of cells in series
LAP_COUNT = 3 # Number of laps that the program should run

MOTOCALC_FILEPATH = '201_14x12.csv'
XFLR5_FILEPATH = 'Lark_8lb_VLMvisc.csv'


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

# Calculate remaining parameters
Mass = WEIGHT / GRAVITY # slugs

# These are the functions that do the initial data processing

def imported_data(csv_filepath:str) -> pd.DataFrame:
    """
    Function that pulls data from a MotoCalc csv file and returns thrust
    as a function of airspeed
    """

    df = pd.read_csv(
        csv_filepath,
        skiprows=9,
        skipfooter=4,
        engine='python',
        encoding='latin-1'
    )

    # Define the column names in the correct order
    column_names = [
        'airspeed_mph', 'drag_oz', 'lift_oz', 'batt_amps', 'motor_amps',
        'motor_volts', 'input_watts', 'loss_watts', 'output_watts',
        'motor_eff_pct', 'shaft_eff_pct', 'prop_rpm', 'thrust_oz',
        'prop_speed_mph', 'prop_eff_pct', 'total_eff_pct', 'time_min'
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
    return np.max(new_charge, 0)  # Ensure change in charge isn't negative, also in mAh

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
    df = pd.read_csv(
    filename,
    comment="#",          # skip every line that starts with ‘#’
    skipinitialspace=True # drop spaces that follow each comma
    )
    # clean the headers
    df.columns = df.columns.str.strip()   # remove stray spaces
    # --- convert strings to numbers (just in case) ----------------------
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

    i = 0 # Index of the previous time stop

    CL_Takeoff,CD_Takeoff = xflr_results(coeff_interp, "alpha", 1) # Get CL and CD at 0 deg alpha

    # Calculate the takeoff sequence until the lift is greater than weight
    while state['lift'][i] <= W:

        v = np.linalg.norm(state['velocity'][i]) # Current speed in ft/s

        q = 0.5 * rho * v**2 # Dynamic pressure in slugs/ft/s^2
        drag = CD_Takeoff * q * S # Drag in lbs
        lift = CL_Takeoff * q * S # Lift in lbs

        f_ground = Cf * (W - lift) # Friction force in lbs

        thrust = calculate_thrust(v, thrust_interp) # Thrust in lbs

        new_acceleration = np.array([(thrust - drag - f_ground) / m,0.0]) # Acceleration in ft/s^2
        new_velocity = np.add(state['velocity'][i], new_acceleration*dt)
        new_position = np.add(state['position'][i], state['velocity'][i]*dt)

        # Append to all fields in the state dict:
        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], new_acceleration))
        state['battery_charge'] = np.append(state['battery_charge'],calculate_charge(state['battery_charge'],v,dt,batt_interp))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'],0) # No turning (hopefully)
        state['thrust'] = np.append(state['thrust'],thrust)
        state['Cl'] = np.append(state['Cl'],CL_Takeoff)
        state['Cd'] = np.append(state['Cd'],CD_Takeoff)
        state['alpha'] = np.append(state['alpha'],0) # Hold at 0 aoa
        state['lift'] = np.append(state['lift'], lift)
        state['drag'] = np.append(state['lift'], drag)
        state['F_long'] = np.append(state['F_long'],thrust-drag-f_ground)
        state['F_lat'] = np.append(state['F_lat'],0)

        # Run the loop cleanup tasks
        i = i + 1

# Running the file:
if __name__ == '__main__':

    # First get all the constant parameters into a structured dictionary
    constants = {
        'altitude': LAP_ALTITUDE,
        'theta': CLIMB_ANGLE,
        'S': WING_AREA,
        'W': WEIGHT,
        'Cf': COEFF_FRICTION,
        'm': Mass,
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
        'F_lat': np.array([0.0]) #lbs
    }
    print('State initial conditions completed successfully.')

    # Fourth, model the track that will be run and execute the functions
    
    # Takeoff and getting to altitude:
    takeoff(state,constants,batt_interp,thrust_interp,coeff_interp)
    #climb(state,constante,batt_interp,thrust_interp,coeff_interp)
    print('takeoff complete')

    # Laps
    '''
    for lap in range(constants['lap_count']):
        straight(500 feet)
        turn(180 deg)
        straight(500 feet)
        turn(360 deg)
        straight(500 feet)
        turn(180 deg)
        straight(500 feet)
    '''

    # Fifth, set up the plotting

    fig = plt.figure(figsize=[15,8])
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle(f"PyLapSim", fontsize=16)

    x_velocity = fig.add_subplot(gs[0, 0])
    x_velocity.plot(state['time'], state['battery_charge'], marker='s', linestyle='-')
    x_velocity.set_xlabel("Time"); x_velocity.set_ylabel("Velocity")
    x_velocity.set_title("Velocity vs Time"); x_velocity.grid(True)

    plt.show()

    # Need to fix the battery charging, need to add the other functions, and need to verify results somehow