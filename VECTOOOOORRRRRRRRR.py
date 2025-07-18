# OHHHHHHH YEAHHHHHHHHHHH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    for xflr data you need to run the interp once then bring in coeff_interp into functions
    results = xflr_results(interpolators, "Cl", 0.3778)
    this will ouput correspoding other two values of CL, CD, alpha
    
    """



# Calculate remaining parameters
Mass = WEIGHT / GRAVITY # slugs

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
        'battery_cells': BATTERY_CELLS
    }

    # Second create the interpolators with the imported MotoCalc data to be parsed
    imported_dataframe = imported_data(MOTOCALC_FILEPATH)
    batt_interp = create_batt_interpolator(imported_dataframe)
    thrust_interp = create_thrust_interpolator(imported_dataframe)
    print(f"Interpolators created successfully. {constants['battery_capacity']}")

    # Third set up initial state as well as the state directory
    state = {
        'velocity': 0.0, #ft/s
        'position': 0.0, #ft
        'acceleration': 0.0, #ft/s^2
        'battery_charge': constants['battery_capacity'], #mAh
        'time': 0.0, #seconds
        'turn_angle': 0.0, #degrees
        'thrust': 0.0, #lbs
        'Cl': 0.0,
        'Cd': 0.0,
        'alpha': 0.0,
        'lift': 0.0, #lbs
        'drag': 0.0, #lbs
        'F_long': 0.0, #lbs
        'F_lat': 0.0 #lbs
    }