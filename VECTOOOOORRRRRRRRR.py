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

MOTOCALC_FILEPATH = '201_14x12.csv'
XFLR5_FILEPATH = ''

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

def calculate_charge(current_charge:float,vel_mph:float, dt:float, batt_interp) -> float:
    """
    Calculates the new battery charge given usage at speed vel_mph and the data
    interpolated and passed through batt_interp. Inputs the old charge, the
    velocity in mph, the timestep in seconds, and the batt_interp function.
    Returns the new charge in milliamp-hours.
    """

    amps = batt_interp(vel_mph)
    charge_used = amps * (dt / 3600) * 1000  # Convert to mAh of charge used
    new_charge = current_charge - charge_used
    return max(new_charge, 0)  # Ensure change in charge isn't negative

def calculate_thrust(vel_mph:float, thrust_interp) -> float:
    """
    Calculates the thrust at a given velocity using the thrust_interp function.
    Inputs the velocity in mph and the thrust_interp function.
    Returns the thrust in ounces.
    """
    return thrust_interp(vel_mph)

def xflr_dat(filename,name,value):
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

    cd_of_cl = interp1d(df["CL"], df["CD"], kind='cubic')
    alpha_of_cl = interp1d(df["CL"], df["alpha"], kind='cubic')
    cl_of_cd = interp1d(df["CD"], df["CL"], kind='cubic')
    alpha_of_cd = interp1d(df["CD"], df["alpha"], kind='cubic')
    cd_of_alpha = interp1d(df["alpha"], df["CD"], kind='cubic')
    cl_of_alpha = interp1d(df["alpha"], df["CL"], kind='cubic')

    if name == "CL" or "cl" or "Cl" or "cL":
        CD = cd_of_cl(value)
        alpha = alpha_of_cl(value)
        return CD,alpha
    if name == "CD" or "cd" or 'Cd' or "cD":
        CL = cl_of_cd(value)
        alpha = alpha_of_cd(value)
        return CL, alpha
    if name == "alpha" or "Alpha":
        CL = cl_of_alpha(value)
        CD = cd_of_alpha(value)
        return CL,CD

# Running the file:
if __name__ == '__main__':
    
    # First get all the constant parameters into a structured dictionary

    # Second create the interpolators with the imported MotoCalc data to be parsed
    imported_dataframe = imported_data(MOTOCALC_FILEPATH)
    batt_interp = create_batt_interpolator(imported_dataframe)
    thrust_interp = create_thrust_interpolator(imported_dataframe)
    print("Interpolators created successfully.")