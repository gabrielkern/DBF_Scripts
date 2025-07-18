# OHHHHHHH YEAHHHHHHHHHHH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# SET STATIC PARAMETERS


def thrust_data(csv_filepath:str) -> pd.DataFrame:
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

# Running the file:
if __name__ == '__main__':
    propellor_data_file_path = "201_14x12.csv"
    prop_data = thrust_data(propellor_data_file_path)
    batt_interpolator = create_batt_interpolator(prop_data)

    # 3. Use the function in a simulation loop.
    initial_charge_mAh = 3000.0  # Example initial charge
    time_step_s = 1.0           # Simulate 1-second intervals
        
    current_charge = initial_charge_mAh
        
    print(f"\nStarting simulation with {current_charge:.2f} mAh.")
        
    # Simulate flying at 50 mph for 10 steps (10 seconds)
    for i in range(10):
        airspeed = 3*(i**2) # Constant speed for this example
        current_charge = calculate_charge(current_charge, airspeed, time_step_s, batt_interpolator)
        print(f"After step {i+1}, charge is: {current_charge:.2f} mAh")
            
        if current_charge == 0:
            print("Battery depleted!")
            break

