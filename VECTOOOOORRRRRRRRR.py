# OHHHHHHH YEAHHHHHHHHHHH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print(thrust_data('201_14x12.csv'))