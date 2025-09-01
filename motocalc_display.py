import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_motocalc_data(filename):
    """Load MotoCalc data from CSV file, skipping header rows"""
    df = pd.read_csv(
        filename,
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

def convert_units(df):
    """Convert units: mph to m/s, oz to N"""
    df['airspeed_ms'] = df['airspeed_mph'] * 0.44704  # mph to m/s
    df['thrust_n'] = df['thrust_oz'] * 0.278014   # oz to N
    return df

def create_plots(df):
    """Create airspeed vs thrust and airspeed vs amps plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Airspeed vs Thrust
    ax1.plot(df['airspeed_ms'], df['thrust_n'], 'b-o', markersize=4)
    ax1.set_xlabel('Airspeed (m/s)')
    ax1.set_ylabel('Thrust (N)')
    ax1.set_title('Airspeed vs Thrust')
    ax1.grid(True, alpha=0.3)
    
    # Airspeed vs Battery Amps
    ax2.plot(df['airspeed_ms'], df['batt_amps'], 'r-o', markersize=4)
    ax2.set_xlabel('Airspeed (m/s)')
    ax2.set_ylabel('Battery Amps (A)')
    ax2.set_title('Airspeed vs Battery Amps')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to load data and create plots"""
    filename = 'Lark8lb10x6.csv'
    
    df = load_motocalc_data(filename)
    df = convert_units(df)
    create_plots(df)

if __name__ == "__main__":
    main()