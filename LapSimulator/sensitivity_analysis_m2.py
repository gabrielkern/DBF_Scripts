"""
to run use the command: python -m LapSimulator.sensitivity_analysis_m2
"""

import numpy as np
import matplotlib.pyplot as plt
from .VECTOOOOORRRRRRRRR import expose_vector

# constants dict needed for lapsim
"""    constants = {
        'altitude': LAP_ALTITUDE,
        'theta': CLIMB_ANGLE,
        'S': WING_AREA,
        'W': WEIGHT,
        'Cf': COEFF_FRICTION,
        'm': MASS,
        'g': GRAVITY,
        'rho': RHO,
        'dt': DT, 
        'battery_capacity': BATTERY_CAPACITY,
        'battery_cells': BATTERY_CELLS,
        'max_aoa': MAX_AOA,
        'motocalc_filepath': MOTOCALC_FILEPATH,
        'xflr5_filepath': XFLR5_FILEPATH,
    }
"""

# SET BASELINES
EMPTY_WEIGHT = 10 #lb
CARGO_CAPACITY = 12 #pucks
PASSENGER_CAPACITY = 36 #ducks
BATTERY_CAPACITY = 6000 #mAh

# SET STATIC PARAMETERS
LAP_ALTITUDE = 200 #ft
CLIMB_ANGLE = 30 #deg
WING_AREA = 4.8333 #ft^2
COEFF_FRICTION = 0.04 #unitless
GRAVITY = 32.174 #ft/s^2
RHO = 0.0023769 #slugs/ft^3 at sea level
DT = 0.05 #seconds
BATTERY_CELLS = 6 #number of cells in series

MOTOCALC_FILEPATH = 'Lark8lb10x6.csv'
XFLR5_FILEPATH = 'Lark45lbfull04m.csv'

constants = {
    'altitude': LAP_ALTITUDE,
    'theta': CLIMB_ANGLE,
    'S': WING_AREA,
    'Cf': COEFF_FRICTION,
    'g': GRAVITY,
    'rho': RHO,
    'dt': DT, 
    'battery_capacity': BATTERY_CAPACITY,
    'battery_cells': BATTERY_CELLS,
    'max_aoa': 10, #degrees
    'motocalc_filepath': MOTOCALC_FILEPATH,
    'xflr5_filepath': XFLR5_FILEPATH,
}

varied_parameters = {
    'EW': EMPTY_WEIGHT, #lb
    'Cargo': CARGO_CAPACITY, #pucks
    'Passengers': PASSENGER_CAPACITY, #ducks
    'Battery_Capacity': BATTERY_CAPACITY, #mAh
}

def mission_score_m2(constants, cargo, passengers, battery_capacity):
    """Runs lapsim and finds the score from the given setup for M2."""
    lap_count = expose_vector(constants)
    income = ( passengers * (6 + ( 2 * lap_count)) ) + ( cargo * (10 + ( 8 * lap_count)) )
    EF = battery_capacity * constants['battery_cells'] * 4.2 / 1000 / 100
    cost = lap_count * (10 + (0.5 * passengers) + (2 * cargo)) * EF
    return income - cost

constants['W'] = varied_parameters['EW'] + (varied_parameters['Cargo'] * 0.375) + (varied_parameters['Passengers'] * 0.04375)
constants['m'] = constants['W'] / constants['g']
constants['battery_capacity'] = varied_parameters['Battery_Capacity']
base = mission_score_m2(constants, varied_parameters['Cargo'], varied_parameters['Passengers'], varied_parameters['Battery_Capacity'])
sweep = np.linspace(-0.5,0.5,50)

plt.figure(figsize=(12,8))
plt.suptitle('Mission 2 Score Sensitivity Analysis', fontsize=16, fontweight='bold')

for items, values in varied_parameters.items():
    prc_list = []
    for position in sweep:
        temp_params = varied_parameters.copy()
        temp_params[items] = values + (values * position)
        constants['W'] = temp_params['EW'] + (temp_params['Cargo'] * 0.375) + (temp_params['Passengers'] * 0.04375)
        constants['m'] = constants['W'] / constants['g']
        constants['battery_capacity'] = temp_params['Battery_Capacity']
        abs_val = mission_score_m2(constants, temp_params['Cargo'], temp_params['Passengers'], temp_params['Battery_Capacity'])
        prc_list.append((abs_val-base)/base * 100)
    plt.plot(sweep*100, prc_list, label=items, linewidth=2, marker='o', markersize=4)

plt.xlabel('Parameter Variation (%)', fontsize=12)
plt.ylabel('Mission Score Change (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)

print(f"Base Mission Score: {base:.3f}")

plt.tight_layout()
plt.show()