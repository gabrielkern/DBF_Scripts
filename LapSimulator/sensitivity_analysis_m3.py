"""
to run use the command: python -m LapSimulator.sensitivity_analysis_m3
"""

import numpy as np
import matplotlib.pyplot as plt
from .LapSimM3 import expose_vector

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
BATTERY_CAPACITY = 6000 #mAh
BANNER_LENGTH = 60 #in
WING_SPAN = 5 #ft

# SET STATIC PARAMETERS
LAP_ALTITUDE = 200 #ft
CLIMB_ANGLE = 30 #deg
WING_AREA = 4.8333 #ft^2
COEFF_FRICTION = 0.04 #unitless
GRAVITY = 32.174 #ft/s^2
RHO = 0.0023769 #slugs/ft^3 at sea level
DT = 0.05 #seconds
BATTERY_CELLS = 6 #number of cells in series
BATTERY_CAPACITY = 4500 #mAh

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
    'Empty_Weight': EMPTY_WEIGHT, #lbs
    'Banner_Length': BANNER_LENGTH, #in
    'Wing_Span': WING_SPAN, #ft
}

def mission_score_m3(constants):
    """Runs lapsim and finds the score from the given setup for M2."""
    lap_count = expose_vector(constants)
    RAC = (0.05 * constants['wing_span']) + 0.75 
    score = lap_count * constants['banner_length'] / RAC
    return score

constants['W'] = varied_parameters['Empty_Weight']
constants['m'] = constants['W'] / constants['g']
constants['wing_span'] = varied_parameters['Wing_Span']
constants['banner_length'] = varied_parameters['Banner_Length']
base = mission_score_m3(constants)
sweep = np.linspace(-0.25,0.25,50)

plt.figure(figsize=(12,8))
plt.suptitle('Mission 3 Airplane Sensitivity', fontsize=16, fontweight='bold')

colors = ['#B30638', '#A6A6A6', '#000000']

for idx, (items, values) in enumerate(varied_parameters.items()):
    prc_list = []
    for position in sweep:
        temp_params = varied_parameters.copy()
        temp_params[items] = values + (values * position)
        constants['W'] = temp_params['Empty_Weight']
        constants['m'] = constants['W'] / constants['g']
        constants['wing_span'] = temp_params['Wing_Span']
        constants['banner_length'] = temp_params['Banner_Length']
        abs_val = mission_score_m3(constants)
        prc_list.append((abs_val-base)/base * 100)
    plt.plot(sweep*100, prc_list, label=items, linewidth=2, marker='o', markersize=4, color=colors[idx])

plt.xlabel('Parameter Variation (%)', fontsize=12)
plt.ylabel('Mission Score Change (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)

print(f"Base Mission Score: {base:.3f}")

plt.tight_layout()
plt.show()