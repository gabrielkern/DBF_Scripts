"""
APC Propeller Performance Data Analyzer

This module provides functionality to analyze APC propeller performance data
from the PERFILES_WEB database. It calculates thrust and current draw based on
propeller geometry, motor characteristics, and flight conditions.
"""

import re
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from scipy.interpolate import interp1d

# Constants
VOLTAGE_PER_CELL = 3.7  # Volts per LiPo cell
VOLTAGE_CORRECTION_FACTOR = 0.75  # Correction factor for battery voltage under load
MAX_CURRENT_AMPS = 100.0  # Maximum current draw in Amps
RPM_STEP_DOWN = 100  # RPM reduction step when current limit is exceeded

# Path to APC propeller data files
APC_DATA_DIR = Path(__file__).parent.parent / "PERFILES2"


def find_propeller_file(diameter: float, pitch: float) -> Path:
    """
    Find the APC propeller data file for the given diameter and pitch.

    Args:
        diameter: Propeller diameter in inches
        pitch: Propeller pitch in inches

    Returns:
        Path to the propeller data file

    Raises:
        FileNotFoundError: If no matching propeller file is found
    """
    # Convert diameter and pitch to the filename format
    # Handle decimal values (e.g., 5.25 becomes 525)
    if diameter == int(diameter):
        diameter_str = str(int(diameter))
    else:
        diameter_str = str(diameter).replace(".", "")

    if pitch == int(pitch):
        pitch_str = str(int(pitch))
    else:
        pitch_str = str(pitch).replace(".", "")

    # Try exact match first: PER3_{diameter}x{pitch}E.dat
    filename = f"PER3_{diameter_str}x{pitch_str}E.dat"
    filepath = APC_DATA_DIR / filename

    if filepath.exists():
        return filepath

    raise FileNotFoundError(
        f"No APC propeller data file found for {diameter}x{pitch}. "
        f"Searched for: {filename} in {APC_DATA_DIR}"
    )


def parse_apc_data_file(filepath: Path) -> Dict[int, np.ndarray]:
    """
    Parse an APC propeller data file and extract performance data for each RPM.

    Args:
        filepath: Path to the APC propeller data file

    Returns:
        Dictionary mapping RPM values to numpy arrays with columns:
        [airspeed_mph, power_watts, thrust_newtons]
    """
    rpm_data = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_rpm = None
    data_rows = []

    for line in lines:
        # Check for RPM section header
        rpm_match = re.search(r'PROP RPM\s*=\s*(\d+)', line)
        if rpm_match:
            # Save previous RPM data if it exists
            if current_rpm is not None and data_rows:
                rpm_data[current_rpm] = np.array(data_rows)

            # Start new RPM section
            current_rpm = int(rpm_match.group(1))
            data_rows = []
            continue

        # Parse data rows (they start with numbers)
        if current_rpm is not None:
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                try:
                    # Split the line and extract relevant columns
                    parts = stripped.split()
                    if len(parts) >= 11:
                        airspeed_mph = float(parts[0])      # V (mph)
                        power_watts = float(parts[8])       # PWR (W)
                        thrust_newtons = float(parts[10])   # Thrust (N)

                        data_rows.append([airspeed_mph, power_watts, thrust_newtons])
                except (ValueError, IndexError):
                    # Skip lines that don't parse correctly
                    continue

    # Save the last RPM section
    if current_rpm is not None and data_rows:
        rpm_data[current_rpm] = np.array(data_rows)

    return rpm_data


def interpolate_propeller_performance(
    rpm_data: Dict[int, np.ndarray],
    target_rpm: float,
    airspeed_mph: float
) -> Tuple[float, float]:
    """
    Interpolate propeller performance at a specific RPM and airspeed.

    Args:
        rpm_data: Dictionary of RPM -> performance data arrays
        target_rpm: Target RPM value
        airspeed_mph: Airspeed in miles per hour

    Returns:
        Tuple of (thrust_newtons, power_watts)

    Raises:
        ValueError: If target RPM or airspeed is out of range
    """
    available_rpms = sorted(rpm_data.keys())

    if not available_rpms:
        raise ValueError("No RPM data available")

    # Clamp target_rpm to available range
    min_rpm = available_rpms[0]
    max_rpm = available_rpms[-1]

    if target_rpm < min_rpm:
        target_rpm = min_rpm
    elif target_rpm > max_rpm:
        target_rpm = max_rpm

    # Find the two closest RPM values
    lower_rpm = None
    upper_rpm = None

    for rpm in available_rpms:
        if rpm <= target_rpm:
            lower_rpm = rpm
        if rpm >= target_rpm and upper_rpm is None:
            upper_rpm = rpm

    # If exact match, just interpolate at that RPM
    if lower_rpm == upper_rpm:
        data = rpm_data[lower_rpm]
        airspeeds = data[:, 0]
        powers = data[:, 1]
        thrusts = data[:, 2]

        # Create copy of the incoming airspeed
        actual_airspeed_mph = airspeed_mph

        # Interpolate for airspeed
        if actual_airspeed_mph < airspeeds[0]:
            actual_airspeed_mph = airspeeds[0]
        elif actual_airspeed_mph > airspeeds[-1]:
            actual_airspeed_mph = airspeeds[-1]

        power_interp = interp1d(airspeeds, powers, kind='linear', fill_value='extrapolate')
        thrust_interp = interp1d(airspeeds, thrusts, kind='linear', fill_value='extrapolate')

        return float(thrust_interp(actual_airspeed_mph)), float(power_interp(actual_airspeed_mph))

    # Create copy of the incoming airspeed
    actual_airspeed_mph = airspeed_mph

    # Interpolate between two RPM values
    lower_data = rpm_data[lower_rpm]
    upper_data = rpm_data[upper_rpm]

    # Get performance at lower RPM
    lower_airspeeds = lower_data[:, 0]
    lower_powers = lower_data[:, 1]
    lower_thrusts = lower_data[:, 2]

    if actual_airspeed_mph < lower_airspeeds[0]:
        actual_airspeed_mph = lower_airspeeds[0]
    elif actual_airspeed_mph > lower_airspeeds[-1]:
        actual_airspeed_mph = lower_airspeeds[-1]

    lower_power_interp = interp1d(lower_airspeeds, lower_powers, kind='linear', fill_value='extrapolate')
    lower_thrust_interp = interp1d(lower_airspeeds, lower_thrusts, kind='linear', fill_value='extrapolate')

    lower_power = float(lower_power_interp(actual_airspeed_mph))
    lower_thrust = float(lower_thrust_interp(actual_airspeed_mph))

    # Create copy of the incoming airspeed
    actual_airspeed_mph = airspeed_mph

    # Get performance at upper RPM
    upper_airspeeds = upper_data[:, 0]
    upper_powers = upper_data[:, 1]
    upper_thrusts = upper_data[:, 2]

    if actual_airspeed_mph < upper_airspeeds[0]:
        actual_airspeed_mph = upper_airspeeds[0]
    elif actual_airspeed_mph > upper_airspeeds[-1]:
        actual_airspeed_mph = upper_airspeeds[-1]

    upper_power_interp = interp1d(upper_airspeeds, upper_powers, kind='linear', fill_value='extrapolate')
    upper_thrust_interp = interp1d(upper_airspeeds, upper_thrusts, kind='linear', fill_value='extrapolate')

    upper_power = float(upper_power_interp(actual_airspeed_mph))
    upper_thrust = float(upper_thrust_interp(actual_airspeed_mph))

    # Linear interpolation between RPMs
    rpm_fraction = (target_rpm - lower_rpm) / (upper_rpm - lower_rpm)
    interpolated_power = lower_power + rpm_fraction * (upper_power - lower_power)
    interpolated_thrust = lower_thrust + rpm_fraction * (upper_thrust - lower_thrust)

    return interpolated_thrust, interpolated_power


def get_propeller_performance(
    diameter: float,
    pitch: float,
    motor_kv: float,
    battery_cell_count: int,
    airspeed_mph: float
) -> Tuple[float, float]:
    """
    Calculate propeller thrust and current draw for given conditions.

    This function:
    1. Finds the appropriate APC propeller data file
    2. Calculates battery voltage with correction factor
    3. Calculates motor RPM from voltage and KV
    4. Interpolates thrust and power from propeller data
    5. Limits RPM if current exceeds 100A

    Args:
        diameter: Propeller diameter in inches
        pitch: Propeller pitch in inches
        motor_kv: Motor velocity constant in RPM/volt
        battery_cell_count: Number of battery cells in series
        airspeed_mph: Airspeed in miles per hour

    Returns:
        Tuple of (thrust_lbs, current_amps)

    Raises:
        FileNotFoundError: If propeller data file not found
        ValueError: If inputs are invalid or data out of range
    """
    # Calculate battery voltage
    voltage = battery_cell_count * VOLTAGE_PER_CELL * VOLTAGE_CORRECTION_FACTOR

    # Calculate initial RPM from motor KV and voltage
    target_rpm = motor_kv * voltage

    # Find and parse propeller data file
    filepath = find_propeller_file(diameter, pitch)
    rpm_data = parse_apc_data_file(filepath)

    # Get available RPM range
    available_rpms = sorted(rpm_data.keys())
    min_available_rpm = available_rpms[0]
    max_available_rpm = available_rpms[-1]

    # Start with target RPM, but clamp to available range
    current_rpm = min(max(target_rpm, min_available_rpm), max_available_rpm)

    # Iterate to find RPM that satisfies current limit
    while current_rpm >= min_available_rpm:
        # Get thrust and power at current RPM and airspeed
        thrust, power = interpolate_propeller_performance(rpm_data, current_rpm, airspeed_mph)

        # Calculate current draw
        current = power / voltage

        # Check if current is within limits
        if current <= MAX_CURRENT_AMPS:
            thrust_lbs = thrust * 0.224809  # Convert Newtons to pounds
            return thrust_lbs, current

        # Reduce RPM and try again
        current_rpm -= RPM_STEP_DOWN

        # If we've gone below minimum RPM, use minimum and break
        if current_rpm < min_available_rpm:
            current_rpm = min_available_rpm
            thrust, power = interpolate_propeller_performance(rpm_data, current_rpm, airspeed_mph)
            current = power / voltage
            thrust_lbs = thrust * 0.224809  # Convert Newtons to pounds
            return thrust_lbs, current

    # This should never happen, but just in case
    thrust, power = interpolate_propeller_performance(rpm_data, min_available_rpm, airspeed_mph)
    current = power / voltage
    thrust_lbs = thrust * 0.224809
    return thrust_lbs, current