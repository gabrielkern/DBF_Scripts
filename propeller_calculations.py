import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set values
PANEL_COUNT = 10
A_GUESS = 0.5
B_GUESS = 0.125
PITCH = 12.0 # in inches
DIAMETER = 15.0 # in inches
RPM_RANGE = (4,20) # in 1000 rpms
V_INF = 25 # m/s
RHO = 1.225 # kg/m^3
PROP_COUNT = 2
TOL = 1e-4
PLOT_FUNCTION = True
HEATMAP = False

def find_coefficients(alpha):
    """Flat-plate airfoil coefficients. Can be modified in future to use/match real data."""
    Cla = 2 * np.pi
    Cdo = 0.01
    k = 0.05
    Cl = Cla * alpha
    Cd = Cdo + k * Cl**2
    return Cl, Cd

def find_local_chord(r_rel):
    """Function to generate estimate of local chord based on radial position."""

    # Constant 18 mm for now
    return 0.018


def apply_bemt(panel_count=PANEL_COUNT, a_guess=A_GUESS, b_guess=B_GUESS, pitch=PITCH, diameter=DIAMETER, 
               rpm_range=RPM_RANGE, v_inf=V_INF, rho=RHO, B = PROP_COUNT, tol=TOL):
    """Applies blade-element momentum theory to propeller defined by parameters."""

    # Quick check to make sure things don't break
    if v_inf <= 0:
        v_inf = 0.0001
    if rpm_range[0] <= 0:
        rpm_range = (1, rpm_range[1])
    if rpm_range[1] < rpm_range[0]:
        rpm_range = (1,1)

    # Calculated constants and unit conversions
    pitch = pitch * 0.0254 # convert to meters
    diameter = diameter * 0.0254 # convert to meters
    radius = diameter / 2
    dR = radius / panel_count

    # Initialize final results dictionary
    final_values = {}

    # Define variables and loop over RPM values
    for rpm in range(rpm_range[0], rpm_range[1]+1):
        rpm_true = rpm * 1000
        omega = rpm_true * 2 * np.pi / 60
        print(omega)
        r_values = [(i + 0.5) * dR for i in range(panel_count)]
        
        # Initialize lists to store results
        rpm_dict = {
            'r': [],
            'a': [],
            'b': [],
            'dT': [],
            'dQ': [],
            'Thrust': 0,
            'Torque': 0,
            'CT': 0,
            'CQ': 0,
            'Efficiency': 0
        }

        # Loop over each section
        for r in r_values:
            
            # Iterative solution for a and b, where a is axial inflow factor
            # and b is the angular inflow or swirl factor
            a = a_guess
            b = b_guess
            converged = False

            # Calculate theta
            theta = np.arctan(pitch / (2 * np.pi * r))

            # Convergence process for particular station
            while not converged:

                # Calculate the velocities & angles based on guess a,b
                V_in = v_inf * (1 + a)
                V_rot = omega * r * (1 - b)
                V_tot = np.sqrt(V_in**2 + V_rot**2)
                phi = np.arctan(V_in/V_rot)
                alpha = theta - phi

                # Calcuilate lift and drag coefficients
                Cl, Cd = find_coefficients(alpha)

                # Find local chord
                c = find_local_chord(r / radius)

                # Calculate section thrust and torque
                dT = 0.5 * rho * V_tot**2 * c * ( (Cl * np.cos(phi) ) - ( Cd * np.sin(phi)) ) * B * dR
                dQ = 0.5 * rho * V_tot**2 * c * ( (Cd * np.cos(phi) ) + ( Cl * np.sin(phi)) ) * B * r * dR

                # Calculate new a and b values
                c = dT / ( 4 * np.pi * r * rho * v_inf**2 * dR )
                a_new = ( -1 + np.sqrt(1 + (4 * c)) ) / 2
                b_new = dQ / ( 4 * np.pi * r**3 * v_inf * (1 + a_new) * omega * dR )
                print(f"C: {c}")
                print(f"A: {a_new}")
                print(f"B: {b_new}")
                if np.isnan(a_new) or np.isnan(b_new):
                    raise Exception("NaN value encountered in calculations. Check inputs and coefficients.")
                else:
                    print("Values OK")

                # Check for convergence
                if abs(a_new - a) < tol and abs(b_new - b) < tol:
                    converged = True

                relaxation_factor = 0.3  # Start conservative
                a = relaxation_factor * a_new + (1 - relaxation_factor) * a  
                b = relaxation_factor * b_new + (1 - relaxation_factor) * b

            # Do final calculations to get values

            # Store values for this station
            rpm_dict['r'].append(r)
            rpm_dict['a'].append(a)
            rpm_dict['b'].append(b)
            rpm_dict['dT'].append(dT)
            rpm_dict['dQ'].append(dQ)

        # Store final values for this rpm
        n = rpm_true / 60
        rpm_dict['Thrust'] = sum(rpm_dict['dT'])
        rpm_dict['Torque'] = sum(rpm_dict['dQ'])
        rpm_dict['CT'] = rpm_dict['Thrust'] / (rho * n**2 * diameter**4)
        rpm_dict['CQ'] = rpm_dict['Torque'] / (rho * n**2 * diameter**5)
        J = v_inf / (n * diameter)
        if rpm_dict['CQ'] > 0:
            rpm_dict['Efficiency'] = (J * rpm_dict['CT']) / (2 * np.pi * rpm_dict['CQ'])
        else:
            rpm_dict['Efficiency'] = 0
        final_values[str(rpm_true)] = rpm_dict

    return final_values

def plot_prop_data(results_data):
    """Plot propeller performance data in a 2x2 grid layout."""
    
    # Extract data from results dictionary
    rpm_values = []
    thrust_values = []
    torque_values = []
    ct_values = []
    cq_values = []
    efficiency_values = []
    
    for rpm_str, data in results_data.items():
        rpm_values.append(int(rpm_str) / 1000)  # Convert to thousands
        thrust_values.append(data['Thrust'])
        torque_values.append(data['Torque'])
        ct_values.append(data['CT'])
        cq_values.append(data['CQ'])
        efficiency_values.append(data['Efficiency'])
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Propeller Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: RPM vs Thrust
    ax1.plot(rpm_values, thrust_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('RPM (×1000)')
    ax1.set_ylabel('Thrust (N)')
    ax1.set_title('Thrust vs RPM')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RPM vs Torque
    ax2.plot(rpm_values, torque_values, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('RPM (×1000)')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_title('Torque vs RPM')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RPM vs CT & CQ (same graph)
    ax3.plot(rpm_values, ct_values, 'g-o', linewidth=2, markersize=6, label='CT')
    ax3.plot(rpm_values, cq_values, 'm-s', linewidth=2, markersize=6, label='CQ')
    ax3.set_xlabel('RPM (×1000)')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Thrust & Torque Coefficients vs RPM')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RPM vs Efficiency
    ax4.plot(rpm_values, efficiency_values, 'orange', marker='o', linewidth=2, markersize=6)
    ax4.set_xlabel('RPM (×1000)')
    ax4.set_ylabel('Efficiency')
    ax4.set_title('Efficiency vs RPM')
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def calculate_efficiency_heatmap(pitch_range, diameter_range, rpm_range=RPM_RANGE, v_inf=V_INF, prop_count=PROP_COUNT):
    """Calculate efficiency for all pitch/diameter combinations and return as arrays for heatmap."""
    
    efficiency_matrix = np.zeros((len(pitch_range), len(diameter_range)))
    
    for i, pitch in enumerate(pitch_range):
        for j, diameter in enumerate(diameter_range):
            print(f"Calculating for pitch={pitch:.1f}, diameter={diameter:.1f}")
            
            # Run BEMT analysis for this pitch/diameter combination
            results = apply_bemt(
                panel_count=PANEL_COUNT, 
                a_guess=A_GUESS, 
                b_guess=B_GUESS, 
                pitch=pitch, 
                diameter=diameter, 
                rpm_range=rpm_range, 
                v_inf=v_inf, 
                rho=RHO, 
                B=prop_count, 
                tol=TOL
            )
            
            # Calculate average efficiency across RPM range
            efficiencies = [data['Efficiency'] for data in results.values()]
            avg_efficiency = np.mean(efficiencies)
            efficiency_matrix[i, j] = max(avg_efficiency,0)
    
    return efficiency_matrix

def plot_efficiency_heatmap(pitch_range, diameter_range, efficiency_matrix):
    """Plot efficiency heatmap for pitch vs diameter combinations."""
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        efficiency_matrix,
        xticklabels=[f'{d:.1f}' for d in diameter_range],
        yticklabels=[f'{p:.1f}' for p in pitch_range],
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Efficiency'}
    )
    
    plt.title(f'Propeller Efficiency Heatmap\n(Average across RPM {RPM_RANGE[0]*1000}-{RPM_RANGE[1]*1000}, V∞={V_INF} m/s, {PROP_COUNT} Props)', fontsize=14, fontweight='bold')
    plt.xlabel('Diameter (inches)', fontsize=12)
    plt.ylabel('Pitch (inches)', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Original analysis (hidden when PLOT_FUNCTION = False)
    if PLOT_FUNCTION:
        results = apply_bemt()
        for rpm, data in results.items():
            print(f"RPM: {rpm}")
            print(f"  Thrust (N): {data['Thrust']:.2f}")
            print(f"  Torque (Nm): {data['Torque']:.2f}")
            print(f"  CT: {data['CT']:.4f}")
            print(f"  CQ: {data['CQ']:.4f}")
            print(f"  Efficiency: {data['Efficiency']:.4f}")
            print()
        plot_prop_data(results)
    
    if HEATMAP:
        # New heatmap analysis
        pitch_range = np.linspace(6, 14, 9)  # 4 to 10 inches
        diameter_range = np.linspace(10, 18, 9)  # 10 to 18 inches
        
        print("Generating efficiency heatmap...")
        efficiency_matrix = calculate_efficiency_heatmap(pitch_range, diameter_range)
        plot_efficiency_heatmap(pitch_range, diameter_range, efficiency_matrix)
    
    results = apply_bemt()