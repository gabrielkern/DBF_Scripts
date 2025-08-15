import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Subplot layout

# Functions
def calculate_wing_geometry(c_root, taper_ratio, AR):
    """Calculates basic wing geometry parameters."""
    c_tip = taper_ratio * c_root
    b = AR * c_root * (1 + taper_ratio) / 2.0
    s = b / 2.0 # semi-span
    S_wing = (c_root + c_tip) / 2.0 * b
    return c_tip, b, s, S_wing

def get_local_chord_and_mu(phi, c_root, taper_ratio, semi_span, section_lift_slope_a0):
    """Calculates local chord c and parameter mu."""
    
    local_chord = c_root * (1.0 - (1.0 - taper_ratio) * np.abs(np.cos(phi)))
    mu_phi = (local_chord * section_lift_slope_a0) / (8.0 * semi_span)
    return local_chord, mu_phi

def solve_for_An_coeffs(num_An_terms, collocation_phi_points, 
                        alpha_wing_rad, alpha_L0_section_rad, 
                        c_root, taper_ratio, semi_span, section_lift_slope_a0):
    """Solves for Fourier coefficients An w/ Monoplane Equation."""
    
    M = num_An_terms
    n_indices = np.array([2 * i - 1 for i in range(1, M + 1)]) # Odd indices

    matrix_coeffs = np.zeros((M, M))
    rhs_vector = np.zeros(M)

    for j in range(M):  # For each value phi
        phi_j = collocation_phi_points[j]
        _ , mu_j = get_local_chord_and_mu(phi_j, c_root, taper_ratio, 
                                          semi_span, section_lift_slope_a0)
        
        # Calculate mu side
        rhs_vector[j] = mu_j * (alpha_wing_rad - alpha_L0_section_rad) * np.sin(phi_j)

        # Calculate A_n side
        for k in range(M):
            n_k_val = n_indices[k]
            matrix_coeffs[j, k] = np.sin(n_k_val * phi_j) * (n_k_val * mu_j + np.sin(phi_j))
    
    # Solve the linear system
    An_vector = np.linalg.solve(matrix_coeffs, rhs_vector)
        
    return An_vector, n_indices

def calculate_finite_wing_coeffs(An_vector, n_indices, AR):
    """Calculates wing CL and CDi from An"""

    A1 = An_vector[0]
    CL_wing = A1 * np.pi * AR

    CDi_wing_sum_term = np.sum(n_indices * (An_vector**2))
    CDi_wing = np.pi * AR * CDi_wing_sum_term
    
    return CL_wing, CDi_wing

def get_spanwise_Cl_data(An_vector, n_indices, c_root, taper_ratio, semi_span, num_plot_points):
    """Calculates local section Cl for plotting. Returns y/s and Cl_local."""
    
    phi_for_plot = np.linspace(0, np.pi, num_plot_points)
    y_over_s_plot = -np.cos(phi_for_plot) 

    Gamma_norm_plot = np.zeros_like(phi_for_plot)
    for i in range(len(An_vector)):
        Gamma_norm_plot += An_vector[i] * np.sin(n_indices[i] * phi_for_plot)
            
    local_chord_plot = c_root * (1.0 - (1.0 - taper_ratio) * np.abs(np.cos(phi_for_plot)))
    
    Cl_local_plot = (8 * semi_span * Gamma_norm_plot) / local_chord_plot
    
    # Ensure Cl is zero at tips if Gamma_norm is zero
    Cl_local_plot[np.isclose(Gamma_norm_plot, 0.0)] = 0.0

    return y_over_s_plot, Cl_local_plot

# Main
def main():
    # Adjustable Input Parameters
    AR_input = 7.0                                     
    taper_ratios_input = np.array([0.4, 0.5, 0.6, 1.0]) 
    c_root_input = 1.0                                 
    section_lift_slope_a0_input = 2 * np.pi            
    num_An_terms_input = 4
    alpha_wing_deg_input = 4.0                         
    alpha_L0_section_deg_input = 0    
    num_plot_points = 100             

    alpha_wing_rad = np.deg2rad(alpha_wing_deg_input)
    alpha_L0_section_rad = np.deg2rad(alpha_L0_section_deg_input)

    calculated_CLs = []
    calculated_CDis = []
    spanwise_data_all_tapers = []

    print(f"VECTORRRRRR (OHHHHHH YEAAAAAHHHHH!!!!)")
    print(f"AR: {AR_input}, Root Chord: {c_root_input}, Section a0: {section_lift_slope_a0_input / np.pi:.2f}*pi")
    print(f"Alpha: {alpha_wing_deg_input} deg, Section alpha_L0: {alpha_L0_section_deg_input} deg")
    print(f"Number of A_n terms: {num_An_terms_input}")
    
    # Collocation points phi_j for solving A_n (M points in (0, pi/2])
    collocation_phi_points = np.array([j * np.pi / (2.0 * num_An_terms_input) 
                                       for j in range(1, num_An_terms_input + 1)])

    for current_taper_ratio in taper_ratios_input:
        
        _ , _ , semi_span_s, _ = calculate_wing_geometry(c_root_input, 
                                                          current_taper_ratio, 
                                                          AR_input)

        An_coeffs, n_indices_vals = solve_for_An_coeffs(num_An_terms_input, collocation_phi_points, 
                                                        alpha_wing_rad, alpha_L0_section_rad, 
                                                        c_root_input, current_taper_ratio, 
                                                        semi_span_s, section_lift_slope_a0_input)
        
        CL_wing, CDi_wing = calculate_finite_wing_coeffs(An_coeffs, n_indices_vals, AR_input)
        
        calculated_CLs.append(CL_wing)
        calculated_CDis.append(CDi_wing)
        
        y_s_plot_data, Cl_local_data = get_spanwise_Cl_data(
                                             An_coeffs, n_indices_vals,
                                             c_root_input, current_taper_ratio, semi_span_s,
                                             num_plot_points)
        spanwise_data_all_tapers.append((y_s_plot_data, Cl_local_data, current_taper_ratio))

    # Plotting
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig) # Create a 2x2 grid
    
    fig.suptitle(f"Monoplane Equation Results (AR={AR_input}, $\\alpha=${alpha_wing_deg_input}°, $\\alpha_{{L0,sec}}=${alpha_L0_section_deg_input}°)", fontsize=16)

    ax1 = fig.add_subplot(gs[0, 0]) # Top-left
    ax1.plot(taper_ratios_input, calculated_CLs, marker='o', linestyle='-')
    ax1.set_xlabel("Taper Ratio ($\lambda$)")
    ax1.set_ylabel("Wing Lift Coefficient ($C_L$)")
    ax1.set_title("$C_L$ vs. Taper Ratio")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1]) # Top-right
    ax2.plot(taper_ratios_input, calculated_CDis, marker='s', linestyle='-')
    ax2.set_xlabel("Taper Ratio ($\lambda$)")
    ax2.set_ylabel("Wing Induced Drag Coeff. ($C_{D,i}$)")
    ax2.set_title("$C_{D,i}$ vs. Taper Ratio")
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, :]) # Bottom
    for y_s_data, Cl_data, lambda_val in spanwise_data_all_tapers:
        if y_s_data.size > 1 : 
             ax3.plot(y_s_data, Cl_data, label=f"$\lambda = {lambda_val:.1f}$")
    ax3.set_xlabel("Normalized Spanwise Location ($y/s$)")
    ax3.set_ylabel("Local Section Lift Coefficient ($C_l(y)$)")
    ax3.set_title("Spanwise Local Lift Coefficient Distribution")
    ax3.legend(loc='best')
    ax3.grid(True)
    ax3.set_xlim([-1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout, leave space for subtitle
    plt.show()

if __name__ == '__main__':
    main()
