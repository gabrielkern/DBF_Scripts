import numpy as np
import matplotlib.pyplot as plt


########################################################################

Speed = 95 # ft/s
term = 100 # number of terms

#Wing
taper = np.array([0.6]) # taper ratio
Cr = 14.25 /12 # root chord inches #############
a0 = 2*np.pi #section lift coefficient
b_input= 60  /12 # span inches
alphain = 3 # angle of attack
alphanolift = -2.1
rho = 0.0023769  # slugs/ft³

# flaps
perc_chord = 20 
perc_span = 60
dClmaxf = 0.60 #based on deflection

# ailerons  rest of span not flapped
perc_chord = 20
dClmaxa = 0.1

##########################################################################

fig, axs = plt.subplots(1,1, figsize=(15, 8.8))  # 1 rows, 1 column of subplots

# Gabes Code

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

def calculate_finite_wing_coeffs(An_vector, n_indices, AR):
    """Calculates wing CL and CDi from An"""
    A1 = An_vector[0]
    CL_wing = A1 * np.pi * AR
    CDi_wing_sum_term = np.sum(n_indices * (An_vector**2))
    CDi_wing = np.pi * AR * CDi_wing_sum_term

    return CL_wing, CDi_wing

alpha_wing_rad = np.deg2rad(alphain)
alpha_L0_section_rad = np.deg2rad(alphanolift)

calculated_CLs = []
calculated_CDis = []
spanwise_data_all_tapers = []
U_neccesary =[]

    # Collocation points phi_j
collocation_phi_points = np.array([j * np.pi / (2.0 * term)
                                       for j in range(1, term + 1)])

for current_taper_ratio in taper:
        AspectR = b_input / ( Cr* (1+current_taper_ratio)/2 )
        _, _, semi_span_s, _ = calculate_wing_geometry(Cr,
                                                       current_taper_ratio,
                                                       AspectR)

        An_coeffs, n_indices_vals = solve_for_An_coeffs(term, collocation_phi_points,
                                                        alpha_wing_rad, alpha_L0_section_rad,
                                                        Cr, current_taper_ratio,
                                                        semi_span_s, a0)
        CL_wing, CDi_wing = calculate_finite_wing_coeffs(An_coeffs, n_indices_vals, AspectR)


        #print(CDi_wing)
        y_s_plot_data, Cl_local_data = get_spanwise_Cl_data(
                                             An_coeffs, n_indices_vals,
                                             Cr, current_taper_ratio, semi_span_s,
                                             term*2)

        full_y = y_s_plot_data * b_input/2 
        C_y = Cr * ( 1- 2*abs(full_y)/b_input * (1-current_taper_ratio))
        Intboy = np.zeros_like(C_y)
        for r in range(len(Cl_local_data)):
            Intboy[r] = C_y[r] * Cl_local_data[r]
        #print(Intboy)

        S = ((Cr + Cr*current_taper_ratio)*b_input/2)

        # solve for shear from lift
        LiftForce = np.zeros_like(C_y)
        LiftForcef = np.zeros_like(C_y)
       
        for index in range(len(C_y)-1):
            LiftForce[index] = 1/2 * rho * Speed**2 * (C_y[index] * Cl_local_data[index]+C_y[index+1] * Cl_local_data[index+1])/2 * abs(full_y[index]-full_y[index+1])
        
        flap_Cl = np.zeros_like(Cl_local_data)
        flap_len = b_input/2 * perc_span / 100
        
        for i in range(len(full_y)):
            flap_Cl[i] = (abs(full_y[i])< flap_len)*(Cl_local_data[i] + dClmaxf) + (abs(full_y[i]) > flap_len)*(Cl_local_data[i] + dClmaxa)
        
        for index in range(len(C_y)-1):
            LiftForcef[index] = 1/2 * rho * Speed**2 * (C_y[index] * flap_Cl[index]+C_y[index+1] * flap_Cl[index+1])/2 * abs(full_y[index]-full_y[index+1])

        
        Lift_noflap = np.sum(LiftForce)
        Lift_flap = np.sum(LiftForcef)

        axs.plot(full_y, Cl_local_data, label=f"Cl No Flap ")
        axs.plot(full_y, LiftForce, label=f"Lift Force No Flap")
        axs.plot(full_y, flap_Cl, label=f"Cl No Flap")
        axs.plot(full_y, LiftForcef, label=f"Lift Force No Flap")
        axs.legend(loc='best')
        axs.grid(True)
        axs.set_xlabel("Distance from the Center (ft)")
        axs.set_ylabel("Cl and Lift Force (lbs)")
        axs.set_xlim(right=2.5)
        axs.set_xlim(left=-2.5)
        # Add total lift annotations in the plot
        axs.text(0.05, 0.95,
         f"Total Lift No Flap = {Lift_noflap:.1f} lbs\nTotal Lift With Flap = {Lift_flap:.1f} lbs",
         transform=axs.transAxes,
         fontsize=10, va="top", ha="left",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.show()
