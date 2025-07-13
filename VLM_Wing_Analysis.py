import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Functions

def calculate_panel_geometry(c_root, taper_ratio, AR, dihedral_deg, N_spanwise, M_chordwise, wing_sweep_deg):
    # Calculates panel geometry. N_spanwise per semi-span. M_chordwise fixed to 1.
    if M_chordwise != 1:
        M_chordwise = 1 

    dihedral_rad = np.deg2rad(dihedral_deg)
    sweep_rad = np.deg2rad(wing_sweep_deg) 

    c_tip = taper_ratio * c_root
    wing_span_b = AR * c_root * (1 + taper_ratio) / 2.0
    semi_span_s = wing_span_b / 2.0
    wing_area_S = (c_root + c_tip) / 2.0 * wing_span_b

    panels = [] 
    y_stations_semispan = np.linspace(0, semi_span_s, N_spanwise + 1)

    for i in range(N_spanwise): 
        y_inboard_flat = y_stations_semispan[i]
        y_outboard_flat = y_stations_semispan[i+1]
        
        chord_inboard = c_root - (c_root - c_tip) * (y_inboard_flat / semi_span_s)
        chord_outboard = c_root - (c_root - c_tip) * (y_outboard_flat / semi_span_s)

        x_le_inboard = y_inboard_flat * np.tan(sweep_rad)
        x_le_outboard = y_outboard_flat * np.tan(sweep_rad)

        P1_flat = np.array([x_le_inboard, y_inboard_flat, 0.0])
        P2_flat = np.array([x_le_outboard, y_outboard_flat, 0.0])
        P3_flat = np.array([x_le_outboard + chord_outboard, y_outboard_flat, 0.0])
        P4_flat = np.array([x_le_inboard + chord_inboard, y_inboard_flat, 0.0])
        
        panel_corners_dihedral = []
        for p_flat in [P1_flat, P2_flat, P3_flat, P4_flat]:
            y_flat_coord = p_flat[1]
            z_coord_dihedral = y_flat_coord * np.sin(dihedral_rad)
            y_coord_dihedral_proj = y_flat_coord * np.cos(dihedral_rad)
            panel_corners_dihedral.append(np.array([p_flat[0], y_coord_dihedral_proj, z_coord_dihedral]))
            
        P1, P2, P3, P4 = panel_corners_dihedral

        bv_A = P1 + 0.25 * (P4 - P1) 
        bv_B = P2 + 0.25 * (P3 - P2) 
        bound_vortex_center_y = (bv_A[1] + bv_B[1]) / 2.0 

        cp_A = P1 + 0.75 * (P4 - P1) 
        cp_B = P2 + 0.75 * (P3 - P2) 
        control_point = (cp_A + cp_B) / 2.0

        vec1_diag = P2 - P1 
        vec2_diag = P4 - P1 
        cross_prod_normal = np.cross(vec1_diag, vec2_diag) 
        norm_magnitude = np.linalg.norm(cross_prod_normal)
        if norm_magnitude < 1e-9: 
            normal_vector = np.array([0.,0.,1.])
        else:
            normal_vector = cross_prod_normal / norm_magnitude
        if normal_vector[2] < 0: 
            normal_vector = -normal_vector
            
        panel_info = {
            'id': i, 'corners': [P1, P2, P3, P4], 'bound_vortex_A': bv_A, 
            'bound_vortex_B': bv_B, 'bound_vortex_center_y': bound_vortex_center_y, 
            'control_point': control_point, 'normal_vector': normal_vector,
            'area': 0.5 * np.linalg.norm(np.cross(P3-P1, P4-P2)) 
        }
        panels.append(panel_info)
    return panels, wing_area_S, semi_span_s, wing_span_b


def induced_velocity_by_finite_vortex_segment(P, A, B, Gamma=1.0):
    # Calculates velocity at P induced by vortex segment A-B.
    r_AP = P - A
    r_BP = P - B
    norm_r_AP = np.linalg.norm(r_AP)
    norm_r_BP = np.linalg.norm(r_BP)
    
    r_AP_cross_r_BP = np.cross(r_AP, r_BP)
    norm_r_AP_cross_r_BP_sq = np.dot(r_AP_cross_r_BP, r_AP_cross_r_BP)

    if norm_r_AP_cross_r_BP_sq < 1e-12 or norm_r_AP < 1e-9 or norm_r_BP < 1e-9:
        return np.array([0., 0., 0.])

    r_AB = B - A
    term1_val = np.dot(r_AB, r_AP) / norm_r_AP
    term2_val = np.dot(r_AB, r_BP) / norm_r_BP
    scalar_factor = term1_val - term2_val
    
    induced_vel = (Gamma / (4 * np.pi)) * (scalar_factor / norm_r_AP_cross_r_BP_sq) * r_AP_cross_r_BP
    return induced_vel


def calculate_horseshoe_velocity(P_eval, bv_A, bv_B, V_inf_dir_unit, Gamma=1.0, far_field_factor=1000):
    # Calculates velocity at P_eval induced by a horseshoe vortex.
    approx_span_of_segment = np.linalg.norm(bv_B - bv_A)
    span_ref_for_far_field = approx_span_of_segment * 10 if approx_span_of_segment > 1e-6 else 1.0
    P_far_downstream = V_inf_dir_unit * far_field_factor * span_ref_for_far_field

    v_bound = induced_velocity_by_finite_vortex_segment(P_eval, bv_A, bv_B, Gamma)
    v_trail_B = induced_velocity_by_finite_vortex_segment(P_eval, bv_B, bv_B + P_far_downstream, Gamma)
    v_trail_A = induced_velocity_by_finite_vortex_segment(P_eval, bv_A + P_far_downstream, bv_A, Gamma) 
    return v_bound + v_trail_B + v_trail_A


def solve_vlm(panels, V_infinity_freestream, alpha_wing_rad):
    # Builds and solves [AIC]{Gamma} = {RHS}.
    num_panels = len(panels)
    AIC_matrix = np.zeros((num_panels, num_panels))
    RHS_vector = np.zeros(num_panels)

    V_inf_vector_global = V_infinity_freestream * np.array([np.cos(alpha_wing_rad), 0, np.sin(alpha_wing_rad)])
    V_inf_dir_unit = V_inf_vector_global / (np.linalg.norm(V_inf_vector_global) + 1e-9) # Add epsilon for safety

    for i in range(num_panels): 
        panel_i = panels[i]
        cp_i = panel_i['control_point']
        normal_i = panel_i['normal_vector']
        RHS_vector[i] = -np.dot(V_inf_vector_global, normal_i)
        
        for j in range(num_panels): 
            panel_j = panels[j]
            vel_induced_by_horseshoe_j = calculate_horseshoe_velocity(
                cp_i, panel_j['bound_vortex_A'], panel_j['bound_vortex_B'], 
                V_inf_dir_unit, Gamma=1.0
            )
            AIC_matrix[i, j] = np.dot(vel_induced_by_horseshoe_j, normal_i)
    try:
        Gamma_strengths = np.linalg.solve(AIC_matrix, RHS_vector)
    except np.linalg.LinAlgError:
        Gamma_strengths = np.zeros(num_panels)
    return Gamma_strengths


def calculate_forces_coeffs_VLM(panels, Gamma_strengths, V_infinity_freestream, alpha_wing_rad, 
                               rho_air, S_wing, AR, V_inf_dir_unit):
    # Calculates Lift, Induced Drag, and coefficients.
    num_panels = len(panels)
    total_force_vector = np.array([0., 0., 0.])
    V_inf_vector_global = V_infinity_freestream * np.array([np.cos(alpha_wing_rad), 0, np.sin(alpha_wing_rad)])

    for i in range(num_panels):
        panel_i = panels[i]
        Gamma_i = Gamma_strengths[i]
        bound_vortex_vector_l = panel_i['bound_vortex_B'] - panel_i['bound_vortex_A']
        bound_vortex_center = (panel_i['bound_vortex_A'] + panel_i['bound_vortex_B']) / 2.0
        
        v_induced_at_bv_center = np.array([0., 0., 0.])
        for j in range(num_panels):
            if i == j: continue
            panel_j = panels[j]
            v_induced_at_bv_center += calculate_horseshoe_velocity(
                bound_vortex_center, panel_j['bound_vortex_A'], panel_j['bound_vortex_B'], 
                V_inf_dir_unit, Gamma=Gamma_strengths[j]
            )
        V_local_at_bv = V_inf_vector_global + v_induced_at_bv_center
        force_on_segment = rho_air * Gamma_i * np.cross(V_local_at_bv, bound_vortex_vector_l)
        total_force_vector += force_on_segment
            
    total_force_vector *= 2.0 # Double for full wing from semi-span calculation

    lift_total = total_force_vector[2] * np.cos(alpha_wing_rad) - total_force_vector[0] * np.sin(alpha_wing_rad)
    drag_induced = total_force_vector[0] * np.cos(alpha_wing_rad) + total_force_vector[2] * np.sin(alpha_wing_rad)

    q_inf = 0.5 * rho_air * V_infinity_freestream**2
    CL = lift_total / (q_inf * S_wing) if S_wing > 1e-9 else 0
    CDi = drag_induced / (q_inf * S_wing) if S_wing > 1e-9 else 0
    
    return CL, CDi

# Main
def main():
    # Input Parameters
    AR_values_input = np.array([5.0, 8.0]) # Vector of Aspect Ratios
    taper_ratio_val = 1
    dihedral_angle_deg_val = 0
    wing_sweep_deg_val = 45    
    N_spanwise_panels_semispan = 4 
    M_chordwise_panels = 1       
    c_root_val = 1.0 
    V_inf_val = 50.0  
    alpha_main_deg_val = 5.0 # Primary AoA for CL/CDi vs AR plots
    rho_air_val = 1.225 
    
    # Alpha range for CL vs Alpha plot
    alpha_range_for_Cl_plot_deg = np.linspace(-4, 12, 17) # e.g., -4 to 12 degrees, 17 points

    print("--- VLM Analysis Inputs ---")
    print(f"Taper: {taper_ratio_val:.2f}, Root Chord: {c_root_val:.2f}m, Dihedral: {dihedral_angle_deg_val:.1f}deg, Sweep: {wing_sweep_deg_val:.1f}deg")
    print(f"N_spanwise (semi): {N_spanwise_panels_semispan}, M_chordwise: {M_chordwise_panels}")
    print(f"V_inf: {V_inf_val:.1f}m/s, Rho: {rho_air_val:.3f}kg/m^3, Primary Alpha: {alpha_main_deg_val:.1f}deg")

    results_for_AR_plots = {'AR': [], 'CL': [], 'CDi': []}
    results_for_CL_alpha_plot = {} # Dict to store {AR: {'alphas': [], 'CLs': []}}

    for current_AR in AR_values_input:
        alpha_main_rad_val = np.deg2rad(alpha_main_deg_val)
        V_inf_dir_unit_main_alpha = np.array([np.cos(alpha_main_rad_val), 0, np.sin(alpha_main_rad_val)])
        V_inf_dir_unit_main_alpha /= (np.linalg.norm(V_inf_dir_unit_main_alpha) + 1e-9)


        panels, S_wing_total, semi_span_s, wing_span_b_total = calculate_panel_geometry(
            c_root_val, taper_ratio_val, current_AR, dihedral_angle_deg_val,
            N_spanwise_panels_semispan, M_chordwise_panels, wing_sweep_deg_val
        )
        if not panels or S_wing_total < 1e-9: 
            results_for_AR_plots['AR'].append(current_AR)
            results_for_AR_plots['CL'].append(np.nan)
            results_for_AR_plots['CDi'].append(np.nan)
            results_for_CL_alpha_plot[current_AR] = {'alphas_deg': alpha_range_for_Cl_plot_deg, 'CLs': np.full_like(alpha_range_for_Cl_plot_deg, np.nan)}
            continue
        
        # Calculate for CL vs AR and CDi vs AR plots (at primary alpha)
        Gamma_strengths_main = solve_vlm(panels, V_inf_val, alpha_main_rad_val)
        CL_main, CDi_main = calculate_forces_coeffs_VLM(
            panels, Gamma_strengths_main, V_inf_val, alpha_main_rad_val, 
            rho_air_val, S_wing_total, current_AR, V_inf_dir_unit_main_alpha
        )
        results_for_AR_plots['AR'].append(current_AR)
        results_for_AR_plots['CL'].append(CL_main)
        results_for_AR_plots['CDi'].append(CDi_main)
        print(f"  For alpha={alpha_main_deg_val:.1f}deg: CL={CL_main:.4f}, CDi={CDi_main:.5f}")

        # Calculate for CL vs alpha plot
        alphas_rad_for_plot = np.deg2rad(alpha_range_for_Cl_plot_deg)
        cls_for_this_AR = []
        for alpha_test_rad in alphas_rad_for_plot:
            V_inf_dir_unit_test_alpha = np.array([np.cos(alpha_test_rad), 0, np.sin(alpha_test_rad)])
            V_inf_dir_unit_test_alpha /= (np.linalg.norm(V_inf_dir_unit_test_alpha) + 1e-9)

            Gamma_strengths_test = solve_vlm(panels, V_inf_val, alpha_test_rad)
            CL_test, _ = calculate_forces_coeffs_VLM( # Only need CL here
                panels, Gamma_strengths_test, V_inf_val, alpha_test_rad, 
                rho_air_val, S_wing_total, current_AR, V_inf_dir_unit_test_alpha
            )
            cls_for_this_AR.append(CL_test)
        results_for_CL_alpha_plot[current_AR] = {'alphas_deg': alpha_range_for_Cl_plot_deg, 'CLs': np.array(cls_for_this_AR)}


    # Plotting
    fig = plt.figure(figsize=(15, 8)) # Adjusted for better fit
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3) 
    
    fig.suptitle(f"VLM Results for Swept Wing ($\lambda$={taper_ratio_val:.1f}, Sweep={wing_sweep_deg_val}°, Dihedral={dihedral_angle_deg_val}°)", fontsize=16)

    ax1 = fig.add_subplot(gs[0, 0]) # Top-left
    ax1.plot(results_for_AR_plots['AR'], results_for_AR_plots['CL'], marker='o', linestyle='-')
    ax1.set_xlabel("Aspect Ratio (AR)")
    ax1.set_ylabel("Wing Lift Coefficient ($C_L$)")
    ax1.set_title(f"$C_L$ vs. AR (at $\\alpha=${alpha_main_deg_val}°)")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1]) # Top-right
    ax2.plot(results_for_AR_plots['AR'], results_for_AR_plots['CDi'], marker='s', linestyle='-')
    ax2.set_xlabel("Aspect Ratio (AR)")
    ax2.set_ylabel("Wing Induced Drag Coeff. ($C_{D,i}$)")
    ax2.set_title(f"$C_Di$ vs. AR (at $\\alpha=${alpha_main_deg_val}°)")
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, :]) # Bottom, spanning all columns
    for ar_val, cl_alpha_data in results_for_CL_alpha_plot.items():
        ax3.plot(cl_alpha_data['alphas_deg'], cl_alpha_data['CLs'], marker='.', linestyle='-', label=f"AR = {ar_val:.1f}")
    ax3.set_xlabel("Angle of Attack ($\\alpha$) [degrees]")
    ax3.set_ylabel("Wing Lift Coefficient ($C_L$)")
    ax3.set_title("$C_L$ vs. Angle of Attack for Different Aspect Ratios")
    ax3.legend(loc='best')
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
    plt.show()

if __name__ == '__main__':
    main()