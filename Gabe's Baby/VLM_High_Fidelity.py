import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Needs fixing w/regard to adding dihedral and sweep

# READ BEFORE USE:
# This code does not account for fuselage, viscous effects, or compressibility effects. Use with care

# Geometry generation
def generate_geometry(
    root_chord_length: float,
    span_b: float,
    taper_ratio: float,
    sweep_deg_func,         
    dihedral_deg_func,      
    twist_deg_func,         
    camber_percent_func,    
    N_spanwise_half: int,   
    M_chordwise: int        
    ) -> list:
    """
    Generates 3D panel geometry for a symmetric wing for VLM.
    Sweep, Dihedral, and Twist can vary along the span.
    Returns a list of panel dictionaries for the FULL WING.
    """

    semi_span_s = span_b / 2.0
    if semi_span_s < 1e-9: 
        print("Warning: Semi-span is very small or zero. Returning empty panel list.")
        return [] 

    phi_stations = np.linspace(0, np.pi / 2, N_spanwise_half + 1)
    y_flat_stations = semi_span_s * np.sin(phi_stations)

    x_le_at_stations = np.zeros_like(y_flat_stations)
    z_dih_at_stations = np.zeros_like(y_flat_stations)

    for j in range(N_spanwise_half): 
        y_s_start_seg = y_flat_stations[j] / semi_span_s if semi_span_s > 1e-9 else 0
        y_s_end_seg = y_flat_stations[j+1] / semi_span_s if semi_span_s > 1e-9 else 0
        y_s_mid_seg = (y_s_start_seg + y_s_end_seg) / 2.0
        delta_y_flat = y_flat_stations[j+1] - y_flat_stations[j]
        
        local_sweep_rad = np.deg2rad(sweep_deg_func(y_s_mid_seg))
        local_dihedral_rad = np.deg2rad(dihedral_deg_func(y_s_mid_seg))
        
        x_le_at_stations[j+1] = x_le_at_stations[j] + delta_y_flat * np.tan(local_sweep_rad)
        z_dih_at_stations[j+1] = z_dih_at_stations[j] + delta_y_flat * np.tan(local_dihedral_rad)

    x_local_chord_fractions = np.linspace(0, 1, M_chordwise + 1)
    grid_points_3d_right = np.zeros((N_spanwise_half + 1, M_chordwise + 1, 3))

    for i in range(N_spanwise_half + 1): 
        y_f = y_flat_stations[i]
        y_s_norm = y_f / semi_span_s if semi_span_s > 1e-9 else 0
        local_chord = root_chord_length * (1.0 - (1.0 - taper_ratio) * y_s_norm)
        x_le_current_station = x_le_at_stations[i] 
        z_dih_current_station = z_dih_at_stations[i]
        local_twist_rad = np.deg2rad(twist_deg_func(y_s_norm))

        for k in range(M_chordwise + 1): 
            x_frac = x_local_chord_fractions[k] 
            x_section = x_frac * local_chord 
            z_c_over_c = camber_percent_func(x_frac, y_s_norm) if camber_percent_func else 0.0
            z_section = z_c_over_c * local_chord 

            x_s_twisted = x_section * np.cos(local_twist_rad) - z_section * np.sin(local_twist_rad)
            z_s_twisted = x_section * np.sin(local_twist_rad) + z_section * np.cos(local_twist_rad)

            X_coord = x_le_current_station + x_s_twisted
            Y_coord_projected = y_f * np.cos(np.deg2rad(dihedral_deg_func(y_s_norm))) 
            Z_coord = z_dih_current_station + z_s_twisted 
            grid_points_3d_right[i, k, :] = [X_coord, Y_coord_projected, Z_coord]

    all_panels_construction = [] 
    panel_id_counter = 0

    # Right semi-span panels
    for i in range(N_spanwise_half): # Iterate over N_spanwise_half strips     
        y_s_strip_center_val = ((y_flat_stations[i] + y_flat_stations[i+1]) / 2.0) / semi_span_s if semi_span_s > 1e-9 else 0
        for k in range(M_chordwise):  # Iterate over M_chordwise panels in strip
            P1 = grid_points_3d_right[i,   k,   :] 
            P2 = grid_points_3d_right[i+1, k,   :] 
            P3 = grid_points_3d_right[i+1, k+1, :] 
            P4 = grid_points_3d_right[i,   k+1, :] 
            panel_corners = [P1, P2, P3, P4]
            vortex_A = P1 + 0.25 * (P4 - P1) 
            vortex_B = P2 + 0.25 * (P3 - P2) 
            cp_inboard_edge = P1 + 0.75 * (P4 - P1)
            cp_outboard_edge = P2 + 0.75 * (P3 - P2)
            control_point = (cp_inboard_edge + cp_outboard_edge) / 2.0
            diag1 = P3 - P1; diag2 = P4 - P2
            panel_normal_vec = np.cross(diag1, diag2)
            norm_mag = np.linalg.norm(panel_normal_vec)
            if norm_mag < 1e-9: panel_normal_vec = np.array([0,0,1])
            else: panel_normal_vec = panel_normal_vec / norm_mag
            if panel_normal_vec[2] < 0: panel_normal_vec = -panel_normal_vec
            
            all_panels_construction.append({
                'id_orig_right': panel_id_counter, 'side': 'right', 'corners': panel_corners,
                'vortex_A': vortex_A, 'vortex_B': vortex_B,
                'control_point': control_point, 'normal_vector': panel_normal_vec,
                'area': 0.5 * norm_mag,
                'local_chord_approx': (np.linalg.norm(P4-P1) + np.linalg.norm(P3-P2))/2.0,
                'spanwise_center_y': control_point[1],
                'y_s_strip_center': y_s_strip_center_val # For grouping chordwise panels
            })
            panel_id_counter += 1
    
    all_panels_final = []
    left_panels_temp = []
    # Mirror right panels to create left panels. Iterate to maintain strip structure.
    for i in range(N_spanwise_half):
        y_s_strip_center_val = ((y_flat_stations[i] + y_flat_stations[i+1]) / 2.0) / semi_span_s if semi_span_s > 1e-9 else 0
        for k in range(M_chordwise):
            # Find corresponding right panel (more robustly than relying on reversed all_panels_construction)
            # This assumes all_panels_construction is ordered spanwise then chordwise for right wing
            right_panel_idx = i * M_chordwise + k
            p_right = all_panels_construction[right_panel_idx]

            P1_r, P2_r, P3_r, P4_r = p_right['corners']
            P1_l = np.array([P1_r[0], -P1_r[1], P1_r[2]])
            P2_l = np.array([P2_r[0], -P2_r[1], P2_r[2]])
            P3_l = np.array([P3_r[0], -P3_r[1], P3_r[2]])
            P4_l = np.array([P4_r[0], -P4_r[1], P4_r[2]])
            panel_corners_l = [P1_l, P2_l, P3_l, P4_l]
            
            v_A_r = p_right['vortex_A']; v_B_r = p_right['vortex_B'] 
            vortex_A_l = np.array([v_B_r[0], -v_B_r[1], v_B_r[2]]) 
            vortex_B_l = np.array([v_A_r[0], -v_A_r[1], v_A_r[2]]) 
            
            control_point_l = np.array([p_right['control_point'][0], -p_right['control_point'][1], p_right['control_point'][2]])
            n_r = p_right['normal_vector']
            panel_normal_vec_l = np.array([n_r[0], -n_r[1], n_r[2]]) 
            
            left_panels_temp.append({
                'id': 0, 'side': 'left', 'corners': panel_corners_l, 
                'vortex_A': vortex_A_l, 'vortex_B': vortex_B_l,
                'control_point': control_point_l, 'normal_vector': panel_normal_vec_l,
                'area': p_right['area'],
                'local_chord_approx': p_right['local_chord_approx'],
                'spanwise_center_y': control_point_l[1],
                'y_s_strip_center': -y_s_strip_center_val # Mirrored y/s for strip center
            })

    # Combine panels: left tip to root, then right root to tip
    all_panels_final.extend(list(reversed(left_panels_temp))) 
    all_panels_final.extend(all_panels_construction)
    
    for i, panel in enumerate(all_panels_final):
        panel['id'] = i
    return all_panels_final

# VLM Calcs
def induced_velocity_by_finite_vortex_segment(P_eval, A, B, Gamma=1.0):
    r_AP = P_eval - A; r_BP = P_eval - B
    norm_r_AP = np.linalg.norm(r_AP); norm_r_BP = np.linalg.norm(r_BP)
    r_AP_cross_r_BP = np.cross(r_AP, r_BP)
    norm_r_AP_cross_r_BP_sq = np.dot(r_AP_cross_r_BP, r_AP_cross_r_BP)
    if norm_r_AP_cross_r_BP_sq < 1e-12 or norm_r_AP < 1e-9 or norm_r_BP < 1e-9:
        return np.array([0., 0., 0.])
    r_AB = B - A 
    scalar_factor = np.dot(r_AB, r_AP) / norm_r_AP - np.dot(r_AB, r_BP) / norm_r_BP
    return (Gamma / (4 * np.pi)) * (scalar_factor / (norm_r_AP_cross_r_BP_sq + 1e-12) ) * r_AP_cross_r_BP

def calculate_horseshoe_velocity(P_eval, bv_A, bv_B, V_inf_dir_unit, Gamma=1.0, far_field_factor=1000):
    segment_length = np.linalg.norm(bv_B - bv_A)
    far_distance = far_field_factor * (segment_length if segment_length > 1e-6 else 1.0)
    A_far = bv_A + V_inf_dir_unit * far_distance 
    B_far = bv_B + V_inf_dir_unit * far_distance 
    v_bound = induced_velocity_by_finite_vortex_segment(P_eval, bv_A, bv_B, Gamma)
    v_trail_B = induced_velocity_by_finite_vortex_segment(P_eval, bv_B, B_far, Gamma)
    v_trail_A = induced_velocity_by_finite_vortex_segment(P_eval, A_far, bv_A, Gamma) 
    return v_bound + v_trail_A + v_trail_B

def solve_vortex_strengths(all_panels_input, V_infinity_freestream, alpha_wing_rad):
    num_total_panels = len(all_panels_input)
    if num_total_panels == 0: return np.array([])
    AIC_matrix = np.zeros((num_total_panels, num_total_panels))
    RHS_vector = np.zeros(num_total_panels)
    V_inf_vector_global = V_infinity_freestream * np.array([np.cos(alpha_wing_rad), 0, np.sin(alpha_wing_rad)])
    V_inf_dir_unit = V_inf_vector_global / (np.linalg.norm(V_inf_vector_global) + 1e-9)

    for i in range(num_total_panels): 
        panel_i = all_panels_input[i]
        RHS_vector[i] = -np.dot(V_inf_vector_global, panel_i['normal_vector'])
        for j in range(num_total_panels): 
            panel_j = all_panels_input[j]
            vel_induced = calculate_horseshoe_velocity(
                panel_i['control_point'], panel_j['vortex_A'], panel_j['vortex_B'], 
                V_inf_dir_unit, Gamma=1.0
            )
            AIC_matrix[i, j] = np.dot(vel_induced, panel_i['normal_vector'])
    try:
        Gamma_strengths = np.linalg.solve(AIC_matrix, RHS_vector)
    except np.linalg.LinAlgError:
        print(f"Error: Singular AIC matrix for alpha={np.rad2deg(alpha_wing_rad):.1f} deg. Returning zeros.")
        Gamma_strengths = np.zeros(num_total_panels)
    return Gamma_strengths

def calculate_aerodynamic_forces_and_coeffs(
    all_panels_input, gamma_values, V_infinity_freestream, alpha_wing_rad, 
    rho_air, S_wing_ref, AR_wing_ref, span_b_ref 
    ):
    num_total_panels = len(all_panels_input)
    if num_total_panels == 0 or len(gamma_values) != num_total_panels:
        return 0.0, 0.0, np.array([]), {} # CL, CDi, panel_Cl_values, panel_details_for_avg

    total_force_global_frame = np.array([0., 0., 0.]) 
    V_inf_vector_global = V_infinity_freestream * np.array([np.cos(alpha_wing_rad), 0, np.sin(alpha_wing_rad)])
    V_inf_dir_unit = V_inf_vector_global / (np.linalg.norm(V_inf_vector_global) + 1e-9)

    panel_details_for_avg = {} # Store Cl and y_s_strip_center for averaging

    for i in range(num_total_panels):
        panel_i = all_panels_input[i]
        Gamma_i = gamma_values[i]
        bv_A = panel_i['vortex_A']; bv_B = panel_i['vortex_B']
        bound_vortex_vector_l = bv_B - bv_A 
        bound_vortex_center = (bv_A + bv_B) / 2.0
        
        v_induced_at_bv_center = np.array([0., 0., 0.])
        for j in range(num_total_panels):
            if i == j: continue
            panel_j = all_panels_input[j]
            v_induced_at_bv_center += calculate_horseshoe_velocity(
                bound_vortex_center, panel_j['vortex_A'], panel_j['vortex_B'], 
                V_inf_dir_unit, Gamma=gamma_values[j]
            )
        V_local_at_bv = V_inf_vector_global + v_induced_at_bv_center
        force_on_segment_global = rho_air * Gamma_i * np.cross(V_local_at_bv, bound_vortex_vector_l)
        total_force_global_frame += force_on_segment_global

        panel_chord = panel_i['local_chord_approx']
        cl_panel = 0.0
        if V_infinity_freestream > 1e-6 and panel_chord > 1e-6:
            cl_panel = (2 * Gamma_i) / (V_infinity_freestream * panel_chord)
        
        y_s_strip = panel_i.get('y_s_strip_center', panel_i['spanwise_center_y'] / (span_b_ref/2.0)) # Fallback
        if y_s_strip not in panel_details_for_avg:
            panel_details_for_avg[y_s_strip] = []
        panel_details_for_avg[y_s_strip].append(cl_panel)


    L_total = total_force_global_frame[2]*np.cos(alpha_wing_rad) - total_force_global_frame[0]*np.sin(alpha_wing_rad)
    Di_total = total_force_global_frame[0]*np.cos(alpha_wing_rad) + total_force_global_frame[2]*np.sin(alpha_wing_rad)

    q_inf = 0.5 * rho_air * V_infinity_freestream**2
    CL_wing = L_total / (q_inf * S_wing_ref) if S_wing_ref > 1e-9 and q_inf > 1e-9 else 0
    CDi_wing = Di_total / (q_inf * S_wing_ref) if S_wing_ref > 1e-9 and q_inf > 1e-9 else 0
    
    # For the 3D plot
    all_panel_cl_values = np.zeros(num_total_panels)
    for idx, panel_data_item in enumerate(all_panels_input):
        panel_chord_val = panel_data_item['local_chord_approx']
        gamma_val = gamma_values[idx]
        if V_infinity_freestream > 1e-6 and panel_chord_val > 1e-6:
             all_panel_cl_values[idx] = (2 * gamma_val) / (V_infinity_freestream * panel_chord_val)


    return CL_wing, CDi_wing, all_panel_cl_values, panel_details_for_avg


if __name__ == '__main__':

    # Set parameters
    def twist_func(y_s): return 0.0 #5.0 - 10.0 * y_s  
    def camber_func(x_on_c, y_s): 
        # m = 0.06 - 0.04 * y_s; p = 0.4 
        # if x_on_c <= p: return (m / (p**2 + 1e-9)) * (2 * p * x_on_c - x_on_c**2)
        # else: return (m / ((1 - p)**2 + 1e-9)) * ((1 - 2 * p) + 2 * p * x_on_c - x_on_c**2)
        return 0.02
    def sweep_func(y_s): return 0.0 #25.0 
    def dihedral_func(y_s): return 0.0 #25.0 * y_s**3

    root_chord_val = 1.2564
    span_b_val = 6.0
    taper_ratio_val = 0.7
    V_inf_val = 117.33
    alpha_primary_deg_val = 3.0
    alpha_min_deg_val = -2.0
    alpha_max_deg_val = 12.0
    alpha_separation = 1.0
    rho_air_val = 0.0023
    N_span_val = 10
    M_chord_val = 3 

    # Start analysis
    AR_val = span_b_val**2 / ( (root_chord_val + root_chord_val*taper_ratio_val)/2 * span_b_val ) 
    print(f"AR: {AR_val:.2f}, Taper: {taper_ratio_val:.1f}, Sweep: {sweep_func(0):.0f}deg")

    generated_panels = generate_geometry(
        root_chord_val, span_b_val, taper_ratio_val, sweep_func, dihedral_func,
        twist_func, camber_func, N_span_val, M_chord_val      
    )
    print(f"Generated {len(generated_panels)} total panels.")
    S_wing_ref = (root_chord_val + root_chord_val*taper_ratio_val)/2 * span_b_val
    
    alpha_range_deg = np.arange(alpha_min_deg_val, alpha_max_deg_val + alpha_separation, alpha_separation)
    CL_vs_alpha_list = []; CDi_vs_alpha_list = []

    print("Calculating CL and CDi vs Alpha...")
    for alpha_d in alpha_range_deg:
        alpha_r = np.deg2rad(alpha_d)
        gamma_vals = solve_vortex_strengths(generated_panels, V_inf_val, alpha_r)
        if gamma_vals.size == len(generated_panels):
            cl, cdi, _, _ = calculate_aerodynamic_forces_and_coeffs(
                generated_panels, gamma_vals, V_inf_val, alpha_r,
                rho_air_val, S_wing_ref, AR_val, span_b_val
            )
            CL_vs_alpha_list.append(cl); CDi_vs_alpha_list.append(cdi)
        else:
            CL_vs_alpha_list.append(np.nan); CDi_vs_alpha_list.append(np.nan)
        print(f"  Alpha={alpha_d:.1f} deg: CL={CL_vs_alpha_list[-1]:.4f}, CDi={CDi_vs_alpha_list[-1]:.5f}")

    alpha_primary_rad_val = np.deg2rad(alpha_primary_deg_val)
    gamma_primary = solve_vortex_strengths(generated_panels, V_inf_val, alpha_primary_rad_val)
    
    panel_Cl_primary = np.array([])
    avg_cl_data_primary = {}
    if gamma_primary.size == len(generated_panels):
        _, _, panel_Cl_primary, avg_cl_data_primary = calculate_aerodynamic_forces_and_coeffs(
            generated_panels, gamma_primary, V_inf_val, alpha_primary_rad_val,
            rho_air_val, S_wing_ref, AR_val, span_b_val
        )
    
    # Prepare data for averaged spanwise Cl plot
    plot_y_s_averaged = sorted(avg_cl_data_primary.keys())
    plot_Cl_averaged = [np.mean(avg_cl_data_primary[y_s]) if avg_cl_data_primary.get(y_s) else np.nan for y_s in plot_y_s_averaged]
    
    # Plotting
    fig = plt.figure(figsize=(15, 8)) 
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle(f"VLM Analysis", fontsize=16)

    ax_cdi_alpha = fig.add_subplot(gs[0, 0])
    ax_cdi_alpha.plot(alpha_range_deg, CDi_vs_alpha_list, marker='s', linestyle='-')
    ax_cdi_alpha.set_xlabel("Angle of Attack ($\\alpha$) [degrees]"); ax_cdi_alpha.set_ylabel("$C_{D,i}$")
    ax_cdi_alpha.set_title("$C_{D,i}$ vs. Angle of Attack"); ax_cdi_alpha.grid(True)

    ax_cl_spanwise = fig.add_subplot(gs[0, 1])
    if plot_y_s_averaged and plot_Cl_averaged:
        ax_cl_spanwise.plot(plot_y_s_averaged, plot_Cl_averaged, marker='.', linestyle='-')
    ax_cl_spanwise.set_xlabel("$y/s$"); ax_cl_spanwise.set_ylabel("Avg. Section $C_l$")
    ax_cl_spanwise.set_title(f"Avg. Spanwise $C_l$ (at $\\alpha=${alpha_primary_deg_val}°)"); ax_cl_spanwise.grid(True)
    ax_cl_spanwise.set_xlim([-1, 1])
    ax_cl_spanwise.set_ylim([0, np.max(plot_Cl_averaged) * 1.1 if plot_Cl_averaged else 1.0])

    ax_cl_alpha = fig.add_subplot(gs[1, 0]) 
    ax_cl_alpha.plot(alpha_range_deg, CL_vs_alpha_list, marker='o', linestyle='-')
    ax_cl_alpha.set_xlabel("Angle of Attack ($\\alpha$) [degrees]"); ax_cl_alpha.set_ylabel("$C_L$")
    ax_cl_alpha.set_title("$C_L$ vs. Angle of Attack"); ax_cl_alpha.grid(True)

    ax3d = fig.add_subplot(gs[1, 1], projection='3d')
    if generated_panels and panel_Cl_primary.size == len(generated_panels):
        min_cl_p, max_cl_p = (np.min(panel_Cl_primary), np.max(panel_Cl_primary)) if panel_Cl_primary.size > 0 else (0,1)
        norm_cl_p = (panel_Cl_primary - min_cl_p) / (max_cl_p - min_cl_p + 1e-9) if (max_cl_p - min_cl_p) > 1e-9 else np.zeros_like(panel_Cl_primary) + 0.5
        colors_cl = plt.cm.viridis(norm_cl_p)
        for i, panel in enumerate(generated_panels):
            corners = panel['corners']; plot_corners = np.array([corners[0],corners[1],corners[2],corners[3],corners[0]])
            ax3d.plot(plot_corners[:,0], plot_corners[:,1], plot_corners[:,2], color='darkgrey', alpha=0.3, lw=0.5)
            v_A, v_B = panel['vortex_A'], panel['vortex_B']
            ax3d.plot([v_A[0],v_B[0]], [v_A[1],v_B[1]], [v_A[2],v_B[2]], color=colors_cl[i], lw=2)
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        ax3d.set_title(f"Panel $C_l$ (at $\\alpha=${alpha_primary_deg_val}°)")
        x_coords = np.array([c[0] for p in generated_panels for c in p['corners']])
        y_coords = np.array([c[1] for p in generated_panels for c in p['corners']])
        z_coords = np.array([c[2] for p in generated_panels for c in p['corners']])
        if x_coords.size > 0: 
            xc,yc,zc=np.mean(x_coords),np.mean(y_coords),np.mean(z_coords)
            max_r=0.5*max(np.ptp(x_coords),np.ptp(y_coords),np.ptp(z_coords),1e-9)
            if max_r > 1e-9: ax3d.set_xlim(xc-max_r,xc+max_r); ax3d.set_ylim(yc-max_r,yc+max_r); ax3d.set_zlim(zc-max_r,zc+max_r)
        ax3d.view_init(elev=20., azim=-135)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_cl_p, vmax=max_cl_p))
        sm.set_array([]); fig.colorbar(sm, ax=ax3d, shrink=0.6, aspect=10, label='Local Panel $C_l$')
    else: ax3d.set_title("3D Wing Geometry (No $C_l$ data)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for main title
    plt.show()


    '''
    Perfect! I've completed the analysis of the geometry generation function and identified the key issues. Let me summarize what I found:

  1. Incorrect dihedral calculation: The code uses incremental accumulation with tan(dihedral) instead of proper rigid body rotation
  2. Missing Y-coordinate projection: The Y coordinate projection for dihedral is commented out
  3. Extreme camber value: The 200% camber creates large Z-displacements that compound the problem

  These issues explain why dihedral appears to be added even when dihedral_func returns 0. The flawed implementation can accumulate 
  numerical errors and doesn't properly handle coordinate transformations.

  The user now has a clear understanding of the root cause and the specific lines that need to be fixed to properly implement dihedral 
  geometry.
  '''
