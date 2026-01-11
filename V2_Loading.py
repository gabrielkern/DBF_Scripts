import numpy as np
import matplotlib.pyplot as plt


########################################################################

Acc = 606 # number of points do not touch causes problems with moment
WT = 16.075

#Fuelselage
Wb = 2 # body weight lbs
nw = 10  # g load
qload = [0.25,.035795] # pounds per inch
qplacement = [10,13,13,24] # start,end first is pucks second ducks
placedlocations = [3,8,10] # location of motor,esc,battery
placedweights = [0.75,0.325,1.5] # weights corresponding to above
# x=0 at nose
cg = 7.5 # location of cg
w = 7.25 # quarter chord positions
L = 39 # total length

#Wing
Ww = 2 # wing weight in pounds
taper = np.array([0.6]) # taper ratio
Cr = 13.75 /12 # root chord inches #############
a0 = 2*np.pi #section lift coefficient
b_input= 36  /12 # span inches
B_width = 3.25  /12# body width in inches
term = 100 # number of terms
alphain = 3 # angle of attack
alphanolift = -2.1
rho = 0.0023769  # slugs/ft³

#Spar
max_height = 1.08
min_height = 0.648
thickness_guess = np.linspace(0.0625,5,50000)
Ebass = 1.3*10**6 #psi
Ebalsa = 4.4*10**5 #psi
basst = 0.125 #inches
sigma_bass = 1250 #psi
sigma_balsa = 1000 #psi per
shear_balsa = 800 #psi
FOS = 1.75 # factor of safety 
n = Ebass/Ebalsa
height_left= np.linspace(min_height,max_height,int((1313-1)/2)) # inches
height_left_right = np.flipud(height_left)
height = np.concatenate((height_left, [max_height], height_left_right))

########################################################################

# Fuelselage

# distributed load segments (pairs)
q_pairs = [(qplacement[i], qplacement[i+1]) for i in range(0, len(qplacement), 2)]

# distributed load totals & centroids
dist_totals = [q * (end - start) for q, (start, end) in zip(qload, q_pairs)]
dist_centroids = [(start + end) / 2.0 for (start, end) in q_pairs]

# point loads (downward = negative, scaled by g-load)
point_forces = [-wt * nw for wt in placedweights]   # placed weights
point_positions = placedlocations.copy()
point_forces.append(-Wb * nw)   # body weight
point_positions.append(cg)

# distributed loads as equivalent point loads
dist_forces_signed = [-f * nw for f in dist_totals]

# solve reactions at wing (x=w) and tail (x=L)
A = np.array([[1.0, 1.0],
              [w,   L  ]], dtype=float)
b = np.array([
    - (sum(point_forces) + sum(dist_forces_signed)),
    - (sum([f * p for f, p in zip(point_forces, point_positions)]) +
       sum([f * c for f, c in zip(dist_forces_signed, dist_centroids)]))
], dtype=float)
R_w, R_L = np.linalg.solve(A, b)

# collect all forces
force_positions = point_positions + [w, L]
force_values = point_forces + [R_w, R_L]
sorted_idx = sorted(range(len(force_positions)), key=lambda i: force_positions[i])
force_positions = [force_positions[i] for i in sorted_idx]
force_values = [force_values[i] for i in sorted_idx]

# x grid
x_vals = np.linspace(0.0, L, Acc)
dx = x_vals[1] - x_vals[0]

# shear (upward positive)
shear = np.zeros_like(x_vals)
for pos, val in zip(force_positions, force_values):
    shear += np.where(x_vals >= pos, val, 0.0)

for q_val, (start, end) in zip(qload, q_pairs):
    seg_len = end - start
    shear += -q_val * np.clip(x_vals - start, 0.0, seg_len) * nw

# integrate shear to get moment (lb*in)
moment_in = np.zeros_like(x_vals)
acc = 0.0
for j in range(len(x_vals) - 1):
    acc += 0.5 * (shear[j] + shear[j+1]) * dx
    moment_in[j+1] = acc

# diagnostics
print("Reactions (upward positive):")
print(f" R_w @ {w:.2f} in = {R_w:.3f} lb")
print(f" R_L @ {L:.2f} in = {R_L:.3f} lb")
print("Net force check (should be ~0):", R_w + R_L + sum(point_forces) + sum(dist_forces_signed))
print("Moment at right end (lb*in):", moment_in[-1])

# plotting
fig, axs = plt.subplots(4,1, figsize=(12,8))
axs[0].plot(x_vals/12.0, shear, label='Shear Force', color='royalblue')
axs[0].plot(x_vals/12.0, moment_in/12.0, label='Moment', color='hotpink')
axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
axs[0].set_title("Fuselage Shear / Moment Diagram")
axs[0].set_xlabel("Fuselage Length (ft)")
axs[0].set_ylabel("Shear Force (lb) and Moment (lb/ft)")
axs[0].grid(True)
axs[0].legend()



########################################################################

b = b_input - B_width  ####### span accounting for body being in the middle in feet
y = np.linspace(0,b/2,Acc)
for j in taper:
    C_y = Cr * ( 1- 2*y/b * (1-j))

########################################################################

# Load on wing Span Wise

TotalLoad = (WT) * nw # finds total load of weight at max G value



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
        AspectR = b / ( Cr* (1+current_taper_ratio)/2 )
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
                                             Acc*2)

        full_y = y_s_plot_data * b/2 
        C_y = Cr * ( 1- 2*abs(full_y)/b * (1-current_taper_ratio))
        Intboy = np.zeros_like(C_y)
        for r in range(len(Cl_local_data)):
            Intboy[r] = C_y[r] * Cl_local_data[r]
        #print(Intboy)

        S = ((Cr + Cr*current_taper_ratio)*b/2)
        U_needed = (TotalLoad*2 / rho / np.trapezoid(Intboy,x=full_y) )**0.5  # finds U_inf in ft/s  np.trapz(Intboy,x=full_y)
        #U_neccesary.append(Uneeded)
        print(U_needed/1.467) #prints in mph

        # solve for shear from lift
        LiftForce = np.zeros_like(C_y)
        wing = 0
        wingshear =[]
        for index in range(len(C_y)-1):
            LiftForce[index] = 1/2 * rho * U_needed**2 * (C_y[index] * Cl_local_data[index]+C_y[index+1] * Cl_local_data[index+1])/2 * abs(full_y[index]-full_y[index+1])
        print(sum(LiftForce))
        # moving all points over
        
        for g in range(len(C_y)):
            if full_y[g]<= 0:
                full_y[g] -= B_width/2
            if full_y[g] > 0:
                full_y[g] += B_width/2
        

        
        # creating the body section of the plot
        fill = np.linspace(-B_width/2,B_width/2,101) 
        full_y = np.insert(full_y,Acc,fill) #inserts this into the body

        LiftForce = np.insert(LiftForce,Acc,np.zeros_like(fill))  #inserts zeros for load into the Lift force so weights can be added
        for t in range(len(LiftForce)-2):
                if LiftForce[t+1] == 0:
                    LiftForce[t+1] = -(TotalLoad/nw)/101*nw #distributes wing weight and whole body weight along the contact
        
        for j in range(len(full_y)): 
             wing +=LiftForce[j]
             wingshear.append(wing)
    
        #now solve for moment
        wingmoment =np.zeros_like(full_y)
        val = 0
        for j in range(len(wingshear)-1):
            #if j<= Acc:
                val += 0.5 * b/(Acc*2) * (wingshear[j] + wingshear[j+1])  # trapezoidal integration
            #elif j <= Acc+102:
                #val += 0.5 * B_width/(101) * (wingshear[j] + wingshear[j+1])
            #elif j > Acc+102:
                #val += 0.5 * b/2/(Acc) * (wingshear[j] + wingshear[j+1])

                wingmoment[j+1] = val  # assign to the next point
        


        # finding max bending and shear and finding cross sections of wood to use

        spar_thickness = []
        spar_inertia = []

        for i in range(len(full_y)):
            h = height[i]
            hw = h-basst
            for j in range(len(thickness_guess)):
                thick = thickness_guess[j]
                innerthick = thick/n
                subtract = thick-innerthick
                I = thick * h**3 / 12  - subtract*hw**3 / 12
                Q = h/2*thick * h/4
                shear_max_balsa = wingshear[i]*Q / (I*thick)
                sigma_max_bass = wingmoment[i] * h/2 / I
                sigma_max_balsa = wingmoment[i] * hw/2 / (I*n) 
                principal1_balsa = sigma_max_balsa/2 + ((sigma_max_balsa/2)**2 +shear_max_balsa**2)**(1/2)
                principal2_balsa = sigma_max_balsa/2 - ((sigma_max_balsa/2)**2 +shear_max_balsa**2)**(1/2)
                if  sigma_max_bass <= sigma_bass/FOS and abs(principal1_balsa) <= sigma_balsa/FOS and abs(principal2_balsa) <= sigma_balsa/FOS:
                        break
            spar_thickness.append(thick)

        


        axs[1].plot(full_y, wingshear, label=f"$\\ Shear \\ lambda = {current_taper_ratio:.1f}$")
        axs[1].plot(full_y, wingmoment, label=f"$\\ Moment \\ lambda = {current_taper_ratio:.1f}$")
        axs[1].legend(loc='best')
        axs[1].grid(True)
        axs[1].set_xlabel("Distance from the Center (ft)")
        axs[1].set_ylabel("Shear (lb) and Moment (lbft)")
        #axs[1].set_ylim(bottom=0)
        #axs[1].set_xlim(right=2.5)
        #axs[1].set_xlim(left=-2.5)
        axs[2].plot(full_y, spar_thickness, label=f"$\\ Width \\ lambda = {current_taper_ratio:.1f}$")
        #axs[2].plot(full_y, height, label=f"$\\ Thickness \\ lambda = {current_taper_ratio:.1f}$")
        axs[2].legend(loc='best')
        axs[2].grid(True)
        axs[2].set_xlabel("Distance from the Center (ft)")
        axs[2].set_ylabel("Spar Thickness (in)")
        axs[2].set_ylim(bottom=0)
        #axs[2].set_xlim(right=2.5)
        #axs[2].set_xlim(left=-2.5)
        axs[3].plot(full_y, LiftForce, label=f"$\\ Lift \\ lambda = {current_taper_ratio:.1f}$")
        axs[3].legend(loc='best')
        axs[3].grid(True)
        axs[3].set_xlabel("Distance from the Center (ft)")
        axs[3].set_ylabel("Lift in Pounds")
        axs[3].set_ylim(bottom=0)

plt.tight_layout()
plt.show()
