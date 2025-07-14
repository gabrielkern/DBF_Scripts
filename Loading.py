import numpy as np
import matplotlib.pyplot as plt


########################################################################

#Fuelselage
Wb = 5 # body weight lbs
n = 9  # g load
placedlocations = [ 0.01 , 0.2 , 0.6] # location of motor,esc,battery
placedweights = [1 , 1 , 1] # weights corresponding to above
# x=0 at nose
cg = 0.5 # location of cg
w = 0.5 # quarter chord positions
L = 2 # total length
Acc = 100 # number of points
Bw = 5 # inches this is body width to zero out the lift

#Wing
Ww = 3 # wing weight in pounds
taper = np.array([0.7,0.6,0.8,1]) # taper ratio
Cr = 8 /12 # root chord inches
a0 = 2*np.pi #section lift coefficient
b_input= 60  /12 # span inches
B_width = 5  /12# body width in inches
term = 100 # number of terms
alphain = 3 # angle of attack
alphanolift = -1.6
rho = 0.0023769  # slugs/ft³

########################################################################

# Fuelselage

WT = (Wb+sum(placedweights)) # total load

# solving for the tail force to maintain balance
wtl =[Wb*cg]
for i in range(len(placedlocations)):
    wl = placedlocations[i] * placedweights[i]
    wtl.append(wl)

Ft = n*(WT*w - sum(wtl)) / (L-w) 

# creating weight and location vectors fully
nweights = placedweights + [Wb, -(WT+Ft/n), Ft/n]
nlocations = placedlocations + [cg, w, L]

# Get the sorted indices of x
sorted_indices = sorted(range(len(nlocations)), key=lambda i: nlocations[i])

# Reorder both x and y
locations = [nlocations[i] for i in sorted_indices]
weights = [nweights[i] for i in sorted_indices]


# creating shear and x values vectors
x_vals = np.linspace(0, L, Acc)
shear = np.zeros_like(x_vals)

for j in range(len(x_vals)): 
    for k in range(len(locations)):
        if x_vals[j] >= locations[k]:
            shear[j] = shear[j]+weights[k]*n


# moment calcs
val = 0
moment = np.zeros_like(x_vals)

for j in range(len(moment)-1):
    val += 0.5 * L/Acc * (-shear[j] + -shear[j+1])  # trapezoidal integration
    moment[j+1] = val  # assign to the next point


# Plot shear and moment diagrams

plt.figure(figsize=(10, 4))
plt.plot(x_vals, -shear, label='Shear Force', color='royalblue')
plt.plot(x_vals, moment, label='Moment', color='hotpink')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.title("Fuselage Shear Force and Moment Diagram")
plt.xlabel("Fuselage Length (ft)")
plt.ylabel("Shear Force (lbs) or Moment (lb*in)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


########################################################################

# Visualize Wing Shape

b = b_input - B_width  ####### span accounting for body being in the middle in feet
plt.figure(figsize=(10,4))
y = np.linspace(0,b/2,Acc)
for j in taper:
    C_y = Cr * ( 1- 2*y/b * (1-j))
    
    plt.plot(y,C_y/2, label='Right Wing Dist', color='royalblue')
    plt.plot(-y,C_y/2, label='Left Wing Dist', color='hotpink')
    plt.plot(y,-C_y/2, label='Right Wing Dist', color='royalblue')
    plt.plot(-y,-C_y/2, label='Left Wing Dist', color='hotpink')
    plt.plot([b/2, b/2], [-C_y[-1]/2, C_y[-1]/2], 'k-')
    plt.plot([-b/2, -b/2], [-C_y[-1]/2, C_y[-1]/2], 'k-')

plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.title("Trapezoidal Wing")
plt.xlabel("Distance from Center (in)")
plt.grid(True)
plt.tight_layout()
plt.show()


########################################################################

# Load on wing Span Wise

TotalLoad = (WT+Ww) * n # finds total load of weight at max G value


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


        print(CDi_wing)
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


        U_needed = ((WT+Ww)*n*2 / rho / np.trapz(Intboy,x=full_y) )**0.5  # finds U_inf in ft/s  np.trapz(Intboy,x=full_y)
        #U_neccesary.append(Uneeded)
        print(U_needed/1.467) #prints in mph

        # solve for shear from lift
        LiftForce = np.zeros_like(C_y)
        wing = 0
        wingshear =[]
        for index in range(len(C_y)-1):
            LiftForce[index] = 1/2 * rho * U_needed**2 * (C_y[index] * Cl_local_data[index]+C_y[index+1] * Cl_local_data[index+1])/2 * abs(full_y[index]-full_y[index+1])
        #print(LiftForce)
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
                    LiftForce[t+1] = -(WT+Ww)/102*n #distributes wing weight and whole body weight along the contact

        for j in range(len(full_y)): 
             wing +=LiftForce[j]
             wingshear.append(wing)
    
        #now solve for moment
        wingmoment =np.zeros_like(full_y)
        val = 0
        for j in range(len(wingshear)-1):
            val += 0.5 * b/(2*Acc+101) * (wingshear[j] + wingshear[j+1])  # trapezoidal integration
            wingmoment[j+1] = val  # assign to the next point
                
        plt.plot(full_y,wingshear, label = f"$\\ Shear \\ lambda = {current_taper_ratio:.1f}$")
        plt.plot(full_y,wingmoment, label =f"$\\ Moment \\ lambda = {current_taper_ratio:.1f}$")
        plt.legend()
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel("Distance from the Center")
        plt.ylabel("Shear (lb) and Moment (lbft)")
        plt.tight_layout()
        #print(np.sum(LiftForce))
        spanwise_data_all_tapers.append((y_s_plot_data, Cl_local_data , current_taper_ratio))

plt.show()

'''
plt.figure(figsize=(10, 6))
for y_s_data, Cl_data, lambda_val in spanwise_data_all_tapers:
        if y_s_data.size > 1:
            plt.plot(y_s_data, Cl_data, label=f"$\\lambda = {lambda_val:.1f}$")

plt.xlabel("Normalized Spanwise Location ($y/s$)")
plt.ylabel("Local Section Lift ($C_l(y)$)")
plt.title("Spanwise Local Lift Distribution")
plt.legend(loc='best')
plt.grid(True)
plt.xlim([-1, 1])
plt.tight_layout()
plt.show()
'''