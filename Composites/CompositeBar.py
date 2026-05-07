import numpy as np
import matplotlib.pyplot as plt


######################################################

# Material Properties guess
E1 = 20 *10**6 #psi
E2 = 20 *10**6 #psi
G12 = 0.6 *10**6 #psi
v12 = 0.3 
v21 = v12 * E2 / E1

# Material faulire loads guess
Xt = 200e3
Xc = 150e3
Yt = Xt
Yc = Xc
S12 = 14e3

# Applied loads
Shear = 60
Moment = 30

# Ply List
t = 0.02  # inches, ply thickness
theta_groups = [
    [0,0],   # Group 1: flanges/top
    [45,45]    # Group 2: webs/sides
]

# Spar sizing
height = 1 # inch
width = 0.25 # inch

######################################################

top_thick = len(theta_groups[0])*t
side_thick = len(theta_groups[1])*t
Inertia = 1/12 * (width*height**3) - 1/12 * ( (width-side_thick*2) * (height-top_thick*2)**3) # finds inertia of outer composite bar
Q_area = top_thick*width*(height/2-top_thick/2) + 2*(side_thick*(height/2-top_thick) * (height/2-top_thick)/2)

q = Shear * Q_area / Inertia
Nxy = q*side_thick                         # shear flow is the Nxy
sigma_m = Moment*height/2 / Inertia
Nx = -sigma_m*top_thick         # uses compression

# Each NM vector = [Nx, Ny, Nxy, Mx, My, Mxy]
NM_groups = [
    np.array([Nx, 0, 0, 0, 0, 0]),    # bending for flanges
    np.array([0, 0, Nxy, 0, 0, 0])     # shear for webs
]

def transform_Q(Q, theta_deg):
    theta = np.radians(theta_deg)
    m = np.cos(theta)
    n = np.sin(theta)
    
    Q11 = Q[0,0]; Q12 = Q[0,1]; Q22 = Q[1,1]; Q66 = Q[2,2]
    
    Qbar11 = Q11*m**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*n**4 #
    Qbar12 = (Q11 + Q22 - 4*Q66)*m**2*n**2 + Q12*(m**4 + n**4) #
    Qbar22 = Q11*n**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*m**4 #
    Qbar16 = (Q11 - Q12 - 2*Q66)*m**3*n + (Q12 - Q22 + 2*Q66)*m*n**3 #
    Qbar26 = (Q11 - Q12 - 2*Q66)*m*n**3 + (Q12 - Q22 + 2*Q66)*m**3*n #
    Qbar66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*m**2*n**2 + Q66*(m**4 + n**4) #
    
    Qbar = np.array([
        [Qbar11, Qbar12, Qbar16],
        [Qbar12, Qbar22, Qbar26],
        [Qbar16, Qbar26, Qbar66]
    ])
    
    return Qbar

def global_to_material_strain(eps_global, theta_deg):
    theta = np.radians(theta_deg)
    m = np.cos(theta)
    n = np.sin(theta)
    
    T = np.array([
        [m**2, n**2, m*n],
        [n**2, m**2, -m*n],
        [-2*m*n, 2*m*n, m**2 - n**2]
    ])
    eps_material = T @ eps_global
    return eps_material

def tsai_wu_lambda(s1, s2, t12, XT, XC, YT, YC, S):
    F1  = 1/XT - 1/XC
    F11 = 1/(XT*XC)

    F2  = 1/YT - 1/YC
    F22 = 1/(YT*YC)

    F66 = 1/S**2
    F12 = -1/(2*np.sqrt(XT*XC*YT*YC))

    A = F1*s1 + F2*s2
    B = F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2

    roots = np.roots([B, A, -1])
    roots = roots[np.isreal(roots)]
    roots = roots[roots > 0]
    if len(roots) == 0:
        return np.inf
    else:
        return np.min(roots)

delta = 1-v12*v21
S = np.array([[1/E1 , -v12/E1 , 0 ], [ -v21/E2 , 1/E2 , 0], [0 , 0 , 1/G12]])
Q = np.array([[E1/delta,v12*E2/delta,0],[v12*E2/delta,E2/delta,0],[0,0,G12]])

ply_stresses = []
lam_list_groups = []

for group_idx, theta_list in enumerate(theta_groups):
    print(f"\nProcessing Group {group_idx+1}")

    N_group = len(theta_list)
    # Compute z for this group
    total_thickness = N_group * t
    z_group = [-total_thickness/2 + i*t for i in range(N_group+1)]
    
    # Initialize matrices for this group
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    
    # Compute Qbar, A, B, D for this group
    for k in range(N_group):
        theta = theta_list[k]
        Qbar = transform_Q(Q, theta)
        delta_z = z_group[k+1] - z_group[k]
        A += Qbar * delta_z
        B += 0.5 * Qbar * (z_group[k+1]**2 - z_group[k]**2)
        D += (1/3) * Qbar * (z_group[k+1]**3 - z_group[k]**3)
    
    ABD = np.block([[A,B],[B,D]])
    NM = NM_groups[group_idx]
    strain_curvature = np.linalg.inv(ABD) @ NM
    
    eps0 = strain_curvature[:3]
    kappa = strain_curvature[3:]
    
    # Compute strains and stresses per ply
    lambdas = []
    for k in range(N_group):
        z_mid = (z_group[k] + z_group[k+1])/2
        eps_ply_global = eps0 + z_mid * kappa
        eps_mat = global_to_material_strain(eps_ply_global, theta_list[k])
        sigma = Q @ eps_mat
        ply_stresses.append(sigma)
        
        s1, s2, t12 = sigma
        lam = tsai_wu_lambda(s1, s2, t12, Xt, Xc, Yt, Yc, S12)
        lambdas.append(lam)
        print(f"Ply {k+1:2d}  θ={theta_list[k]:>4}°   λ = {lam:8.3f}")
    
    lam_list_groups.append(np.array(lambdas))

all_theta = [theta for group in theta_groups for theta in group]

# Flatten theta and lambdas
all_theta = [theta for group in theta_groups for theta in group]
all_lambdas = np.concatenate(lam_list_groups)
fp = np.argmin(all_lambdas)

print("\nFIRST PLY FAILURE")
print("----------------")
print("Ply", fp+1, "at", all_theta[fp], "deg")
print("Load factor =", all_lambdas[fp])
print("Actual loads at failure:")

# Determine which NM group this ply came from
group_idx = 0
ply_count = 0
for idx, group in enumerate(theta_groups):
    if fp < ply_count + len(group):
        group_idx = idx
        break
    ply_count += len(group)

NM_fail = NM_groups[group_idx]
print("N =", all_lambdas[fp] * NM_fail[:3])
print("M =", all_lambdas[fp] * NM_fail[3:])


# WHAT THIS CODE DOES
# --------------------------------------------------------------------------
# Intake properties of composite as well as orientation
# Finds the basic Q stiffness matrix
# Solves for the ABD matrix for wach ply and adds the terms to create one ABD
# Inverts the ABD and multiplies it by the input loads for strain and bending
# Strain and bending combined for each ply to get stresses
# Stresses are converted to be in line with fiber direction
# The stresses are then placed through the Tsai-Wu failure criteria for FOS
#
# This process is completed for top/bottom (bending) favoring compression due 
# to worse properties for composites in compression
# 
# The process is also completed for the sides (shear) with no favoring
# occuring
#
# This assumes the top and bottom will be carrying all of the bending load
# and the sides will be carrying all of the shear
#
# Safety factors need to be applied further for any complexities past square