import numpy as np
from abdcal import QPlaneStress, abd, compliance_rotation
from deformation import load_applied, ply_strain, ply_stress
from failure import max_stress, tsai_hill, tsai_wu

"""
This is a general-puurpose script for analyzing the mechanical behavior of a composite laminate.

Note that all units should be in imperial (inches, pounds, psi).

We hate metric.
"""

# First, define the properties of the dry fabric and the epoxy
E_f = 33*1e6 # psi, tensile modulus of dry fibers
poisson_ratio_f = 0.3 # Poisson's ratio of dry fibers
V_f = 0.45 # fiber volume fraction
G_f = E_f / (2 * (1 + poisson_ratio_f)) # shear modulus of dry fibers
sigma_f = 700e3 # psi, tensile strength of dry fibers

E_e = 4.6*1e5 # psi, tensile modulus of epoxy
poisson_ratio_e = 0.35 # Poisson's ratio of epoxy
V_e = 1 - V_f # epoxy volume fraction
G_e = E_e / (2 * (1 + poisson_ratio_e)) # shear modulus of epoxy
sigma_e_tensile = 7.3e3 # psi, tensile strength of epoxy
sigma_e_compressive = 11.5e3 # psi, compressive strength of epoxy

# Define some other parameters
zeta_E = 2
zeta_G = 1
eta_E = ((E_f / E_e) - 1) / ((E_f / E_e) + zeta_E)
eta_G = ((G_f / G_e) - 1) / ((G_f / G_e) + zeta_G)

# Define strengths
sigma_longitudinal_tensile = sigma_f * V_f + sigma_e_tensile * V_e  # psi, tensile strength of composite
sigma_longitudinal_compressive = 0.5 * sigma_longitudinal_tensile  # psi, compressive strength of composite (conservative)
sigma_transverse_tensile = sigma_e_tensile  # psi, transverse tensile strength of composite (epoxy dominated)
sigma_transverse_compressive = sigma_e_compressive  # psi, transverse compressive strength of composite (epoxy dominated)
tau = 8000 # psi, shear strength of composite (approximation, highly conservative)

print(f"Initial material properties set:")
print(f"Fabric has tensile modulus {E_f/1e6} psi, Poisson's ratio {poisson_ratio_f}, and shear modulus {G_f/1e6} psi.")
print(f"Epoxy has tensile modulus {E_e/1e6} psi, Poisson's ratio {poisson_ratio_e}, and shear modulus {G_e/1e6} psi.")
print(f"Zeta of {zeta_E} is used for longitudinal modulus calculations, and zeta of {zeta_G} is used for shear modulus calculations.")
print(f"Eta for longitudinal modulus is {eta_E}, and eta for shear modulus is {eta_G}.")
print(f"Fiber volume fraction is {V_f*100}%.")

# Use the rule of mixtures to estimate composite properties
E1 = (E_f * V_f) + (E_e * (1 - V_f)) # psi, longitudinal modulus
E2 = (E_e) * (1 + (zeta_E * eta_E * V_f)) / (1 - (eta_E * V_f)) # psi, transverse modulus  
G12 = (G_e) * (1 + (zeta_G * eta_G * V_f)) / (1 - (eta_G * V_f)) # psi, shear modulus
NU12 = (poisson_ratio_f * V_f) + (poisson_ratio_e * (1 - V_f)) # Poisson's ratio

# Generate reduced stiffness matrix [Q]
Q = QPlaneStress(E1, E2, NU12, G12)

print("Calculated composite reduced stiffness matrix:")
print(Q)

# Define stacking sequence (angles in degrees, top to bottom)
angles = [0, 45, -45, 90, 90, -45, 45, 0]  # Symmetric quasi-isotropic

Q_all_layers = []
for Q_i in range(len(angles)):
    Q_all_layers.append(Q)

print(f"Stacking sequence of {np.shape(angles)} plies defined: {angles}. Rotation matrices have shape: {np.shape(Q_all_layers)}")

# Define ply thicknesses
thickness = [0.006] * 8  # 0.006 in per ply (fabric only)

print(f"Thickness matrix set with shape: {np.shape(thickness)}.")

# Calculate ABD matrix
ABD = abd(Q_all_layers, angles, thickness)
abd_inv = np.linalg.inv(ABD)
# Returns the ABD matrix (maps )

print(f"ABD matrix calculated with shape: {np.shape(ABD)}.")

# Apply loads (lbs for forces, lb-in for moments)
# NOTE: This method creates a flat plate - imagine a large panel where the x-axis runs down the "main" axis
# whether that be down the fuse or along the wing spanwise. Extrapolate other dims from there.
load = np.array([[50],    # Nx - running load x
                 [0],       # Ny - running load y
                 [0],       # Nxy - shear load
                 [0],       # Mx - bending moment x
                 [0],       # My - bending moment y
                 [0]])      # Mxy - twisting moment

print(f"Applied load vector set with shape: {np.shape(load)}.")

# Calculate midplane strains and curvatures
deformation = load_applied(abd_inv, load)
# Returns: [εx, εy, γxy, κx, κy, κxy]

print(f"Midplane strains (ε,γ) and curvatures (κ) calculated with shape: {np.shape(deformation)}.")

# Calculate ply stresses of the top and bottom ply
stresses = ply_stress(deformation, Q_all_layers, angles, thickness)

print(f"Ply stresses calculated with shape: {np.shape(stresses)}.")

# Check failure criteria
strength_values = {
    'sl_max': sigma_longitudinal_tensile, # Longitudinal tensile strength
    'sl_min': -sigma_longitudinal_compressive, # Longitudinal compressive strength
    'st_max': sigma_transverse_tensile, # Transverse tensile strength
    'st_min': -sigma_transverse_compressive, # Transverse compressive strength
    'tlt_max': tau # Shear strength
}

passed = max_stress(stresses, **strength_values)
passed_tw = tsai_wu(stresses, **strength_values)

print("Failure analysis results:")
print(f"The max stress criteria was {"FAILED" if not passed else "PASSED"} and the Tsai-Wu criteria was {"FAILED" if not passed_tw else "PASSED"}.")