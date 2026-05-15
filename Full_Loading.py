import numpy as np
import matplotlib.pyplot as plt
Acc = 606 # number of points do not touch causes problems with moment

########################################################################

#--Plane Specs----------------------------------------------------------

WT = 10 # full weight

#Fuelselage
Wb = 0 # body weight lbs
nw = 15  # g load
qload = [.042] # pounds per inch
qplacement = [19.85,27.35] # start,end first is pucks second ducks
placedlocations = [4.06,7.07,10,18.14] # location of motor,esc,battery,puck
placedweights = [0.75,0.325,1.5,0.375] # weights corresponding to above
# x=0 at nose
cg = 7.5 # location of cg
w = 14.36 # quarter chord positions
L = 43.8 # total length

#Wing
Ww = 2 # wing weight in pounds
taper = np.array([1]) # taper ratio
Cr = 11/12 # root chord inches #############
a0 = 2*np.pi #section lift coefficient
b_input= 36  /12 # span inches
B_width = 3.25  /12# body width in inches
term = 100 # number of terms
alphain = 3 # angle of attack
alphanolift = -1
rho = 0.0023769  # slugs/ft³

#--Composites-----------------------------------------------------------

E1_in=20e6 #psi
E2_in=20e6
G12_in=0.6e6
v12=0.3
Xt=200e3 #psi 
Xc=150e3
Yt=200e3
Yc=150e3
S12_in=14e3
eps_max = 0.01    # 2% shear strain limit
t_ply=0.02             # thickness of one ply in inches
FOS_goal = 20               # GOAL
Extra_factor = 3       # corrected factor from tests

#Spar
height=1.0 #in
width=0.25
theta_spar = [
    [0],   # Group 1: flanges/top
    [45]    # Group 2: webs/sides
]

#Fuelselage
D = 3            * np.ones(Acc*2+1, dtype=int) #inches
size = "full"
theta_fuse = [
    [0,45]   # Group 1: Fuselage
]

hatch_sections = [      # [start, end] in inches
    [20, 30],
    [35, 40],
] 


######################################################################## 






########################################################################

# Functions
def analyze_composite_spar(
    # --- Material properties ---
    E1,                # psi, fiber-direction modulus
    E2,                # psi, transverse modulus
    G12,               # psi, shear modulus
    v12,               # major Poisson's ratio

    # --- Strength properties ---
    Xt,                # psi, fiber tensile strength
    Xc,                # psi, fiber compressive strength
    Yt,                # psi, transverse tensile strength
    Yc,                # psi, transverse compressive strength
    S12,               # psi, in-plane shear strength

    # --- Max strain criterion ---
    eps_max,           # allowable strain magnitude (applied to ε1 and ε2)

    # --- Applied loads ---
    Shear,             # lbs, transverse shear
    Moment,            # lb-in, bending moment

    # --- Layups (independent) ---
    flange_angles,     # list of ply angles for top/bottom flanges, e.g. [0, 0]
    web_angles,        # list of ply angles for left/right webs,    e.g. [45, -45]
    t_ply,             # inches, single ply thickness

    # --- Section geometry ---
    height,            # inch, spar height
    width,             # inch, spar flange width

    verbose=True
):
    """
    Analyze a composite box-beam spar under bending + shear.

    Flanges (top/bottom) carry all bending  → Nx only, built from flange_angles
    Webs   (left/right)  carry all shear    → Nxy only, built from web_angles

    Each region has its own independent ABD matrix and per-ply failure check.
    The overall safety factor is the minimum across both regions.

    Returns
    -------
    dict:
        'safety_factor'      : minimum SF across all plies in both regions
        'critical_ply'       : 1-based ply index within its region
        'critical_region'    : 'Flange' or 'Web'
        'critical_angle'     : ply angle at critical ply
        'critical_criterion' : 'Tsai-Wu' or 'Max Strain'
        'critical_component' : strain component if Max Strain governs
        'tsai_wu_sf'         : min Tsai-Wu SF across all plies
        'max_strain_sf'      : min Max Strain SF across all plies
        'flange_results'     : per-ply list for flange group
        'web_results'        : per-ply list for web group
    """

    # ── derived material constants ─────────────────────────────────────────
    v21   = v12 * E2 / E1
    delta = 1.0 - v12 * v21

    Q_mat = np.array([
        [E1/delta,      v12*E2/delta, 0.0],
        [v12*E2/delta,  E2/delta,     0.0],
        [0.0,           0.0,          G12 ]
    ])

    # ── section properties ─────────────────────────────────────────────────
    flange_t = len(flange_angles) * t_ply
    web_t    = len(web_angles)    * t_ply

    I = (1/12)*width*height**3 - (1/12)*(width - 2*web_t)*(height - 2*flange_t)**3

    Q_stat = (flange_t * width * (height/2 - flange_t/2)
              + 2 * web_t * (height/2 - flange_t) * (height/2 - flange_t) / 2)

    q        = Shear * Q_stat / I
    Nxy      = q * web_t                        # shear resultant in web
    sigma_bend = Moment * (height/2) / I
    Nx       = -sigma_bend * flange_t           # bending resultant (compression)

    # ── analyze one group ──────────────────────────────────────────────────
    def analyze_group(theta_list, NM_vec, region_name):
        n     = len(theta_list)
        thick = n * t_ply
        z     = [-thick/2 + k*t_ply for k in range(n + 1)]

        A_lam = np.zeros((3,3))
        B_lam = np.zeros((3,3))
        D_lam = np.zeros((3,3))

        for k, theta in enumerate(theta_list):
            Qbar   = _transform_Qbar(Q_mat, theta)
            dz     = z[k+1] - z[k]
            A_lam += Qbar * dz
            B_lam += 0.5   * Qbar * (z[k+1]**2 - z[k]**2)
            D_lam += (1/3) * Qbar * (z[k+1]**3 - z[k]**3)

        ABD         = np.block([[A_lam, B_lam], [B_lam, D_lam]])
        strain_curv = np.linalg.inv(ABD) @ NM_vec
        eps0        = strain_curv[:3]
        kappa       = strain_curv[3:]

        plies = []
        for k, theta in enumerate(theta_list):
            z_mid     = (z[k] + z[k+1]) / 2
            eps_glob  = eps0 + z_mid * kappa
            eps_mat   = _global_to_material_strain(eps_glob, theta)
            sigma_mat = Q_mat @ eps_mat

            s1, s2, t12         = sigma_mat
            eps1, eps2, gamma12 = eps_mat

            tw_sf            = _tsai_wu_sf(s1, s2, t12, Xt, Xc, Yt, Yc, S12)
            ms_sf, ms_comp   = _max_strain_sf1(eps1, eps2, eps_max)

            governing_sf     = min(tw_sf, ms_sf)
            governing_crit   = "Tsai-Wu" if tw_sf <= ms_sf else "Max Strain"

            plies.append({
                "region"         : region_name,
                "ply"            : k + 1,
                "angle_deg"      : theta,
                "eps_material"   : eps_mat,
                "sigma_material" : sigma_mat,
                "tsai_wu_sf"     : tw_sf,
                "max_strain_sf"  : ms_sf,
                "max_strain_comp": ms_comp,
                "sf"             : governing_sf,
                "criterion"      : governing_crit,
            })

        return plies

    # ── run both regions independently ─────────────────────────────────────
    # Flanges: axial bending load only  [Nx, Ny, Nxy, Mx, My, Mxy]
    flange_NM = np.array([Nx,  0, 0,   0, 0, 0])
    # Webs:    shear load only
    web_NM    = np.array([0,   0, Nxy, 0, 0, 0])

    flange_results = analyze_group(flange_angles, flange_NM, "Flange")
    web_results    = analyze_group(web_angles,    web_NM,    "Web")

    all_plies    = flange_results + web_results
    critical     = min(all_plies, key=lambda p: p["sf"])
    overall_sf   = critical["sf"]
    tw_sf_global = min(p["tsai_wu_sf"]    for p in all_plies)
    ms_sf_global = min(p["max_strain_sf"] for p in all_plies)

    # ── print summary ──────────────────────────────────────────────────────
    return {
        "safety_factor"      : overall_sf,
        "critical_ply"       : critical["ply"],
        "critical_region"    : critical["region"],
        "critical_angle"     : critical["angle_deg"],
        "critical_criterion" : critical["criterion"],
        "critical_component" : critical.get("max_strain_comp"),
        "tsai_wu_sf"         : tw_sf_global,
        "max_strain_sf"      : ms_sf_global,
        "flange_results"     : flange_results,
        "web_results"        : web_results,
    }


# ── helpers ────────────────────────────────────────────────────────────────────

def _transform_Qbar(Q, theta_deg):
    th = np.radians(theta_deg)
    m, n = np.cos(th), np.sin(th)
    Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]
    return np.array([
        [Q11*m**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*n**4,
         (Q11+Q22-4*Q66)*m**2*n**2 + Q12*(m**4+n**4),
         (Q11-Q12-2*Q66)*m**3*n + (Q12-Q22+2*Q66)*m*n**3],
        [(Q11+Q22-4*Q66)*m**2*n**2 + Q12*(m**4+n**4),
         Q11*n**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*m**4,
         (Q11-Q12-2*Q66)*m*n**3 + (Q12-Q22+2*Q66)*m**3*n],
        [(Q11-Q12-2*Q66)*m**3*n + (Q12-Q22+2*Q66)*m*n**3,
         (Q11-Q12-2*Q66)*m*n**3 + (Q12-Q22+2*Q66)*m**3*n,
         (Q11+Q22-2*Q12-2*Q66)*m**2*n**2 + Q66*(m**4+n**4)]
    ])


def _global_to_material_strain(eps_global, theta_deg):
    th = np.radians(theta_deg)
    m, n = np.cos(th), np.sin(th)
    T = np.array([
        [ m**2,   n**2,   m*n],
        [ n**2,   m**2,  -m*n],
        [-2*m*n,  2*m*n,  m**2 - n**2]
    ])
    return T @ eps_global


def _tsai_wu_sf(s1, s2, t12, Xt, Xc, Yt, Yc, S):
    F1  = 1/Xt - 1/Xc
    F11 = 1/(Xt*Xc)
    F2  = 1/Yt - 1/Yc
    F22 = 1/(Yt*Yc)
    F66 = 1/S**2
    F12 = -1/(2*np.sqrt(Xt*Xc*Yt*Yc))

    A = F1*s1 + F2*s2
    B = F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2

    roots = np.roots([B, A, -1])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    return float(np.min(roots)) if len(roots) > 0 else np.inf


def _max_strain_sf1(eps1, eps2, eps_max):
    """Check ε1 and ε2 against eps_max. Returns (SF, component_name)."""
    checks = [
        (abs(eps1), "ε1 (fiber)"),
        (abs(eps2), "ε2 (transverse)"),
    ]
    best_sf, best_comp = np.inf, "None"
    for actual, name in checks:
        if actual > 1e-14:
            sf = eps_max / actual
            if sf < best_sf:
                best_sf, best_comp = sf, name
    return best_sf, best_comp
def analyze_composite_tube(
    # --- Material properties ---
    E1,                # psi, fiber-direction modulus
    E2,                # psi, transverse modulus
    G12,               # psi, shear modulus
    v12,               # major Poisson's ratio

    # --- Strength properties ---
    Xt,                # psi, fiber tensile strength
    Xc,                # psi, fiber compressive strength
    Yt,                # psi, transverse tensile strength
    Yc,                # psi, transverse compressive strength
    S12,               # psi, in-plane shear strength

    # --- Max strain criterion ---
    eps_max,           # allowable strain magnitude (applied to ε1 and ε2)

    # --- Applied loads ---
    Shear,             # lbs, transverse shear
    Moment,            # lb-in, bending moment

    # --- Layup ---
    ply_angles,        # list of ply angles in degrees, e.g. [0, 45, 0]
    t_ply,             # inches, single ply thickness

    # --- Geometry ---
    D,                 # inches, outer diameter
    half_section=0,    # 0 = full tube, 1 = half tube (hatch cutout)

    verbose=True
):
    """
    Analyze a composite circular tube (fuselage) under bending + shear.

    Parameters
    ----------
    half_section : int
        0 → full circular tube
        1 → half tube (open section, e.g. hatch cutout on bottom half)

    Returns
    -------
    dict:
        'safety_factor'      : minimum SF across all plies
        'critical_ply'       : 1-based ply index
        'critical_angle'     : ply angle at critical ply
        'critical_criterion' : 'Tsai-Wu' or 'Max Strain'
        'tsai_wu_sf'         : Tsai-Wu SF (min across plies)
        'max_strain_sf'      : Max Strain SF (min across plies)
        'per_ply'            : list of per-ply result dicts
        'NM_at_failure'      : {'N': ..., 'M': ...} scaled to failure load
    """

    if half_section not in (0, 1):
        raise ValueError("half_section must be 0 (full) or 1 (half).")

    # ── derived material constants ─────────────────────────────────────────
    v21   = v12 * E2 / E1
    delta = 1.0 - v12 * v21

    Q_mat = np.array([
        [E1/delta,      v12*E2/delta, 0.0],
        [v12*E2/delta,  E2/delta,     0.0],
        [0.0,           0.0,          G12 ]
    ])

    # ── geometry & section properties ─────────────────────────────────────
    n_plies = len(ply_angles)
    thick   = n_plies * t_ply
    R       = D / 2
    r       = R - thick

    if half_section == 1:
        A_area = np.pi / 2 * (R**2 - r**2)
        ybar   = (4 / (3 * np.pi)) * (R**3 - r**3) / (R**2 - r**2)
        I_base = np.pi / 64 * (D**4 - (D - thick*2)**4)
        Inertia = I_base - A_area * ybar**2
        Q_area  = (R**3 - r**3) / 3
    else:  # full tube
        Inertia = np.pi / 64 * (D**4 - (D - thick*2)**4)
        Q_area  = 2 / 3 * (R**3 - r**3)

    # Stress resultants
    q        = Shear * Q_area / Inertia
    Nxy      = q                                    # shear flow
    sigma_m  = Moment * (D / 2) / 2 / Inertia
    Nx       = -sigma_m * thick                     # bending (compression)

    NM_vec = np.array([Nx, 0, Nxy, 0, 0, 0])

    # ── CLT: build ABD ─────────────────────────────────────────────────────
    z = [-thick/2 + k * t_ply for k in range(n_plies + 1)]

    A_lam = np.zeros((3, 3))
    B_lam = np.zeros((3, 3))
    D_lam = np.zeros((3, 3))

    for k, theta in enumerate(ply_angles):
        Qbar  = _transform_Qbar(Q_mat, theta)
        dz    = z[k+1] - z[k]
        A_lam += Qbar * dz
        B_lam += 0.5   * Qbar * (z[k+1]**2 - z[k]**2)
        D_lam += (1/3) * Qbar * (z[k+1]**3 - z[k]**3)

    ABD           = np.block([[A_lam, B_lam], [B_lam, D_lam]])
    strain_curv   = np.linalg.inv(ABD) @ NM_vec
    eps0          = strain_curv[:3]
    kappa         = strain_curv[3:]

    # ── per-ply analysis ───────────────────────────────────────────────────
    per_ply = []

    for k, theta in enumerate(ply_angles):
        z_mid      = (z[k] + z[k+1]) / 2
        eps_glob   = eps0 + z_mid * kappa
        eps_mat    = _global_to_material_strain(eps_glob, theta)
        sigma_mat  = Q_mat @ eps_mat

        s1, s2, t12        = sigma_mat
        eps1, eps2, gamma12 = eps_mat

        # Tsai-Wu SF
        tw_sf = _tsai_wu_sf(s1, s2, t12, Xt, Xc, Yt, Yc, S12)

        # Max Strain SF (ε1 and ε2 checked against eps_max)
        ms_sf, ms_comp = _max_strain_sf(eps1, eps2, eps_mat, eps_max) #########################################

        governing_sf   = min(tw_sf, ms_sf)
        governing_crit = "Tsai-Wu" if tw_sf <= ms_sf else "Max Strain"

        per_ply.append({
            "ply"            : k + 1,
            "angle_deg"      : theta,
            "eps_material"   : eps_mat,
            "sigma_material" : sigma_mat,
            "tsai_wu_sf"     : tw_sf,
            "max_strain_sf"  : ms_sf,
            "max_strain_comp": ms_comp,
            "sf"             : governing_sf,
            "criterion"      : governing_crit,
        })

    # ── first-ply failure ──────────────────────────────────────────────────
    critical     = min(per_ply, key=lambda p: p["sf"])
    overall_sf   = critical["sf"]
    tw_sf_global = min(p["tsai_wu_sf"]   for p in per_ply)
    ms_sf_global = min(p["max_strain_sf"] for p in per_ply)

    NM_fail = {
        "N": overall_sf * NM_vec[:3],
        "M": overall_sf * NM_vec[3:],
    }

    # -- results ---------------------------------------------------------------

    return {
        "safety_factor"      : overall_sf,
        "critical_ply"       : critical["ply"],
        "critical_angle"     : critical["angle_deg"],
        "critical_criterion" : critical["criterion"],
        "critical_component" : critical.get("max_strain_comp"),
        "tsai_wu_sf"         : tw_sf_global,
        "max_strain_sf"      : ms_sf_global,
        "per_ply"            : per_ply,
        "NM_at_failure"      : NM_fail,
    }


# -- helpers ────────────────────────────────────────────────────────────────────

def _transform_Qbar(Q, theta_deg):
    th = np.radians(theta_deg)
    m, n = np.cos(th), np.sin(th)
    Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]
    return np.array([
        [Q11*m**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*n**4,
         (Q11+Q22-4*Q66)*m**2*n**2 + Q12*(m**4+n**4),
         (Q11-Q12-2*Q66)*m**3*n + (Q12-Q22+2*Q66)*m*n**3],
        [(Q11+Q22-4*Q66)*m**2*n**2 + Q12*(m**4+n**4),
         Q11*n**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*m**4,
         (Q11-Q12-2*Q66)*m*n**3 + (Q12-Q22+2*Q66)*m**3*n],
        [(Q11-Q12-2*Q66)*m**3*n + (Q12-Q22+2*Q66)*m*n**3,
         (Q11-Q12-2*Q66)*m*n**3 + (Q12-Q22+2*Q66)*m**3*n,
         (Q11+Q22-2*Q12-2*Q66)*m**2*n**2 + Q66*(m**4+n**4)]
    ])


def _global_to_material_strain(eps_global, theta_deg):
    th = np.radians(theta_deg)
    m, n = np.cos(th), np.sin(th)
    T = np.array([
        [ m**2,   n**2,   m*n],
        [ n**2,   m**2,  -m*n],
        [-2*m*n,  2*m*n,  m**2 - n**2]
    ])
    return T @ eps_global


def _tsai_wu_sf(s1, s2, t12, Xt, Xc, Yt, Yc, S):
    F1  = 1/Xt - 1/Xc
    F11 = 1/(Xt*Xc)
    F2  = 1/Yt - 1/Yc
    F22 = 1/(Yt*Yc)
    F66 = 1/S**2
    F12 = -1/(2*np.sqrt(Xt*Xc*Yt*Yc))

    A = F1*s1 + F2*s2
    B = F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2

    roots = np.roots([B, A, -1])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    return float(np.min(roots)) if len(roots) > 0 else np.inf


def _max_strain_sf(eps1, eps2, eps_mat, eps_max):
    """
    Check ε1 and ε2 against eps_max.
    Returns (SF, component_name).
    """
    checks = [
        (abs(eps1), "ε1 (fiber)"),
        (abs(eps2), "ε2 (transverse)"),
    ]
    best_sf   = np.inf
    best_comp = "None"
    for actual, name in checks:
        if actual > 1e-14:
            sf = eps_max / actual
            if sf < best_sf:
                best_sf   = sf
                best_comp = name
    return best_sf, best_comp


########################################################################

# Fuelselage

FOS = FOS_goal * Extra_factor

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

print("Max Fuse Shear:")
print(np.max(abs(shear)))

print("Max Fuse Bending:")
print(np.max(abs(moment_in))/12)

x = np.linspace(0, L, Acc)
hatch = np.zeros(Acc, dtype=int)
for start, end in hatch_sections:
        hatch[(x >= start) & (x <= end)] = 1

fuselage_designs = []

for i in range(len(shear)):
        angles = list(theta_fuse[0])
        for _ in range(50):
                results = analyze_composite_tube(
                E1_in, E2_in, G12_in, v12, Xt, Xc, Yt, Yc, S12_in,
                eps_max, shear[i], moment_in[i],
                angles, t_ply, D[i],
                half_section=hatch[i],
        )

                if results['safety_factor'] >= FOS:
                    break
                
        worst = min(results['per_ply'], key=lambda p: p['sf'])
        angles.append(worst['angle_deg'])

        fuselage_designs.append({
            'angles'   : angles,
            'sf'       : results['safety_factor'],
            'n_plies'  : len(angles),
            'hatch'    : hatch[i],
        })

n_zero = [sum(1 for a in d['angles'] if round(a) == 0)  for d in fuselage_designs]
n_45   = [sum(1 for a in d['angles'] if round(a) == 45) for d in fuselage_designs]      
fuse_sf    = [d['sf']  / Extra_factor  for d in fuselage_designs]

# plotting
fig, axs = plt.subplots(4,2, figsize=(15,6))
axs[0,0].plot(x_vals/12.0, shear, label='Shear Force', color='royalblue')
axs[0,0].plot(x_vals/12.0, moment_in/12.0, label='Moment', color='hotpink')
axs[0,0].axhline(0, color='k', linestyle='--', linewidth=1)
#axs[0].set_title("Fuselage Shear / Moment Diagram")
axs[0,0].set_xlabel("Fuselage Length (ft)")
axs[0,0].set_ylabel("Shear(lb),Moment (lb/ft)")
axs[0,0].grid(True)
axs[0,0].legend()
axs[2,0].plot(x_vals/12,n_zero, label='0° plies')
axs[2,0].plot(x_vals/12,n_45,   label='45° plies')
axs[2,0].legend(loc='best')
axs[2,0].grid(True)
axs[2,0].set_xlabel("Distance from the Nose (ft)")
axs[2,0].set_ylabel("Number of Plies")
axs[2,0].legend(loc='best')
axs[2,0].grid(True)
axs[2,0].set_ylim(0, 5)
axs[2,1].plot(x_vals/12,fuse_sf, label = 'Minimum FOS')
axs[2,1].set_ylabel("FOS")
axs[2,1].legend(loc='best')
axs[2,1].grid(True)
axs[2,1].set_ylim(0, 300)
axs[2,1].set_xlabel("Distance from the Nose (ft)")


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
        #print(U_needed/1.467) #prints in mph

        # solve for shear from lift
        LiftForce = np.zeros_like(C_y)
        wing = 0
        wingshear =[]
        for index in range(len(C_y)-1):
            LiftForce[index] = 1/2 * rho * U_needed**2 * (C_y[index] * Cl_local_data[index]+C_y[index+1] * Cl_local_data[index+1])/2 * abs(full_y[index]-full_y[index+1])
        #print(sum(LiftForce))
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

        
        print("Max Wing Shear:")
        print(np.max(wingshear))

        print("Max Wing Bending:")
        print(np.max(wingmoment))
 
        spar_designs = []
        for i in range(len(full_y)):
            flange = list(theta_spar[0])
            web = list(theta_spar[1])

            for _ in range(50):
                results = analyze_composite_spar(
                E1_in, E2_in, G12_in, v12, Xt, Xc, Yt, Yc, S12_in,
                eps_max, wingshear[i], wingmoment[i],
                flange, web, t_ply, height, width,
                 )

                if results['safety_factor'] >= FOS:
                 break

                worst_flange = min(results['flange_results'], key=lambda p: p['sf'])
                worst_web    = min(results['web_results'],    key=lambda p: p['sf'])

                if worst_flange['sf'] <= worst_web['sf']:
                 flange.append(worst_flange['angle_deg'])
                else:
                 web.append(worst_web['angle_deg'])

            spar_designs.append({
            'flange_angles' : flange,
            'web_angles'    : web,
            'safety_factor' : results['safety_factor'],
            'n_flange_plies': len(flange),
            'n_web_plies'   : len(web),
    })


        #--plotting-----------------------------------------------------
        n_flange = [d['n_flange_plies'] for d in spar_designs]
        n_web    = [d['n_web_plies']    for d in spar_designs]
        spar_sf = [d['safety_factor'] / Extra_factor for d in spar_designs]

        axs[0,1].plot(full_y, wingshear, label=f"$\\ Shear \\ $")
        axs[0,1].plot(full_y, wingmoment, label=f"$\\ Moment \\ $")
        axs[0,1].legend(loc='best')
        axs[0,1].grid(True)
        axs[0,1].set_xlabel("Distance from the Center (ft)")
        axs[0,1].set_ylabel("Shear(lb),Moment (lb/ft)")
        axs[0,1].set_xlim(right=1.5)
        axs[0,1].set_xlim(left=-1.5)
        axs[1,0].plot(full_y, LiftForce, label=f"$\\ Lift \\ lambda = {current_taper_ratio:.1f}$")
        axs[1,0].legend(loc='best')
        axs[1,0].grid(True)
        axs[1,0].set_xlabel("Distance from the Center (ft)")
        axs[1,0].set_ylabel("Lift in Pounds")
        axs[1,0].set_ylim(bottom=0)

        axs[3,0].plot(full_y,n_web, label = f"$\\ Web Plies \\ lambda = {current_taper_ratio:.1f}$")
        axs[3,0].plot(full_y,n_flange, label = f"$\\ Cap Plies \\ lambda = {current_taper_ratio:.1f}$")
        axs[3,0].set_xlabel("Distance from the Center (ft)")
        axs[3,0].set_ylabel("Number of Plies")
        axs[3,0].legend(loc='best')
        axs[3,0].grid(True)
        axs[3,0].set_ylim(0, 5)
        axs[3,1].plot(full_y,spar_sf, label = f"$\\ Minnimum FOS \\ lambda = {current_taper_ratio:.1f}$")
        axs[3,1].set_ylabel("FOS")
        axs[3,1].legend(loc='best')
        axs[3,1].grid(True)
        axs[3,1].set_ylim(0, 300)
        axs[3,1].set_xlabel("Distance from the Center (ft)")


plt.tight_layout()
plt.show()