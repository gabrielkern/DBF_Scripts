import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass
from scipy.interpolate import interp1d, UnivariateSpline
import saved_airfoils


# Set parameters at the bottom of the file


@dataclass
class Airfoil:
    """Airfoil data structure."""
    name: str
    x_coordinates: List[float]
    y_coordinates: List[float]


@dataclass
class VLMResults:
    """Container for VLM analysis results"""
    alpha_degrees: List[float]
    CL: List[float]
    CDi: List[float]
    CM: List[float]
    L_over_D: List[float]
    alpha_L0: float  # Zero-lift angle of attack
    CL_alpha: float  # Lift curve slope (per radian)


@dataclass
class ConceptualWing:
    """All the wing parameters for conceptual analysis."""
    airfoil: Airfoil
    wing_area: float  # m²
    wing_span: float  # m
    taper_ratio: float
    aoa_min: Optional[int] = 0
    aoa_max: Optional[int] = 10
    aoa_diff: Optional[float] = 1.0
    airspeed: Optional[int] = 30  # m/s
    vlm_results: Optional[VLMResults] = None
    CD_parasitic: Optional[float] = None


@dataclass
class Panel:
    """VLM panel data structure"""
    corners: np.ndarray  # [4, 3] array of corner coordinates
    vortex_A: np.ndarray  # Bound vortex start point
    vortex_B: np.ndarray  # Bound vortex end point
    control_point: np.ndarray  # Control point for boundary condition
    normal_vector: np.ndarray  # Panel normal vector
    area: float
    local_chord: float
    y_position: float  # Spanwise position


def process_airfoil(airfoil: Airfoil) -> Tuple[float, Callable[[float], float], float]:
    """
    Process airfoil coordinate data to extract key geometric properties needed for wing analysis.
    This function takes raw airfoil coordinates and transforms them into three essential outputs:
    the total surface area (perimeter) of the airfoil for unit depth, a continuous function
    that can evaluate the camber line at any chordwise position, and the maximum thickness-to-chord
    ratio for parasitic drag calculations. The function automatically handles coordinate validation,
    ensures proper closure of the airfoil shape, adds additional points if the resolution is too
    low for accurate analysis, and separates the coordinates into upper and lower surfaces for
    proper camber line extraction and thickness measurement. This preprocessing step is crucial
    for VLM analysis as it provides the geometric foundation for generating wing panels with
    realistic airfoil shapes and the dimensional data needed for viscous drag estimation.
    """
    x_coords = np.array(airfoil.x_coordinates)
    y_coords = np.array(airfoil.y_coordinates)
    
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coordinates and y_coordinates must have the same length")
    
    if len(x_coords) < 4:
        raise ValueError("Airfoil must have at least 4 coordinate points")
    
    if not (np.isclose(x_coords[0], x_coords[-1], atol=1e-6) and 
            np.isclose(y_coords[0], y_coords[-1], atol=1e-6)):
        warnings.warn("Airfoil coordinates do not form a closed loop. Adding closure point.")
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
    
    # This can be added to increase the number of points/resolution of airfoil (not needed)
    '''
    if len(x_coords) < 100:
        x_coords, y_coords = _add_points(x_coords, y_coords, target_points=100)
    '''

    surface_area = _calculate_surface_area(x_coords, y_coords)
    upper_x, upper_y, lower_x, lower_y = _separate_surfaces(x_coords, y_coords)
    camber_function = _create_camber_function(upper_x, upper_y, lower_x, lower_y)
    thickness_to_chord_ratio = _calculate_thickness_ratio(upper_x, upper_y, lower_x, lower_y)
    
    return surface_area, camber_function, thickness_to_chord_ratio


def _calculate_thickness_ratio(upper_x: np.ndarray, upper_y: np.ndarray, 
                             lower_x: np.ndarray, lower_y: np.ndarray) -> float:
    """Calculate maximum thickness-to-chord ratio from airfoil coordinate data."""
    x_sample = np.linspace(0, 1, 100)
    
    upper_sort_idx = np.argsort(upper_x)
    lower_sort_idx = np.argsort(lower_x)
    
    upper_interp = interp1d(upper_x[upper_sort_idx], upper_y[upper_sort_idx], 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    lower_interp = interp1d(lower_x[lower_sort_idx], lower_y[lower_sort_idx], 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    
    thickness_distribution = upper_interp(x_sample) - lower_interp(x_sample)
    max_thickness = np.max(thickness_distribution)
    
    return max_thickness


def analyze_conceptual_wing(wing: ConceptualWing, N: int = 8, M: int = 4, rho: float = 1.225) -> ConceptualWing:
    """
    Perform complete conceptual wing analysis using Vortex Lattice Method to determine aerodynamic
    characteristics across a range of angles of attack, combined with empirical parasitic drag
    estimation to provide total aircraft performance predictions. This comprehensive function
    integrates airfoil processing, wing geometry generation, VLM panel creation, inviscid force
    calculations, and viscous drag estimation to produce engineering-level estimates suitable
    for conceptual aircraft design. The analysis assumes inviscid flow for lift and induced drag
    calculations using VLM theory, then adds viscous effects through empirical correlations for
    skin friction and form drag. The function processes the airfoil coordinates to extract camber
    and thickness characteristics, generates a trapezoidal wing planform with the specified geometry
    parameters, creates a structured panel mesh for VLM analysis, solves the aerodynamic influence
    coefficient system for each angle of attack, calculates lift, induced drag, and pitching moment
    coefficients, estimates parasitic drag using component buildup methodology, and combines all
    results to provide complete aerodynamic performance data for aircraft design optimization.
    """
    # Process airfoil to get geometry data needed for both VLM and parasitic drag analysis
    surface_area_normalized, camber_func, thickness_to_chord_ratio = process_airfoil(wing.airfoil)
    
    # Calculate mean chord from wing planform area and span
    mean_chord = wing.wing_area / wing.wing_span
    
    # Scale normalized airfoil perimeter to actual physical dimensions
    airfoil_surface_area_actual = surface_area_normalized * mean_chord
    
    print(f"Airfoil Analysis:")
    print(f"  Normalized perimeter: {surface_area_normalized:.4f}")
    print(f"  Mean chord: {mean_chord:.3f} m")
    print(f"  Actual perimeter: {airfoil_surface_area_actual:.4f} m")
    print(f"  Thickness-to-chord ratio: {thickness_to_chord_ratio:.3f}")
    
    # Set up angle of attack range for VLM analysis
    aoa_range = {
        "min": wing.aoa_min or 0,
        "max": wing.aoa_max or 10, 
        "diff": wing.aoa_diff or 1.0
    }
    
    # Perform VLM analysis for inviscid aerodynamic coefficients
    vlm_results = analyze_wing_vlm(
        camber_function=camber_func,
        wing_area=wing.wing_area,
        wing_span=wing.wing_span,
        taper_ratio=wing.taper_ratio,
        aoa_range=aoa_range,
        airspeed=wing.airspeed,
        N=N,
        M=M,
        rho=rho
    )
    
    # Calculate parasitic drag coefficient using empirical methods
    CD_parasitic = calculate_parasitic_drag(
        wing=wing,
        airfoil_surface_area=airfoil_surface_area_actual,
        mean_chord=mean_chord,
        thickness_to_chord_ratio=thickness_to_chord_ratio
    )
    
    # Store results in wing object for easy access and plotting
    wing.vlm_results = vlm_results
    wing.CD_parasitic = CD_parasitic
    
    print(f"\nComplete Aerodynamic Analysis Summary:")
    print(f"  Parasitic drag coefficient: {CD_parasitic:.5f}")
    print(f"  Induced drag range: {min(vlm_results.CDi):.5f} to {max(vlm_results.CDi):.5f}")
    print(f"  Total drag range: {min(vlm_results.CDi) + CD_parasitic:.5f} to {max(vlm_results.CDi) + CD_parasitic:.5f}")
    
    return wing


def analyze_wing_vlm(
    camber_function: Callable[[float], float],
    wing_area: float,
    wing_span: float,
    taper_ratio: float,
    aoa_range: dict,
    airspeed: float,
    N: int,
    M: int,
    rho: float = 1.225
) -> VLMResults:
    """
    Execute Vortex Lattice Method analysis on a trapezoidal wing configuration to determine
    aerodynamic coefficients as functions of angle of attack. This method discretizes the wing
    surface into quadrilateral panels, each containing a horseshoe vortex with bound and trailing
    segments. The analysis enforces the flow tangency boundary condition at control points by
    solving a linear system where the aerodynamic influence coefficients relate the strength
    of each vortex to the velocity it induces at every control point. Forces are calculated
    using the Kutta-Joukowski theorem, which relates circulation strength to the force per unit
    length on each bound vortex segment. The method provides excellent accuracy for attached flow
    conditions and captures important three-dimensional effects like induced drag and spanwise
    load distribution that are essential for aircraft performance estimation.
    """
    if wing_area <= 0 or wing_span <= 0:
        raise ValueError("Wing area and span must be positive")
    if not (0 < taper_ratio <= 1):
        raise ValueError("Taper ratio must be between 0 and 1")
    if N < 2 or M < 1:
        raise ValueError("Must have at least 2 spanwise and 1 chordwise panels")
    
    semi_span = wing_span / 2.0
    root_chord = 2 * wing_area / (wing_span * (1 + taper_ratio))
    tip_chord = taper_ratio * root_chord
    
    print(f"Wing Geometry:")
    print(f"  Root chord: {root_chord:.3f} m")
    print(f"  Tip chord: {tip_chord:.3f} m")
    print(f"  Semi-span: {semi_span:.3f} m")
    print(f"  Aspect ratio: {wing_span**2 / wing_area:.2f}")
    
    panels = _generate_wing_panels(
        camber_function, root_chord, tip_chord, semi_span, N, M
    )
    
    print(f"Generated {len(panels)} panels ({N} spanwise × {M} chordwise × 2 wings)")
    
    alpha_degrees = np.arange(
        aoa_range["min"], 
        aoa_range["max"] + aoa_range["diff"], 
        aoa_range["diff"]
    )
    
    CL_results = []
    CDi_results = []
    CM_results = []
    
    print(f"\nAnalyzing {len(alpha_degrees)} angle of attack points...")
    
    for alpha_deg in alpha_degrees:
        alpha_rad = np.deg2rad(alpha_deg)
        gamma_strengths = _solve_vlm_system(panels, airspeed, alpha_rad)
        CL, CDi, CM = _calculate_coefficients(
            panels, gamma_strengths, airspeed, alpha_rad, rho, wing_area, root_chord
        )
        CL_results.append(CL)
        CDi_results.append(CDi)
        CM_results.append(CM)
        print(f"  α = {alpha_deg:5.1f}°: CL = {CL:6.3f}, CDi = {CDi:7.4f}, CM = {CM:7.4f}")
    
    L_over_D = [cl/cdi if cdi > 1e-6 else 0 for cl, cdi in zip(CL_results, CDi_results)]
    alpha_L0, CL_alpha = _calculate_lift_curve_slope(alpha_degrees, CL_results)
    
    print(f"\nLift Curve Characteristics:")
    print(f"  Zero-lift AoA: {alpha_L0:.2f}°")
    print(f"  Lift curve slope: {CL_alpha:.3f} /radian ({np.rad2deg(CL_alpha):.3f} /degree)")
    
    return VLMResults(
        alpha_degrees=list(alpha_degrees),
        CL=CL_results,
        CDi=CDi_results,
        CM=CM_results,
        L_over_D=L_over_D,
        alpha_L0=alpha_L0,
        CL_alpha=CL_alpha
    )


def create_naca_4digit(naca_code: str, num_points: int = 100) -> Airfoil:
    """
    Generate NACA 4-digit airfoil coordinates using the standard analytical equations that define
    the thickness distribution and mean camber line. This function provides a convenient way to
    create well-known airfoil shapes for testing and validation purposes. The NACA 4-digit series
    uses a systematic naming convention where the first digit represents maximum camber as a percentage
    of chord, the second digit indicates the position of maximum camber in tenths of chord, and the
    last two digits specify maximum thickness as a percentage of chord. The algorithm computes both
    the symmetric thickness distribution and the cambered mean line, then combines them to create
    the final airfoil shape with proper coordinate ordering for downstream processing.
    """
    if len(naca_code) != 4 or not naca_code.isdigit():
        raise ValueError("NACA code must be 4 digits")
    
    m = int(naca_code[0]) / 100
    p = int(naca_code[1]) / 10
    t = int(naca_code[2:]) / 100
    
    beta = np.linspace(0, np.pi, num_points // 2)
    x = 0.5 * (1 - np.cos(beta))
    
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                  0.2843 * x**3 - 0.1036 * x**4)
    
    yc = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < p:
            yc[i] = m * xi / p**2 * (2 * p - xi)
        else:
            yc[i] = m * (1 - xi) / (1 - p)**2 * (1 + xi - 2 * p)
    
    x_coords = []
    y_coords = []
    
    for i in range(len(x) - 1, -1, -1):
        x_coords.append(x[i])
        y_coords.append(yc[i] + yt[i])
    
    for i in range(1, len(x)):
        x_coords.append(x[i])
        y_coords.append(yc[i] - yt[i])
    
    return Airfoil(
        name=f"NACA {naca_code}",
        x_coordinates=x_coords,
        y_coordinates=y_coords
    )


def estimate_skin_friction(Reynolds_number: float, Re_transition: float = 500000.0) -> float:
    """
    Calculate skin friction coefficient using the corrected Prandtl-Schlichting formula for 
    transitional flow. This formula accounts for the physical reality that boundary layers 
    start laminar at the leading edge and transition to turbulent flow downstream. The correction
    term subtracts the laminar contribution that would have occurred if the entire surface
    remained laminar, then adds the actual mixed laminar-turbulent behavior. This approach
    captures the fact that aircraft wings typically operate in the transitional Reynolds
    number regime where both laminar and turbulent boundary layer regions coexist, making
    it more accurate than pure turbulent flat plate theory for conceptual design applications.
    """
    if Reynolds_number <= 0:
        raise ValueError("Reynolds number must be positive")
    
    if Reynolds_number < Re_transition:
        CF_laminar = 1.328 / np.sqrt(Reynolds_number)
        return CF_laminar
    else:
        CF_turbulent = 0.455 / (np.log10(Reynolds_number) ** 2.58)
        laminar_correction = 1700.0 / Reynolds_number
        CF_corrected = CF_turbulent - laminar_correction
        return max(CF_corrected, 0.001)


def form_factor_estimation(thickness_to_chord_ratio: float) -> float:
    """
    Calculate wing form factor using Torenbeek's empirical correlation that accounts for
    the three-dimensional pressure distribution effects of real airfoil shapes compared
    to flat plates. The form factor represents how much additional drag is created by
    the airfoil's thickness and camber, which alter the pressure gradients and boundary
    layer development compared to a simple flat plate. The quadratic and quartic terms
    capture the nonlinear way that thick airfoils create increasingly adverse pressure
    gradients, while the linear term represents the basic effect of displacing the
    streamlines around the airfoil thickness. This correlation is widely used in
    conceptual design because it provides good accuracy with minimal geometric input.
    """
    if not (0 <= thickness_to_chord_ratio <= 1):
        raise ValueError("Thickness-to-chord ratio must be between 0 and 1")
    
    t_over_c = thickness_to_chord_ratio
    form_factor = 1.0 + 2.7 * t_over_c + 100.0 * (t_over_c ** 4)
    
    return form_factor


def calculate_parasitic_drag(wing: ConceptualWing, airfoil_surface_area: float, 
                           mean_chord: float, thickness_to_chord_ratio: float = 0.12) -> float:
    """
    Calculate parasitic drag coefficient for the wing using component buildup methodology
    that combines skin friction effects with form factor corrections. This function represents
    the complete viscous drag estimation process, starting with the fundamental physics of
    boundary layer friction and building up the geometric corrections needed for realistic
    three-dimensional wings. The calculation uses the airfoil perimeter to determine wetted
    area, applies Reynolds number-dependent skin friction correlations to capture the
    boundary layer behavior, includes form factor corrections for airfoil shape effects,
    and properly non-dimensionalizes using the same reference area as the VLM analysis
    to ensure the parasitic and induced drag coefficients can be directly summed for
    total drag estimation.
    """
    if wing.airspeed is None or wing.airspeed <= 0:
        raise ValueError("Wing airspeed must be specified and positive")
    
    air_density = 1.225  # kg/m³ at sea level
    air_viscosity = 1.81e-5  # kg/(m·s) at sea level, 15°C
    
    Reynolds_number = (air_density * wing.airspeed * mean_chord) / air_viscosity
    
    CF_skin = estimate_skin_friction(Reynolds_number)
    
    FF_form = form_factor_estimation(thickness_to_chord_ratio)
    
    wetted_area = airfoil_surface_area * wing.wing_span
    
    reference_area = wing.wing_area
    
    CD_parasitic = CF_skin * FF_form * (wetted_area / reference_area)
    
    print(f"Parasitic Drag Calculation:")
    print(f"  Reynolds number: {Reynolds_number:.2e}")
    print(f"  Skin friction coefficient: {CF_skin:.5f}")
    print(f"  Form factor: {FF_form:.3f}")
    print(f"  Wetted area: {wetted_area:.2f} m²")
    print(f"  Reference area: {reference_area:.2f} m²")
    print(f"  CD_parasitic: {CD_parasitic:.5f}")
    
    return CD_parasitic


def plot_wing_analysis(wing: ConceptualWing, save_plots: bool = False):
    """
    Create comprehensive visualization plots of complete wing analysis results including lift coefficient,
    total drag coefficient, induced and parasitic drag components versus angle of attack, and the total
    drag polar plot. These plots provide essential insights into both inviscid and viscous wing performance
    characteristics and are fundamental tools for aircraft design evaluation. The lift curve shows the linear
    relationship between angle of attack and lift generation from VLM analysis, while the drag plots reveal
    the critical trade-offs between lift-dependent induced drag and approximately constant parasitic drag.
    The total drag polar illustrates the complete aerodynamic efficiency across the flight envelope, showing
    how the wing's lift-to-drag ratio varies with operating conditions. The drag component breakdown helps
    designers understand whether performance is limited by induced drag (suggesting aspect ratio or wing
    loading optimization) or parasitic drag (suggesting surface quality, thickness, or Reynolds number effects).
    These comprehensive visualizations enable informed design decisions for aircraft configuration optimization.
    """
    if wing.vlm_results is None:
        raise ValueError("Wing must be analyzed before plotting. Call analyze_conceptual_wing() first.")
    
    if wing.CD_parasitic is None:
        raise ValueError("Wing parasitic drag must be calculated before plotting complete analysis.")
    
    results = wing.vlm_results
    
    # Calculate total drag coefficients for each angle of attack
    CD_total = [cdi + wing.CD_parasitic for cdi in results.CDi]
    CD_parasitic_array = [wing.CD_parasitic] * len(results.alpha_degrees)
    
    # Calculate total lift-to-drag ratios
    L_over_D_total = [cl/cd_tot if cd_tot > 1e-6 else 0 for cl, cd_tot in zip(results.CL, CD_total)]
    
    # Create comprehensive plot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    
    # Plot 1: Lift Coefficient vs Angle of Attack
    ax1.plot(results.alpha_degrees, results.CL, 'bo-', linewidth=2, markersize=4, label='CL')
    ax1.set_xlabel('Angle of Attack (degrees)')
    ax1.set_ylabel('CL')
    ax1.set_title('Lift Coefficient vs. Angle of Attack')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=results.alpha_L0, color='r', linestyle='--', alpha=0.5, 
                label=f'α_L0 = {results.alpha_L0:.1f}°')
    ax1.legend()
    
    # Plot 2: Drag Component Breakdown vs Angle of Attack
    ax2.plot(results.alpha_degrees, results.CDi, 'ro-', linewidth=2, markersize=4, label='CDi (Induced)')
    ax2.plot(results.alpha_degrees, CD_parasitic_array, 'g-', linewidth=2, label=f'CD0 (Parasitic) = {wing.CD_parasitic:.4f}')
    ax2.plot(results.alpha_degrees, CD_total, 'ko-', linewidth=2, markersize=4, label='CD (Total)')
    ax2.set_xlabel('Angle of Attack (degrees)')
    ax2.set_ylabel('CD')
    ax2.set_title('Drag Coefficient Components vs. Angle of Attack')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Moment Coefficient vs Angle of Attack
    ax3.plot(results.alpha_degrees, results.CM, 'go-', linewidth=2, markersize=4)
    ax3.set_xlabel('Angle of Attack (degrees)')
    ax3.set_ylabel('CM')
    ax3.set_title('Moment Coefficient vs. Angle of Attack')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Complete Drag Polar (CL vs Total CD)
    ax4.plot(CD_total, results.CL, 'mo-', linewidth=2, markersize=4, label='Total Drag Polar')
    ax4.plot(results.CDi, results.CL, 'r--', linewidth=1, alpha=0.7, label='Induced Drag Only')
    ax4.set_xlabel('CD')
    ax4.set_ylabel('CL')
    ax4.set_title('Complete Drag Polar (CL vs CD Total)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Find and annotate maximum total L/D point
    max_ld_total_idx = np.argmax(L_over_D_total)
    max_ld_total = L_over_D_total[max_ld_total_idx]
    if max_ld_total > 0:
        ax4.annotate(f'Max L/D = {max_ld_total:.1f}\nα = {results.alpha_degrees[max_ld_total_idx]:.1f}°\nCL = {results.CL[max_ld_total_idx]:.2f}',
                    xy=(CD_total[max_ld_total_idx], results.CL[max_ld_total_idx]),
                    xytext=(15, 15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.suptitle(f'Complete Wing Analysis: {wing.airfoil.name}\n'
                 f'AR = {wing.wing_span**2/wing.wing_area:.1f}, λ = {wing.taper_ratio:.2f}, '
                 f'V = {wing.airspeed} m/s', fontsize=14)
    plt.tight_layout()
    
    # Print performance summary
    print(f"\nPerformance Summary:")
    print(f"  Maximum total L/D: {max_ld_total:.1f} at α = {results.alpha_degrees[max_ld_total_idx]:.1f}°")
    print(f"  Parasitic drag: {wing.CD_parasitic:.4f} ({wing.CD_parasitic/CD_total[max_ld_total_idx]*100:.1f}% of total at max L/D)")
    print(f"  Induced drag at max L/D: {results.CDi[max_ld_total_idx]:.4f}")
    print(f"  Lift curve slope: {results.CL_alpha:.3f} /rad ({np.rad2deg(results.CL_alpha):.3f} /deg)")
    
    if save_plots:
        plt.savefig('complete_wing_analysis.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'complete_wing_analysis.png'")
    
    plt.show()


def _calculate_surface_area(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    distances = np.sqrt(dx**2 + dy**2)
    surface_area = np.sum(distances)
    return surface_area


def _add_points(x_coords: np.ndarray, y_coords: np.ndarray, target_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    le_idx = np.argmin(x_coords)
    
    upper_x_orig = x_coords[:le_idx + 1]
    upper_y_orig = y_coords[:le_idx + 1]
    
    lower_x_orig = x_coords[le_idx:]
    lower_y_orig = y_coords[le_idx:]
    
    upper_x_orig = upper_x_orig[::-1]
    upper_y_orig = upper_y_orig[::-1]
    
    if len(lower_x_orig) > 1:
        lower_x_orig = lower_x_orig[1:]
        lower_y_orig = lower_y_orig[1:]
    
    n_upper = target_points // 2
    n_lower = target_points - n_upper
    
    try:
        upper_sort_idx = np.argsort(upper_x_orig)
        lower_sort_idx = np.argsort(lower_x_orig)
        
        upper_x_sorted = upper_x_orig[upper_sort_idx]
        upper_y_sorted = upper_y_orig[upper_sort_idx]
        lower_x_sorted = lower_x_orig[lower_sort_idx]
        lower_y_sorted = lower_y_orig[lower_sort_idx]
        
        upper_spline = UnivariateSpline(upper_x_sorted, upper_y_sorted, s=0, k=3)
        lower_spline = UnivariateSpline(lower_x_sorted, lower_y_sorted, s=0, k=3)
        
        beta_upper = np.linspace(0, np.pi, n_upper)
        x_new_upper = 0.5 * (1 - np.cos(beta_upper))
        
        beta_lower = np.linspace(0, np.pi, n_lower)
        x_new_lower = 0.5 * (1 - np.cos(beta_lower))
        
        if not np.any(np.isclose(x_new_upper, 0, atol=1e-6)):
            x_new_upper[0] = 0.0
        if not np.any(np.isclose(x_new_lower, 0, atol=1e-6)):
            x_new_lower[0] = 0.0
        
        y_new_upper = upper_spline(x_new_upper)
        y_new_lower = lower_spline(x_new_lower)
        
        le_idx_upper = np.argmin(x_new_upper)
        le_idx_lower = np.argmin(x_new_lower)
        y_new_upper[le_idx_upper] = 0.0
        y_new_lower[le_idx_lower] = 0.0
        
        x_coords_new = []
        y_coords_new = []
        
        upper_reverse_idx = np.argsort(-x_new_upper)
        for idx in upper_reverse_idx:
            x_coords_new.append(x_new_upper[idx])
            y_coords_new.append(y_new_upper[idx])
        
        lower_forward_idx = np.argsort(x_new_lower)
        for i, idx in enumerate(lower_forward_idx):
            if i == 0:
                continue
            x_coords_new.append(x_new_lower[idx])
            y_coords_new.append(y_new_lower[idx])
        
        return np.array(x_coords_new), np.array(y_coords_new)
        
    except Exception as e:
        warnings.warn(f"Failed to create splines for point interpolation: {e}. Using linear interpolation instead.")
        
        distances = np.cumsum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
        distances = np.insert(distances, 0, 0)
        distances_norm = distances / distances[-1]
        t_new = np.linspace(0, 1, target_points)
        
        x_interp = interp1d(distances_norm, x_coords, kind='linear')
        y_interp = interp1d(distances_norm, y_coords, kind='linear')
        
        x_new = x_interp(t_new)
        y_new = y_interp(t_new)
        
        le_idx = np.argmin(x_new)
        x_new[le_idx] = 0.0
        y_new[le_idx] = 0.0
        
        return x_new, y_new


def _separate_surfaces(x_coords: np.ndarray, y_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    le_idx = np.argmin(x_coords)
    
    upper_x = x_coords[:le_idx + 1]
    upper_y = y_coords[:le_idx + 1]
    
    lower_x = x_coords[le_idx:]
    lower_y = y_coords[le_idx:]
    
    lower_x = lower_x[::-1]
    lower_y = lower_y[::-1]
    
    if len(lower_x) > 1:
        lower_x = lower_x[1:]
        lower_y = lower_y[1:]
    
    upper_x = upper_x[::-1]
    upper_y = upper_y[::-1]
    lower_x = lower_x[::-1]
    lower_y = lower_y[::-1]
    
    return upper_x, upper_y, lower_x, lower_y


def _create_camber_function(upper_x: np.ndarray, upper_y: np.ndarray, 
                          lower_x: np.ndarray, lower_y: np.ndarray) -> Callable[[float], float]:
    try:
        upper_sort_idx = np.argsort(upper_x)
        lower_sort_idx = np.argsort(lower_x)
        
        upper_interp = interp1d(upper_x[upper_sort_idx], upper_y[upper_sort_idx], 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        lower_interp = interp1d(lower_x[lower_sort_idx], lower_y[lower_sort_idx], 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        
    except ValueError as e:
        raise ValueError(f"Error creating interpolation functions: {e}")
    
    def camber_function(x_over_c: float) -> float:
        if not (0 <= x_over_c <= 1):
            warnings.warn(f"x/c = {x_over_c} is outside valid range [0, 1]")
        
        y_upper = upper_interp(x_over_c)
        y_lower = lower_interp(x_over_c)
        camber = (y_upper + y_lower) / 2.0
        
        return float(camber)
    
    return camber_function


def _generate_wing_panels(
    camber_function: Callable[[float], float],
    root_chord: float,
    tip_chord: float,
    semi_span: float,
    N: int,
    M: int
) -> List[Panel]:
    panels = []
    
    y_stations = np.linspace(0, semi_span, N + 1)
    x_stations = np.linspace(0, 1, M + 1)
    
    for wing_side in ['right', 'left']:
        y_multiplier = 1 if wing_side == 'right' else -1
        
        for i in range(N):
            y1 = y_multiplier * y_stations[i]
            y2 = y_multiplier * y_stations[i + 1]
            
            eta1 = abs(y1) / semi_span
            eta2 = abs(y2) / semi_span
            chord1 = root_chord * (1 - (1 - tip_chord/root_chord) * eta1)
            chord2 = root_chord * (1 - (1 - tip_chord/root_chord) * eta2)
            
            for j in range(M):
                x1_norm = x_stations[j]
                x2_norm = x_stations[j + 1]
                
                x1_1 = x1_norm * chord1
                z1_1 = camber_function(x1_norm) * chord1
                
                x1_2 = x1_norm * chord2
                z1_2 = camber_function(x1_norm) * chord2
                
                x2_2 = x2_norm * chord2
                z2_2 = camber_function(x2_norm) * chord2
                
                x2_1 = x2_norm * chord1
                z2_1 = camber_function(x2_norm) * chord1
                
                corners = np.array([
                    [x1_1, y1, z1_1],
                    [x1_2, y2, z1_2],
                    [x2_2, y2, z2_2],
                    [x2_1, y1, z2_1]
                ])
                
                vortex_A = corners[0] + 0.25 * (corners[3] - corners[0])
                vortex_B = corners[1] + 0.25 * (corners[2] - corners[1])
                
                cp_inner = corners[0] + 0.75 * (corners[3] - corners[0])
                cp_outer = corners[1] + 0.75 * (corners[2] - corners[1])
                control_point = (cp_inner + cp_outer) / 2.0
                
                diag1 = corners[2] - corners[0]
                diag2 = corners[3] - corners[1]
                normal = np.cross(diag1, diag2)
                normal_mag = np.linalg.norm(normal)
                if normal_mag > 1e-9:
                    normal = normal / normal_mag
                    if normal[2] < 0:
                        normal = -normal
                else:
                    normal = np.array([0, 0, 1])
                
                area = 0.5 * normal_mag
                local_chord = (chord1 + chord2) / 2.0
                y_position = (y1 + y2) / 2.0
                
                panel = Panel(
                    corners=corners,
                    vortex_A=vortex_A,
                    vortex_B=vortex_B,
                    control_point=control_point,
                    normal_vector=normal,
                    area=area,
                    local_chord=local_chord,
                    y_position=y_position
                )
                
                panels.append(panel)
    
    return panels


def _solve_vlm_system(panels: List[Panel], V_inf: float, alpha: float) -> np.ndarray:
    n_panels = len(panels)
    
    V_inf_vec = V_inf * np.array([np.cos(alpha), 0, np.sin(alpha)])
    V_inf_unit = V_inf_vec / np.linalg.norm(V_inf_vec)
    
    AIC = np.zeros((n_panels, n_panels))
    RHS = np.zeros(n_panels)
    
    for i in range(n_panels):
        RHS[i] = -np.dot(V_inf_vec, panels[i].normal_vector)
        
        for j in range(n_panels):
            velocity_induced = _calculate_horseshoe_velocity(
                panels[i].control_point,
                panels[j].vortex_A,
                panels[j].vortex_B,
                V_inf_unit,
                gamma=1.0
            )
            
            AIC[i, j] = np.dot(velocity_induced, panels[i].normal_vector)
    
    try:
        gamma_strengths = np.linalg.solve(AIC, RHS)
    except np.linalg.LinAlgError:
        print("Warning: Singular AIC matrix. Using least squares solution.")
        gamma_strengths = np.linalg.lstsq(AIC, RHS, rcond=None)[0]
    
    return gamma_strengths


def _calculate_horseshoe_velocity(
    eval_point: np.ndarray,
    vortex_A: np.ndarray,
    vortex_B: np.ndarray,
    V_inf_unit: np.ndarray,
    gamma: float = 1.0,
    far_factor: float = 1000.0
) -> np.ndarray:
    segment_length = np.linalg.norm(vortex_B - vortex_A)
    far_distance = far_factor * max(segment_length, 1.0)
    
    A_trail = vortex_A - V_inf_unit * far_distance
    B_trail = vortex_B + V_inf_unit * far_distance
    
    v_bound = _vortex_segment_velocity(eval_point, vortex_A, vortex_B, gamma)
    v_trail_A = _vortex_segment_velocity(eval_point, A_trail, vortex_A, gamma)
    v_trail_B = _vortex_segment_velocity(eval_point, vortex_B, B_trail, gamma)
    
    return v_bound + v_trail_A + v_trail_B


def _vortex_segment_velocity(
    eval_point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    gamma: float
) -> np.ndarray:
    r1 = eval_point - seg_start
    r2 = eval_point - seg_end
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    if r1_mag < 1e-9 or r2_mag < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    
    r1_cross_r2 = np.cross(r1, r2)
    cross_mag_sq = np.dot(r1_cross_r2, r1_cross_r2)
    
    if cross_mag_sq < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    
    r_seg = seg_end - seg_start
    
    factor = (gamma / (4 * np.pi)) * (
        np.dot(r_seg, r1) / r1_mag - np.dot(r_seg, r2) / r2_mag
    ) / cross_mag_sq
    
    return factor * r1_cross_r2


def _calculate_coefficients(
    panels: List[Panel],
    gamma_strengths: np.ndarray,
    V_inf: float,
    alpha: float,
    rho: float,
    S_ref: float,
    c_ref: float
) -> Tuple[float, float, float]:
    V_inf_vec = V_inf * np.array([np.cos(alpha), 0, np.sin(alpha)])
    V_inf_unit = V_inf_vec / np.linalg.norm(V_inf_vec)
    
    total_force = np.array([0.0, 0.0, 0.0])
    total_moment = np.array([0.0, 0.0, 0.0])
    
    moment_ref = np.array([c_ref * 0.25, 0.0, 0.0])
    
    for i, panel in enumerate(panels):
        gamma_i = gamma_strengths[i]
        
        bound_vortex = panel.vortex_B - panel.vortex_A
        bound_center = (panel.vortex_A + panel.vortex_B) / 2.0
        
        velocity_induced = np.array([0.0, 0.0, 0.0])
        for j, other_panel in enumerate(panels):
            if i != j:
                velocity_induced += _calculate_horseshoe_velocity(
                    bound_center,
                    other_panel.vortex_A,
                    other_panel.vortex_B,
                    V_inf_unit,
                    gamma=gamma_strengths[j]
                )
        
        total_velocity = V_inf_vec + velocity_induced
        force_segment = rho * gamma_i * np.cross(total_velocity, bound_vortex)
        total_force += force_segment
        
        r_arm = bound_center - moment_ref
        moment_segment = np.cross(r_arm, force_segment)
        total_moment += moment_segment
    
    L = total_force[2] * np.cos(alpha) - total_force[0] * np.sin(alpha)
    Di = total_force[0] * np.cos(alpha) + total_force[2] * np.sin(alpha)
    M = total_moment[1]
    
    q_inf = 0.5 * rho * V_inf**2
    
    CL = L / (q_inf * S_ref) if q_inf > 0 and S_ref > 0 else 0
    CDi = Di / (q_inf * S_ref) if q_inf > 0 and S_ref > 0 else 0
    CM = M / (q_inf * S_ref * c_ref) if q_inf > 0 and S_ref > 0 and c_ref > 0 else 0
    
    return CL, CDi, CM


def _calculate_lift_curve_slope(alpha_degrees: np.ndarray, CL_values: List[float]) -> Tuple[float, float]:
    alpha_rad = np.deg2rad(alpha_degrees)
    
    A = np.column_stack([alpha_rad, np.ones(len(alpha_rad))])
    coeffs, _, _, _ = np.linalg.lstsq(A, CL_values, rcond=None)
    
    CL_alpha = coeffs[0]
    CL_at_zero_alpha = coeffs[1]
    
    alpha_L0_rad = -CL_at_zero_alpha / CL_alpha if abs(CL_alpha) > 1e-6 else 0
    alpha_L0_deg = np.rad2deg(alpha_L0_rad)
    
    return alpha_L0_deg, CL_alpha


if __name__ == "__main__":
    # Create a NACA 2412 airfoil with function or import from saved_airfoils.py
    #naca2412 = create_naca_4digit("2412")
    naca2412 = saved_airfoils.naca2412

    # Set to true to get graphs
    plot_results = True
    parametric_study = True
    parameter = "taper_ratio"

    # Determine meshing for VLM analysis
    N = 8  # Number of spanwise stations
    M = 4  # Number of chordwise panels per spanwise station
    cruise_alpha = 3.0 # degrees

    # Define a conceptual wing with realistic parameters
    wing = ConceptualWing(
        airfoil=naca2412,
        wing_area=0.3716, # m² (4ft^2)
        wing_span=1.524, # m (5ft)
        taper_ratio=0.4,
        aoa_min=-2, # deg
        aoa_max=12, # deg
        aoa_diff=1.0, # deg
        airspeed=35.7632 # m/s (80 mph)
    )

    if plot_results and not parametric_study:
        
        # Perform complete aerodynamic analysis including both inviscid and viscous effects
        analyzed_wing = analyze_conceptual_wing(wing, N, M)
        
        # Create comprehensive performance plots
        if plot_results:
            plot_wing_analysis(analyzed_wing)
        
        # Extract and display key performance metrics
        results = analyzed_wing.vlm_results
        print(f"\nDetailed Performance Analysis:")
        print(f"  Aircraft Configuration:")
        print(f"    Airfoil: {analyzed_wing.airfoil.name}")
        print(f"    Wing area: {analyzed_wing.wing_area:.1f} m²")
        print(f"    Wing span: {analyzed_wing.wing_span:.1f} m") 
        print(f"    Aspect ratio: {analyzed_wing.wing_span**2/analyzed_wing.wing_area:.1f}")
        print(f"    Taper ratio: {analyzed_wing.taper_ratio:.2f}")
        print(f"    Mean chord: {analyzed_wing.wing_area/analyzed_wing.wing_span:.2f} m")
        print(f"  Aerodynamic Characteristics:")
        print(f"    Zero-lift AoA: {results.alpha_L0:.2f}°")
        print(f"    Lift curve slope: {results.CL_alpha:.3f} /rad ({np.rad2deg(results.CL_alpha):.3f} /deg)")
        print(f"    Parasitic drag coefficient: {analyzed_wing.CD_parasitic:.5f}")
        print(f"    Maximum total L/D: {max([cl/(cdi + analyzed_wing.CD_parasitic) for cl, cdi in zip(results.CL, results.CDi)]):.1f}")
        
        # Demonstrate how to access results for further analysis
        cruise_alpha_idx = np.argmin(np.abs(np.array(results.alpha_degrees) - cruise_alpha))
        cruise_CL = results.CL[cruise_alpha_idx]
        cruise_CDi = results.CDi[cruise_alpha_idx]
        cruise_CD_total = cruise_CDi + analyzed_wing.CD_parasitic
        
        print(f"  Example Cruise Condition (α = {cruise_alpha}°):")
        print(f"    CL = {cruise_CL:.3f}")
        print(f"    CDi = {cruise_CDi:.5f}")
        print(f"    CD_parasitic = {analyzed_wing.CD_parasitic:.5f}")
        print(f"    CD_total = {cruise_CD_total:.5f}")
        print(f"    L/D = {cruise_CL/cruise_CD_total:.1f}")
        print(f"    Parasitic drag fraction: {analyzed_wing.CD_parasitic/cruise_CD_total*100:.1f}%")

    elif parametric_study:
        # Conduct a parametric study varying aspect ratio and taper ratio
        
        if parameter == "taper_ratio":
            tr_min = 0
            tr_max = 1
            num_points = 11
            taper_ratios = np.linspace(tr_min,tr_max,num_points)
            wing.aoa_min = cruise_alpha
            wing.aoa_max = cruise_alpha
            wing.aoa_diff = 1.0

            for taper_ratio in taper_ratios:
                wing.taper_ratio = taper_ratio
                analyzed_wing = analyze_conceptual_wing(wing, N, M)
