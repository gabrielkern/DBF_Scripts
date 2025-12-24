# Tool to estimate the area density of carbon fiber based on volume fractions and densities of carbon and resin.

def calculate_area_density(V_f, rho_f, layer_count, layer_thickness, spec_gravity_resin):
    """
    Calculate the area density of a carbon fiber composite laminate.

    Parameters:
    V_f (float): Volume fraction of carbon fiber (between 0 and 1).
    rho_f (float): Area density of carbon fiber (in lb/in^3).
    rho_m (float): Volume density of resin matrix (in lb/in^3).
    layer_count (int): Number of layers in the laminate.

    Returns:
    float: Area density of the laminate (in lb/ft^2).
    """
    # Calculate the area density of all dry fabric layers
    thickness = layer_count * layer_thickness
    unit_weight_fabric = layer_count * rho_f

    # Calculate the density of resin (typically given in specific gravity)
    water_density = 0.036 # lb/in^3
    density_resin = spec_gravity_resin * water_density
    unit_weight_resin = density_resin * (1 - V_f) * thickness

    # Total the densities of both materials
    area_density_total = (unit_weight_fabric + unit_weight_resin)
    volume_density_total = area_density_total / thickness

    return area_density_total * 144, volume_density_total * 1728

if __name__ == "__main__":
    print(calculate_area_density(0.45, 0.000221836, 5, 0.006, 1.18))