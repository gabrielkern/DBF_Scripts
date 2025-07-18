import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def xflr_interp(filename):
    # --- read the polar -------------------------------------------------
    df = pd.read_csv(
    filename,
    comment="#",          # skip every line that starts with ‘#’
    skipinitialspace=True # drop spaces that follow each comma
    )
    # clean the headers
    df.columns = df.columns.str.strip()   # remove stray spaces
    # --- convert strings to numbers (just in case) ----------------------
    df = df.apply(pd.to_numeric, errors="coerce")

    coeff_interp = {
    'cl_of_alpha': interp1d(df["alpha"], df["CL"], kind='cubic', fill_value='extrapolate'),
    'cd_of_alpha': interp1d(df["alpha"], df["CD"], kind='cubic', fill_value='extrapolate'),
    'cd_of_cl':    interp1d(df["CL"], df["CD"], kind='cubic', fill_value='extrapolate'),
    'alpha_of_cl': interp1d(df["CL"], df["alpha"], kind='cubic', fill_value='extrapolate')
}
    return  coeff_interp

def xflr_results(interpolators,name,value):
    name = name.lower()  # Normalize to lowercase for easier matching

    if name == "cl":
        CD = interpolators['cd_of_cl'](value)
        alpha = interpolators['alpha_of_cl'](value)
        return float(CD), float(alpha)

    elif name == "cd":
        # You need to define these interpolators first
        CL = interpolators['cl_of_cd'](value)
        alpha = interpolators['alpha_of_cd'](value)
        return float(CL), float(alpha)

    elif name == "alpha":
        CL = interpolators['cl_of_alpha'](value)
        CD = interpolators['cd_of_alpha'](value)

        return float(CL), float(CD)

    else:
        raise ValueError("name must be one of: 'CL', 'CD', 'alpha'")

interpolators = xflr_interp("Lark_8lb_VLMvisc.csv")
results = xflr_results(interpolators, "Cl", 0.3778)
print("Results from CL:", results)