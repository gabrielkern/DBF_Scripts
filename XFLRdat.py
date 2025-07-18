import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def xflr_dat(filename,name,value):
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

    cd_of_cl = interp1d(df["CL"], df["CD"], kind='cubic')
    alpha_of_cl = interp1d(df["CL"], df["alpha"], kind='cubic')
    cl_of_cd = interp1d(df["CD"], df["CL"], kind='cubic')
    alpha_of_cd = interp1d(df["CD"], df["alpha"], kind='cubic')
    cd_of_alpha = interp1d(df["alpha"], df["CD"], kind='cubic')
    cl_of_alpha = interp1d(df["alpha"], df["CL"], kind='cubic')

    if name == "CL" or "cl" or "Cl" or "cL":
        CD = cd_of_cl(value)
        alpha = alpha_of_cl(value)
        return CD,alpha
    if name == "CD" or "cd" or 'Cd' or "cD":
        CL = cl_of_cd(value)
        alpha = alpha_of_cd(value)
        return CL, alpha
    if name == "alpha" or "Alpha":
        CL = cl_of_alpha(value)
        CD = cd_of_alpha(value)
        return CL,CD


results = xflr_dat("Lark_8lb_VLMvisc.csv","Cl",0.3778)
print(results)