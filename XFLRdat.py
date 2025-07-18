import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- read the polar -------------------------------------------------
df = pd.read_csv(
    "Lark_8lb_VLMvisc.csv",
    comment="#",          # skip every line that starts with ‘#’
    skipinitialspace=True # drop spaces that follow each comma
)

# clean the headers
df.columns = df.columns.str.strip()   # remove stray spaces

# --- convert strings to numbers (just in case) ----------------------
df = df.apply(pd.to_numeric, errors="coerce")

# --- use the data ---------------------------------------------------
alpha = df["alpha"]
cl    = df["CL"]
cd    = df["CD"]

cd_of_cl = interp1d(df["CL"], df["CD"], kind='cubic')
alpha_of_cl = interp1d(df["CL"], df["alpha"], kind='cubic')

cl_query = 0.3775
cd_value = cd_of_cl(cl_query)
alpha_value = alpha_of_cl(cl_query)
print(f"Interpolated AOA and CD at CL={cl_query:.2f} are {alpha_value:.5f} and {cd_value:.5f}")
