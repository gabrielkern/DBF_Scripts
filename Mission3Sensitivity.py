import numpy as np
import matplotlib.pyplot as plt

def mission_score_equation(lap_number: int, banner_length: float, RAC: float):
    banner_length = int( np.floor(banner_length * 4) ) / 4
    return (banner_length * lap_number / RAC)

lap_number = 10
banner_length = 120.0
RAC = 1

params = {
    'lap_number': lap_number,
    'banner_length': banner_length,
    'RAC': RAC
}

base = mission_score_equation(**params)
sweep = np.linspace(-0.5,0.5,50)

plt.figure(figsize=(12,8))
plt.suptitle('Mission 3 Score Sensitivity', fontsize=16, fontweight='bold')

for items, values in params.items():
    prc_list = []
    for position in sweep:
        temp_params = params.copy()
        temp_params[items] = values + (values * position)
        abs_val = mission_score_equation(**temp_params)
        prc_list.append((abs_val-base)/base * 100)
    plt.plot(sweep*100, prc_list, label=items, linewidth=2, marker='o', markersize=4)

plt.xlabel('Parameter Variation (%)', fontsize=12)
plt.ylabel('Mission Score Change (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)

print(f"Base Mission Score: {base:.3f}")
print(f"Parameters: lap_number={lap_number}, banner_length={banner_length}, RAC={RAC}")

plt.tight_layout()
plt.show()