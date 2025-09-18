import numpy as np
import matplotlib.pyplot as plt

# baseline constants
pass_number_const = 30
cargo_number_const = 10
num_laps_const = 5
Whrs_const = 110

def mission_score(pass_number, cargo_number, num_laps, Whrs):
    income = (pass_number * (6 + (2 * num_laps))) + (cargo_number * (10 + (8 * num_laps)))
    cost   = num_laps * Whrs/100 * (10 + (pass_number * 0.5)+(cargo_number * 2))
    return income-cost

# baseline score
baseline_score = mission_score(pass_number_const, cargo_number_const, num_laps_const, Whrs_const)

# parameters and labels
params = {
    "Passengers": pass_number_const,
    "Cargo": cargo_number_const,
    "Laps": num_laps_const,
    "Whrs": Whrs_const
}

# sweep from -50% to +50% of baseline
sweep = np.linspace(0.5, 1.5, 10)

plt.figure(figsize=(8,6))

for name, base in params.items():
    pct_scores = []
    for scale in sweep:
        values = {
            "Passengers": pass_number_const,
            "Cargo": cargo_number_const,
            "Laps": num_laps_const,
            "Whrs": Whrs_const
        }
        values[name] = base * scale
        score = mission_score(values["Passengers"], values["Cargo"], values["Laps"], values["Whrs"])
        pct_change = ((score - baseline_score) / abs(baseline_score)) * 100  # % change
        pct_scores.append(pct_change)
    
    plt.plot((sweep-1)*100, pct_scores, label=name)

plt.axhline(0, color='k', linestyle='--', alpha=0.6)
plt.axvline(0, color='k', linestyle='--', alpha=0.6)
plt.xlabel("Parameter Change (%)")
plt.ylabel("Mission Score Change (%)")
plt.title("DBF Mission Score Sensitivity (-50% to +50%)")
plt.legend()
plt.grid(True)
plt.show()
