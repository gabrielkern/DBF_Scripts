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
sweep = np.linspace(-0.5, 0.5, 50)

plt.figure(figsize=(12,8))

for name, base in params.items():
    pct_scores = []
    for scale in sweep:
        values = {
            "Passengers": pass_number_const,
            "Cargo": cargo_number_const,
            "Laps": num_laps_const,
            "Whrs": Whrs_const
        }
        values[name] = base * (1 + scale)
        score = mission_score(values["Passengers"], values["Cargo"], values["Laps"], values["Whrs"])
        pct_change = ((score - baseline_score) / abs(baseline_score)) * 100  # % change
        pct_scores.append(pct_change)

    plt.plot(sweep*100, pct_scores, label=name, linewidth=2, marker='o', markersize=4)

plt.xlabel('Parameter Variation (%)', fontsize=12)
plt.ylabel('Mission Score Change (%)', fontsize=12)
plt.suptitle('Mission 2 Score Sensitivity', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
