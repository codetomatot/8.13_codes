import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

df = pd.read_csv('pe_data2.csv')
df.columns = df.columns.str.strip()

wl_col = df["Wavelength (nm) +- 2"]
v_col = df["V (V) +- .005"]
i_col = df["I (pA)"]
unc_col = df["uncertainty in I (.0001 default)"]
unc_col = pd.to_numeric(unc_col, errors='coerce').fillna(0.0001)

v_col = pd.to_numeric(v_col, errors='coerce')
i_col = pd.to_numeric(i_col, errors='coerce')

plt.figure(figsize=(9, 5))
for wl, group in df.groupby(wl_col):
    sorted_group = group.sort_values(by=df.columns[1])
    plt.plot(
        sorted_group[df.columns[1]],
        sorted_group[df.columns[2]],
        marker='o',
        linestyle='none', 
        label=f'lambda= {wl} nm'
    )

plt.xlabel('Voltage (V)')
plt.ylabel('Current (pA)')
plt.title('Photoelectric Effect: Experimental Data Points')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
# plt.show()


wavelengths = sorted(wl_col.unique())
tangent_cutoffs = {}
fig, axes = plt.subplots(len(wavelengths), 1, figsize=(8, 3 * len(wavelengths)))
for ax, wl in zip(axes, wavelengths):
    group = df[wl_col == wl].sort_values(by=df.columns[1])
    v = group[df.columns[1]].values
    i = group[df.columns[2]].values
    dI_dV = np.gradient(i, v)
    max_slope_idx = np.argmin(dI_dV) 
    start = max(0, max_slope_idx - 1)
    end = min(len(v), max_slope_idx + 0)
    res = linregress(v[start:end], i[start:end])
    slope, intercept = res.slope, res.intercept
    v_cutoff = -intercept / slope
    # print(f"v_cutoff esimtea: {v_cutoff}")
    tangent_cutoffs[wl] = v_cutoff
    v_line = np.linspace(v[start] - 0.5, v_cutoff + 0.3, 100)
    i_line = slope * v_line + intercept
    ax.plot(v, i, 'bo-', label='Data')
    ax.plot(v_line, i_line, 'r--', label=f'Tangent line (m = {slope:.3f})')
    ax.axvline(v_cutoff, color='g', linestyle=':', label=f'V_cutoff = {v_cutoff:.3f} V')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_title(f'Wavelength {wl} nm — Tangent Method')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (pA)')
    ax.set_ylim(-0.005, max(i) * 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
# plt.savefig('tangents.png')
plt.tight_layout()
plt.show()

voltages = [v0 for wl, v0 in tangent_cutoffs.items()] 
wavelengths = list(tangent_cutoffs)

plt.scatter(wavelengths, voltages)
sns.regplot(x=wavelengths, y=voltages)
slope, intercept, r_value, p_value, std_err = linregress(wavelengths, voltages)
print(f"std_err = {std_err}")
plt.errorbar(wavelengths, voltages, res.stderr, ls='none')
plt.show()
c = 3.00e8  
e = 1.602e-19
frequencies = c / (np.array(wavelengths) * 1e-9) 
slope, intercept, r_value, p_value, std_err = linregress(frequencies, voltages)
h_estimated = slope * e 
phi_eV = -intercept 
phi_joules = phi_eV * e
print(f"h estimate: {h_estimated} and work fucntion: {phi_eV}") 
def fowler_square_root_cutoff(v, i):
    sort_idx = np.argsort(v)
    v, i = v[sort_idx], i[sort_idx]
    valid_mask = i > 0
    v_valid, i_valid = v[valid_mask], i[valid_mask]
    
    sqrt_i = np.sqrt(i_valid)
    fit_points = max(3, len(sqrt_i) // 2)
    slope, intercept, _, _, _ = linregress(v_valid[:fit_points], sqrt_i[:fit_points])
    
    v_cutoff = -intercept / slope
    return v_cutoff

fowler_cutoffs = {}
for wl, group in df.groupby(wl_col):
    fowler_cutoffs[wl] = fowler_square_root_cutoff(group[df.columns[1]].values, group[df.columns[2]].values)