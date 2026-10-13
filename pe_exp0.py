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

v_stat_err, v_sys_err = [], []
delta_V = 0.005      # Voltage resolution (V)
delta_I = 0.0001     # Current resolution (pA)
delta_lambda = 2.0

for ax, wl in zip(axes, wavelengths):
    group = df[wl_col == wl].sort_values(by=df.columns[1])
    v = group[df.columns[1]].values
    i = group[df.columns[2]].values
    dI_dV = np.gradient(i, v)
    max_slope_idx = np.argmin(dI_dV) 
    start = max(0, max_slope_idx - 1)
    end = min(len(v), max_slope_idx + 2)
    res = linregress(v[start:end], i[start:end])

    slope, intercept = res.slope, res.intercept
    v_cutoff = -intercept / slope

    sigma_m, sigma_b = res.stderr, res.intercept_stderr

    stat_err = np.sqrt((sigma_b / slope)**2 + (intercept * sigma_m / (slope**2))**2)
    sys_err = np.sqrt(delta_V**2 + (delta_I / abs(slope))**2)
    v_stat_err.append(stat_err)
    v_sys_err.append(sys_err)
    # print(f"v_cutoff esimtea: {v_cutoff}")
    tangent_cutoffs[wl] = v_cutoff
    v_line = np.linspace(v[start] - 0.5, v_cutoff + 0.3, 100)
    i_line = slope * v_line + intercept
    ax.plot(v, i, 'bo-', label='Data')
    ax.plot(v_line, i_line, 'r--', label=f'Tangent line (m = {slope:.3f})')
    ax.axvline(v_cutoff, color='g', linestyle=':', label=f'V_cutoff = {v_cutoff:.3f} V')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_title(f'Wavelength {wl} nm')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (pA)')
    ax.set_ylim(-0.005, max(i) * 1.05)
    # ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
# plt.savefig('tangents.png')
plt.tight_layout()
plt.show()

voltages = [v0 for wl, v0 in tangent_cutoffs.items()] 
wavelengths = list(tangent_cutoffs)

plt.figure()
plt.scatter(wavelengths, voltages)
sns.regplot(x=wavelengths, y=voltages)
plt.xlabel("Wavelengths (nm)")
plt.ylabel("Voltages (V)")
slope, intercept, r_value, p_value, std_err = linregress(wavelengths, voltages)

print(f"voltages: {voltages}")
print(f"v sys uncertainty: {v_sys_err}")
print(f"v stat uncertainty: {v_stat_err}")

plt.errorbar(wavelengths, voltages, yerr=v_sys_err, fmt='none', 
             ecolor='green', elinewidth=2, capsize=4, label='Sys Uncertainty')

# Statistical error bars (Red)
plt.errorbar(wavelengths, voltages, yerr=v_stat_err, fmt='o', color='blue',
             ecolor='red', elinewidth=1, capsize=6, label='Stat Uncertainty')
plt.show()


c = 3.00e8  
e = 1.602e-19



frequencies = c / (np.array(wavelengths) * 1e-9) 
res_h = linregress(frequencies, voltages)
h_estimated = res_h.slope * e 
phi_eV = res_h.intercept 
phi_joules = phi_eV * e
# print(f"h estimate: {h_estimated} and work fucntion: {phi_eV}") 
h_stat = res_h.stderr* e
phi_stat = res_h.intercept_stderr
unique_wls = np.array(sorted(wl_col.unique()))
delta_f = (c / ((unique_wls * 1e-9)**2)) * (delta_lambda * 1e-9)
delta_f_range = frequencies.max() - frequencies.min()
mean_v_sys = np.array(v_sys_err).mean()
mean_delta_f = delta_f.mean()

h_sys_V = (mean_v_sys / delta_f_range) * e
h_sys_f = (res_h.slope * (mean_delta_f / delta_f_range)) * e
h_sys = np.sqrt(h_sys_V**2 + h_sys_f**2)

phi_sys = mean_v_sys
h_tot = np.sqrt(h_stat**2 + h_sys**2)
phi_tot = np.sqrt(phi_stat**2 + phi_sys**2)
print("=== RESULTS ===")
print(f"Planck's Constant (h):")
print(f"  Central Value: {h_estimated:.3e} J*s")
print(f"  Stat Error:    ± {h_stat:.3e} J*s")
print(f"  Sys Error:     ± {h_sys:.3e} J*s")
print(f"  Total Error:   ± {h_tot:.3e} J*s")

print(f"\nWork Function (phi):")
print(f"  Central Value: {phi_eV:.3f} eV")
print(f"  Stat Error:    ± {phi_stat:.3f} eV")
print(f"  Sys Error:     ± {phi_sys:.3f} eV")
print(f"  Total Error:   ± {phi_tot:.3f} eV")