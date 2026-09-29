import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('photoelectrc_data1.csv')
df.columns = df.columns.str.strip()

#wl_col = [c for c in df.columns if 'wave' in c.lower() or 'lambda' in c.lower()][0]
#v_col = [c for c in df.columns if 'volt' in c.lower() or 'v' in c.lower()][0]
#i_col = [c for c in df.columns if 'curr' in c.lower() or 'i' in c.lower() or 'amp' in c.lower()][0]
#print(v_col)
wl_col = df["Wavelength (nm) +- 2"]
v_col = df["V (V) +- .005"]
i_col = df["I (pA)"]

#unc_col = [c for c in df.columns if 'unc' in c.lower() or 'err' in c.lower()]
#unc_col = unc_col[0] if unc_col else 'Uncertainty'
unc_col = df["uncertainty in I (.0001 default)"]

#df[wl_col] = df[wl_col].ffill()

#if unc_col not in df.columns:
    #df[unc_col] = 0.0001
#else:
unc_col = pd.to_numeric(unc_col, errors='coerce').fillna(0.0001)

v_col = pd.to_numeric(v_col, errors='coerce')
i_col = pd.to_numeric(i_col, errors='coerce')
#df = df.dropna(subset=["V (V) += .005", "I (pA)"])

plt.figure(figsize=(9, 5))
for wl, group in df.groupby(wl_col):
    sorted_group = group.sort_values(by=df.columns[1])
    plt.plot(
        sorted_group[df.columns[1]],
        sorted_group[df.columns[2]],
        marker='o',
        linestyle='none',  # Suppresses connecting line segments
        label=f'lambda= {wl} nm'
    )

plt.xlabel('Voltage (V)')
plt.ylabel('Current (pA)')
plt.title('Photoelectric Effect: Experimental Data Points')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()



from scipy.stats import linregress

def tangent_line_cutoff(v, i):
    sort_idx = np.argsort(v)
    v, i = v[sort_idx], i[sort_idx]
    dI_dV = np.gradient(i, v)
    max_slope_idx = np.argmax(np.abs(dI_dV))

    # neighborhood
    start = max(0, max_slope_idx - 1)
    end = min(len(v), max_slope_idx + 2)

    slope, intercept, _, _, _ = linregress(v[start:end], i[start:end])

    # cutoff V0 where I = m*V + b = 0 -> V0 = -b / m
    v_cutoff = -intercept / slope
    return v_cutoff

tangent_cutoffs = {}
for wl, group in df.groupby(wl_col):
    tangent_cutoffs[wl] = tangent_line_cutoff(group[v_col].values, group[i_col].values)
for wl, v0 in tangent_cutoffs.items():
    print(f"Wavelength: {wl:<12} | Cutoff Voltage (V0): {v0:.4f} V")


def fowler_square_root_cutoff(v, i):
    sort_idx = np.argsort(v)
    v, i = v[sort_idx], i[sort_idx]
    
    # Select non-negative currents near threshold transition
    valid_mask = i > 0
    v_valid, i_valid = v[valid_mask], i[valid_mask]
    
    sqrt_i = np.sqrt(i_valid)
    
    # Fit the lower linear portion of sqrt(I) near threshold
    fit_points = max(3, len(sqrt_i) // 2)
    slope, intercept, _, _, _ = linregress(v_valid[:fit_points], sqrt_i[:fit_points])
    
    v_cutoff = -intercept / slope
    return v_cutoff

fowler_cutoffs = {}
for wl, group in df.groupby(wl_col):
    fowler_cutoffs[wl] = fowler_square_root_cutoff(group[v_col].values, group[i_col].values)

print("fowlers method cutoff voltages:")
for wl, v0 in fowler_cutoffs.items():
    print(f"Wavelength: {wl:<12} | Cutoff Voltage (V0): {v0:.4f} V")
