import numpy as np
import sys
import scipy.stats as stats
from scipy.signal import welch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

T = 298.15
eta = 0.89e-3
r = 3.21e-6

fs = 5000 #sample rate in settings

# strain gauge calibartiotn
stage_scan_file = "optical_trapping_number_one_data/z32 225-8 mA.txt"

try:
    stage_data = np.loadtxt(stage_scan_file, skiprows=0) 
    x_stage_um = stage_data[:, 0]  # Known physical distance (um)
    v_sg_cal = stage_data[:, 2]    # Strain Gauge X (Volts)
    
    # Linear fit: V_SG = S_SG * x + intercept
    res_sg = stats.linregress(x_stage_um, v_sg_cal)
    S_sg = res_sg.slope             # V / um
    sigma_S_sg = res_sg.stderr      # Uncertainty in slope
    C_sg = 1.0 / S_sg               # um / V
    
    print(f"--- Strain Gauge Calibration ---")
    print(f"S_SG = {S_sg:.4f} ± {sigma_S_sg:.4f} V/um")
    print(f"C_SG = {C_sg:.4f} um/V\n")
    
except IOError:
    print("file not found")
    sys.exit(1)


file_list = [
    "z32 225-8 mA.txt", "z32 243-4 mA.txt", "z32 282-6 mA.txt",
    "z32 316-0 mA.txt", "z32 339-8 mA.txt", "z32 365-5 mA.txt", "z32 400-1 mA.txt", "z32 421-4 mA.txt"
]
laser_currents = [225.8, 243.4, 282.6, 316.0, 339.8, 365.6, 400.1, 421.4] # mA

alpha_list = []
sigma_alpha_list = []
fc_list = []
P0_list = []

def lorentzian(f, P0, fc):
    # lorentzian spectral density
    return P0 / (1.0 + (f / fc)**2)

def compute_msd(x_signal, max_lags):
    # mean square displacment
    msd = np.zeros(max_lags)
    N = len(x_signal)
    for lag in range(1, max_lags):
        diffs = x_signal[lag:] - x_signal[:N - lag]
        msd[lag] = np.mean(diffs**2)
    return msd

fig, (ax_psd, ax_msd) = plt.subplots(1, 2, figsize=(13, 5))

for idx, fname in enumerate(file_list):
    # Data columns: [0: QPD_X (mV), 1: QPD_Y (mV), 2: SG_X (V), 3: SG_Y (V)] assumed
    data = np.loadtxt("optical_trapping_number_one_data/"+str(fname))
    
    qpd_x_mv = data[:, 0]
    sg_x_v   = data[:, 2]
    
    # linear fit of central scan region: QPD (mV) vs Strain Gauge (V)
    # adjust slicing [100:1000] to isolate the central linear portion of scan
    fit_slice = slice(100, min(1000, len(qpd_x_mv)))
    
    res_rel = stats.linregress(sg_x_v[fit_slice], qpd_x_mv[fit_slice])
    S_rel = res_rel.slope           # mV / V
    sigma_S_rel = res_rel.stderr    # Uncertainty in slope
    
    # alpha calculation and uncertainty included
    alpha_um_per_mv = (1.0 / S_rel) * C_sg
    alpha_nm_per_mv = alpha_um_per_mv * 1000.0  # nm / mV
    
    # relative uncertainty propagation
    rel_err_sq = (sigma_S_rel / S_rel)**2 + (sigma_S_sg / S_sg)**2
    sigma_alpha_nm = alpha_nm_per_mv * np.sqrt(rel_err_sq)
    
    alpha_list.append(alpha_nm_per_mv)
    sigma_alpha_list.append(sigma_alpha_nm)
    
    # Convert QPD (mV) -> Distance (meters & nm)
    qpd_x_centered = qpd_x_mv - np.mean(qpd_x_mv)
    x_pos_m = qpd_x_centered * alpha_nm_per_mv * 1e-9  # meters
    x_pos_nm = qpd_x_centered * alpha_nm_per_mv         # nm

    # Power Spectral Density (PSD)
    nperseg = int(fs / 10)
    freqs, Pxx_m2_hz = welch(x_pos_m, fs=fs, window='hann', nperseg=nperseg)
    
    # filter frequencies
    mask = (freqs > 10) & (freqs < fs / 4)
    f_fit = freqs[mask]
    P_fit = Pxx_m2_hz[mask]
    
    # fit Lorentzian
    p0_initial = [np.max(P_fit), 100.0]
    popt, _ = curve_fit(lorentzian, f_fit, P_fit, p0=p0_initial, bounds=(0, [np.inf, fs/2]))
    
    P0_fit, fc_fit = popt
    # k_fit = 2 * np.pi * gamma * fc_fit # Trap stiffness (N/m)
    
    fc_list.append(fc_fit)
    P0_list.append(P0_fit)
    # k_list.append(k_fit)
    
    # Plot PSD (nm^2 / Hz)
    ax_psd.loglog(freqs, Pxx_m2_hz * 1e18, color='lightgray', alpha=0.4)
    ax_psd.loglog(f_fit, lorentzian(f_fit, *popt) * 1e18, linewidth=2,
                  label=f"{laser_currents[idx]} mA (fc = {fc_fit:.1f} Hz)")
    
    #mean square displacement
    max_lags = int(fs * 0.05) # Calculate up to 50 ms lag time
    msd_nm2 = compute_msd(x_pos_nm, max_lags)
    tau_s = np.arange(max_lags) / fs
    
    ax_msd.plot(tau_s * 1000, msd_nm2, label=f"{laser_currents[idx]} mA")

ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("PSD (nm^2 / Hz)")
ax_psd.set_title("Calibrated Power Spectral Density")
ax_psd.grid(True, which="both", ls="--")
ax_psd.legend()

ax_msd.set_xlabel("Lag Time t (ms)")
ax_msd.set_ylabel("MSD ⟨delta x^2⟩ (nm^2)")
ax_msd.set_title("Mean Squared Displacement vs. Laser Power")
ax_msd.grid(True)
ax_msd.legend()

plt.tight_layout()
plt.show()

print("--- CALIBRATION & CORNER FREQUENCY RESULTS ---")
print(f"{'Current (mA)':<15}{'alpha (nm/mV)':<20}{'Uncertainty (nm/mV)':<25}{'fc (Hz)':<15}{'P0 (nm^2/Hz)':<15}")
print("-" * 90)
for i in range(len(laser_currents)):
    print(f"{laser_currents[i]:<15}{alpha_list[i]:<20.4f}{sigma_alpha_list[i]:<25.4f}{fc_list[i]:<15.2f}{P0_list[i]:<15.4e}")