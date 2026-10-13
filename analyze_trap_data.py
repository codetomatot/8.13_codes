import csv
import math
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent / "3mm_bead_trap_and_calibrate"
OUT = Path(__file__).resolve().parent / "trap_analysis_results"
FS_HZ = 5000.0
DRIVE_F_HZ = 0.5
STAGE_AMPLITUDE_UM = 5.0
TEMPERATURE_K = 298.15
WATER_VISCOSITY_PA_S = 0.00089
BEAD_DIAMETER_UM = 3.0
FIT_F_MIN_HZ = 1.0
FIT_F_MAX_HZ = 1500.0
WELCH_NPERSEG = 8192
N_LOG_BINS = 70
K_B_SI = 1.380649e-23

GROUPS = [
    dict(power_mA=225.8, free="z32 225-8 mA.txt", oscX="z44_226-0_mA_fixed_oscX.txt", oscY="z4_225-9_mA_fixed_oscY.txt"),
    dict(power_mA=243.4, free="z32 243-4 mA.txt", oscX="z44_243-4_mA_fixed_oscX.txt", oscY="z4_243-4_mA_fixed_oscY.txt"),
    dict(power_mA=282.6, free="z32 282-6 mA.txt", oscX="z44_282-3_mA_fixed_oscX.txt", oscY="z4_282-4_mA_fixed_oscY.txt"),
    dict(power_mA=316.0, free="z32 316-0 mA.txt", oscX="z44_315-8_mA_fixed_oscX.txt", oscY="z4_316-0_mA_fixed_oscY.txt"),
    dict(power_mA=339.8, free="z32 339-8 mA.txt", oscX="z44_339-9_mA_fixed_oscX.txt", oscY="z4_339-8_mA_fixed_oscY.txt"),
    dict(power_mA=365.5, free="z32 365-5 mA.txt", oscX="z44_365-5_ma_fixed_oscX.txt", oscY="z4_365-7_mA_fixed_oscY.txt"),
    dict(power_mA=400.1, free="z32 400-1 mA.txt", oscX="z44_400-0_mA_fixed_oscX.txt", oscY="z4_400-0_mA_fixed_oscY.txt"),
    dict(power_mA=421.4, free="z32 421-4 mA.txt", oscX="z44_421-5_mA_fixed_oscX.txt", oscY="z4_421-3_mA_fixed_oscY.txt"),
]

FREE_QPD_COL = {"X": 0, "Y": 1}
CAL_QPD_COL = {"X": 0, "Y": 1}
CAL_STAGE_COL = {"X": 2, "Y": 3}


def read_data(filename: str) -> np.ndarray:
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in {filename}; found shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values in {filename}")
    return arr


def sinusoid_amplitude_phase(y: np.ndarray, fs: float, f_hz: float) -> tuple[float, float, float]:
    t = np.arange(len(y), dtype=float) / fs
    w = 2.0 * np.pi * f_hz
    design = np.column_stack((np.sin(w * t), np.cos(w * t), np.ones(len(t))))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    amp = float(np.hypot(coef[0], coef[1]))
    phase = float(np.arctan2(coef[1], coef[0]))
    residual = y - design @ coef
    amp_sigma = float(np.sqrt(2.0 / len(y)) * np.std(residual, ddof=1))
    snr = amp / amp_sigma if amp_sigma > 0 else float("inf")
    return amp, phase, snr


def log_bin_spectrum(freq: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    high = min(FIT_F_MAX_HZ, float(freq[-1]))
    edges = np.geomspace(FIT_F_MIN_HZ, high, N_LOG_BINS + 1)
    out_f, out_p = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freq >= lo) & (freq < hi) & (psd > 0)
        if np.count_nonzero(m) >= 2:
            out_f.append(float(np.sqrt(lo * hi)))
            out_p.append(float(np.median(psd[m])))
    return np.asarray(out_f), np.asarray(out_p)


def fit_lorentzian(voltage: np.ndarray, fs: float) -> dict:
    nperseg = min(WELCH_NPERSEG, len(voltage))
    freq, psd = welch(
        voltage - np.mean(voltage), fs=fs, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant", scaling="density",
    )
    fb, pb = log_bin_spectrum(freq, psd)
    use = (fb >= FIT_F_MIN_HZ) & (fb <= FIT_F_MAX_HZ)
    use &= ~(((fb >= 30.0) & (fb <= 42.0)) | ((fb >= 72.0) & (fb <= 84.0)))
    ff, pp = fb[use], pb[use]
    if len(ff) < 12:
        raise RuntimeError("Too few frequency bins remain for a Lorentzian fit")

    high_floor = float(np.median(pp[ff >= np.quantile(ff, 0.75)]))
    low_level = float(np.median(pp[ff <= np.quantile(ff, 0.15)]))
    high_floor = max(high_floor, np.finfo(float).tiny)
    low_level = max(low_level, high_floor * 2.0)

    def residual(q: np.ndarray) -> np.ndarray:
        fc, a, c = np.exp(q)
        model = c + a / (ff**2 + fc**2)
        return np.log(model) - np.log(pp)

    best = None
    for fc0 in (2.0, 5.0, 10.0, 20.0, 40.0):
        a0 = max((low_level - high_floor) * fc0**2, 1e-18)
        q0 = np.log([fc0, a0, high_floor])
        lower = np.log([0.05, 1e-18, 1e-18])
        upper = np.log([1000.0, 1e2, 1e-2])
        res = least_squares(
            residual, q0, bounds=(lower, upper), loss="soft_l1", f_scale=0.3,
            max_nfev=20000,
        )
        err = float(np.sqrt(np.mean(residual(res.x) ** 2)))
        if best is None or err < best[0]:
            best = (err, res.x)

    assert best is not None
    rmse_log, q = best
    fc, a, c = np.exp(q)
    model_full = c + a / (freq**2 + fc**2)
    return {
        "freq_hz": freq, "psd_v2_hz": psd, "binned_freq_hz": fb,
        "binned_psd_v2_hz": pb, "fit_freq_hz": ff, "fit_psd_v2_hz": pp,
        "fc_hz": float(fc), "A_v2_hz": float(a), "noise_floor_v2_hz": float(c),
        "fit_log_rmse": rmse_log, "fit_psd_v2_hz_full": model_full,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    radius_m = 0.5 * BEAD_DIAMETER_UM * 1e-6
    drag_gamma = 6.0 * np.pi * WATER_VISCOSITY_PA_S * radius_m
    stage_amp_m = STAGE_AMPLITUDE_UM * 1e-6

    summary_rows = []
    calibration_rows = []
    plot_results = []
    group_report = []

    for group in GROUPS:
        free_arr = read_data(group["free"])
        item = {"power_mA": group["power_mA"], "free_file": group["free"]}
        group_report.append({
            "power_mA": group["power_mA"], "free_bead_file": group["free"],
            "oscX_calibration_file": group["oscX"], "oscY_calibration_file": group["oscY"],
            "free_samples": len(free_arr), "free_duration_s_at_5000_Hz": len(free_arr) / FS_HZ,
            "pairing_method": "explicit numeric-label matching; see script GROUPS map",
        })

        for axis in ("X", "Y"):
            free_col = FREE_QPD_COL[axis]
            cal_file = group["oscX"] if axis == "X" else group["oscY"]
            cal_arr = read_data(cal_file)
            qpd_amp, qpd_phase, lockin_snr = sinusoid_amplitude_phase(
                cal_arr[:, CAL_QPD_COL[axis]], FS_HZ, DRIVE_F_HZ
            )
            stage_amp_signal, stage_phase, stage_snr = sinusoid_amplitude_phase(
                cal_arr[:, CAL_STAGE_COL[axis]], FS_HZ, DRIVE_F_HZ
            )

            free_fit = fit_lorentzian(free_arr[:, free_col], FS_HZ)
            fc = free_fit["fc_hz"]

            # for an overdamped trapped bead driven by moving fluid/stage, the bead
            # displacement amplitude is A_stage * f_drive/sqrt(f_drive^2+fc^2).
            response_fraction = DRIVE_F_HZ / np.sqrt(DRIVE_F_HZ**2 + fc**2)
            bead_amp_m = stage_amp_m * response_fraction
            gain_v_per_m = qpd_amp / bead_amp_m if bead_amp_m > 0 else np.nan
            if not np.isfinite(gain_v_per_m) or gain_v_per_m <= 0:
                raise RuntimeError(f"Invalid calibration gain for {group['power_mA']} mA {axis}")

            # rescale fitted PSD amplitude from V^2 to m^2. Frequency is unchanged.
            A_m2_hz = free_fit["A_v2_hz"] / gain_v_per_m**2
            C_m2_hz = free_fit["noise_floor_v2_hz"] / gain_v_per_m**2
            diffusion_m2_s = np.pi**2 * A_m2_hz
            stiffness_N_m = 2.0 * np.pi * drag_gamma * fc
            kb_psd = diffusion_m2_s * drag_gamma / TEMPERATURE_K
            sample_var_m2 = float(np.var((free_arr[:, free_col] - np.mean(free_arr[:, free_col])) / gain_v_per_m, ddof=1))
            kb_equip = stiffness_N_m * sample_var_m2 / TEMPERATURE_K
            variance_from_fit_m2 = A_m2_hz * np.pi / (2.0 * fc)

            cal_phase_diff = float(np.angle(np.exp(1j * (qpd_phase - stage_phase))))
            row = {
                "power_mA": group["power_mA"], "axis": axis,
                "free_file": group["free"], "calibration_file": cal_file,
                "free_samples": len(free_arr), "free_duration_s": len(free_arr) / FS_HZ,
                "sample_rate_Hz_assumed": FS_HZ,
                "qpd_drive_amplitude_V": qpd_amp, "qpd_drive_lockin_SNR": lockin_snr,
                "strain_gauge_drive_amplitude_raw_units": stage_amp_signal,
                "strain_gauge_drive_lockin_SNR": stage_snr,
                "qpd_minus_stage_phase_rad_diagnostic_only": cal_phase_diff,
                "stage_amplitude_assumed_um": STAGE_AMPLITUDE_UM,
                "model_bead_response_fraction": response_fraction,
                "qpd_sensitivity_V_per_um": gain_v_per_m * 1e-6,
                "corner_frequency_Hz": fc,
                "trap_stiffness_N_per_m": stiffness_N_m,
                "trap_stiffness_pN_per_um": stiffness_N_m * 1e6,
                "trap_stiffness_pN_per_nm": stiffness_N_m * 1e3,
                "lorentzian_A_V2_Hz": free_fit["A_v2_hz"],
                "lorentzian_A_m2_Hz": A_m2_hz,
                "white_noise_floor_m2_per_Hz": C_m2_hz,
                "diffusion_coefficient_m2_per_s": diffusion_m2_s,
                "kB_from_PSD_J_per_K": kb_psd,
                "kB_from_equipartition_raw_sample_variance_J_per_K": kb_equip,
                "kB_PSD_over_CODATA": kb_psd / K_B_SI,
                "kB_equipartition_over_CODATA": kb_equip / K_B_SI,
                "sample_variance_position_m2": sample_var_m2,
                "variance_from_lorentzian_m2": variance_from_fit_m2,
                "lorentzian_log_RMSE": free_fit["fit_log_rmse"],
                "quality_flag": (
                    "CHECK FIT/CALIBRATION: large PSD log-RMSE"
                    if free_fit["fit_log_rmse"] > 0.50 else
                    "PROVISIONAL: assumptions and calibration diagnostics must be checked"
                ),
            }
            summary_rows.append(row)
            plot_results.append((group["power_mA"], axis, free_fit, gain_v_per_m))
            calibration_rows.append({
                "power_mA": group["power_mA"], "axis": axis, "calibration_file": cal_file,
                "qpd_drive_amplitude_V": qpd_amp, "qpd_drive_phase_rad": qpd_phase,
                "qpd_drive_lockin_SNR": lockin_snr,
                "strain_gauge_drive_amplitude_raw_units": stage_amp_signal,
                "strain_gauge_drive_phase_rad": stage_phase,
                "strain_gauge_drive_lockin_SNR": stage_snr,
                "qpd_minus_stage_phase_rad": cal_phase_diff,
                "corner_frequency_Hz_used_for_dynamic_correction": fc,
                "assumed_physical_stage_amplitude_um": STAGE_AMPLITUDE_UM,
                "model_response_fraction": response_fraction,
                "estimated_sensitivity_V_per_um": gain_v_per_m * 1e-6,
            })

    summary = pd.DataFrame(summary_rows).sort_values(["power_mA", "axis"])
    calibrations = pd.DataFrame(calibration_rows).sort_values(["power_mA", "axis"])
    manifest = pd.DataFrame(group_report)
    summary.to_csv(OUT / "summary_metrics.csv", index=False, float_format="%.8g")
    calibrations.to_csv(OUT / "calibration_diagnostics.csv", index=False, float_format="%.8g")
    manifest.to_csv(OUT / "grouping_manifest.csv", index=False)

    fig, axes = plt.subplots(len(GROUPS), 2, figsize=(13, 3.0 * len(GROUPS)), sharex=True)
    for ax, (power, axis, fit, gain_v_per_m) in zip(axes.flat, plot_results):
        scale2 = 1.0 / gain_v_per_m**2
        f = fit["freq_hz"]
        p = fit["psd_v2_hz"] * scale2
        model = fit["fit_psd_v2_hz_full"] * scale2
        ax.loglog(f[1:], p[1:], linewidth=0.7, alpha=0.60, label="Welch PSD")
        ax.loglog(f[1:], model[1:], linewidth=1.8, label="Lorentzian + floor")
        ax.set_title(f"{power:.1f} mA — {axis}; $f_c$={fit['fc_hz']:.2f} Hz")
        ax.set_ylabel("PSD (m$^2$/Hz)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")
    fig.suptitle("Calibrated position PSDs and Lorentzian fits (provisional calibration)", y=1.002)
    fig.tight_layout()
    fig.savefig(OUT / "psd_lorentzian_fits.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis in ("X", "Y"):
        sub = summary[summary.axis == axis].sort_values("power_mA")
        label = f"{axis}"
        axes[0, 0].plot(sub.power_mA, sub.corner_frequency_Hz, marker="o", label=label)
        axes[0, 1].plot(sub.power_mA, sub.trap_stiffness_pN_per_um, marker="o", label=label)
        axes[1, 0].semilogy(sub.power_mA, sub.kB_from_PSD_J_per_K, marker="o", label=label)
        axes[1, 1].semilogy(sub.power_mA, sub.kB_from_equipartition_raw_sample_variance_J_per_K, marker="o", label=label)
    axes[0, 0].set_title("Corner frequency")
    axes[0, 0].set_ylabel("$f_c$ (Hz)")
    axes[0, 1].set_title("Trap stiffness")
    axes[0, 1].set_ylabel(r"$\kappa$ (pN/µm)")
    axes[1, 0].set_title("Boltzmann estimate from PSD amplitude")
    axes[1, 0].set_ylabel("$k_B$ (J/K)")
    axes[1, 1].set_title("Equipartition check using sample variance")
    axes[1, 1].set_ylabel("$k_B$ (J/K)")
    axes[1, 0].axhline(K_B_SI, color="black", linestyle="--", linewidth=1, label="Exact SI value")
    axes[1, 1].axhline(K_B_SI, color="black", linestyle="--", linewidth=1, label="Exact SI value")
    for ax in axes.flat:
        ax.set_xlabel("Laser power label (mA)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "corner_stiffness_boltzmann_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis in ("X", "Y"):
        sub = calibrations[calibrations.axis == axis].sort_values("power_mA")
        axes[0].plot(sub.power_mA, sub.qpd_drive_amplitude_V, marker="o", label=axis)
        axes[1].plot(sub.power_mA, sub.strain_gauge_drive_amplitude_raw_units, marker="o", label=axis)
        axes[2].plot(sub.power_mA, sub.estimated_sensitivity_V_per_um, marker="o", label=axis)
    axes[0].set_ylabel("QPD amplitude at 0.5 Hz (V)")
    axes[1].set_ylabel("Strain-gauge amplitude\n(raw units)")
    axes[2].set_ylabel("Estimated sensitivity (V/µm)")
    axes[2].set_xlabel("Laser power label (mA)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Calibration diagnostics; stage amplitude is assumed to be 5 µm")
    fig.tight_layout()
    fig.savefig(OUT / "calibration_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    medians = summary.groupby("axis")["kB_from_PSD_J_per_K"].median().to_dict()
    ratio_xy = medians.get("X", float("nan")) / medians.get("Y", float("nan"))
    text_lines = [
        "OPTICAL TRAP ANALYSIS, DATA/ASSUMPTION NOTES",
        "",
        f"Free-bead sample rate used: {FS_HZ:g} Hz.",
        "Eight explicit free/X/Y groups were analyzed; pairings are in grouping_manifest.csv.",
        "All ordinary free-bead files have 25,000 rows, i.e. 5 s at 5 kHz.",
        "free_bead_settings.txt says Seconds: 102, so the duration metadata conflicts with the actual row count.",
        "The 482.5 mA file has a large 0.5 Hz strain-gauge oscillation and was not assigned to a group.",
        f"Physical assumptions: stage amplitude {STAGE_AMPLITUDE_UM:g} µm; bead diameter {BEAD_DIAMETER_UM:g} µm;",
        f"water viscosity {WATER_VISCOSITY_PA_S:g} Pa s; temperature {TEMPERATURE_K:g} K; no wall correction.",
        "The dynamic active-calibration formula is model-dependent and must be checked against the apparatus.",
        "Calibration diagnostics also show substantially different X/Y sensitivities; check channel ordering and the meaning/units of the 5.0 amplitude.",
        "Do not treat k_B values as validated if X/Y estimates disagree or differ substantially from the SI value.",
        f"Median PSD-based k_B by axis: X={medians.get('X', float('nan')):.4g} J/K; Y={medians.get('Y', float('nan')):.4g} J/K; X/Y median ratio={ratio_xy:.3g}.",
        f"Reference exact SI value: {K_B_SI:.7g} J/K.",
        "Use the fit RMSE, lock-in SNR, calibration plot and equipartition comparison to decide which runs are usable.",
    ]


    (OUT / "analysis_notes.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")


    cols = ["power_mA", "axis", "corner_frequency_Hz", "trap_stiffness_pN_per_um",
            "qpd_sensitivity_V_per_um", "kB_from_PSD_J_per_K", "lorentzian_log_RMSE"]
    print("\nSUMMARY (provisional; assumptions documented in analysis_notes.txt)")
    print(summary[cols].to_string(index=False, formatters={
        "corner_frequency_Hz": "{:.2f}".format,
        "trap_stiffness_pN_per_um": "{:.3f}".format,
        "qpd_sensitivity_V_per_um": "{:.3f}".format,
        "kB_from_PSD_J_per_K": "{:.3e}".format,
        "lorentzian_log_RMSE": "{:.2f}".format,
    }))
    print(f"\nWrote results to: {OUT}")


if __name__ == "__main__":
    main()
