import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

datasets = {
    "Set 1": (
        np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float),
        np.array(
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
            dtype=float,
        ),
    ),
    "Set 2": (
        np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float),
        np.array(
            [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
            dtype=float,
        ),
    ),
    "Set 3": (
        np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float),
        np.array(
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
            dtype=float,
        ),
    ),
    "Set 4": (
        np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float),
        np.array(
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
            dtype=float,
        ),
    ),
}


def linear_model(x,a,b):
    return a*x + b

def run_chi2_analysis(sigma_y):
  fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
  axes = axes.flatten()

  print(f"results for sigma_y={sigma_y}\n\n")

  for idx, (name, (x, y)) in enumerate(datasets.items()):
    ax = axes[idx]
    sigma_array = np.full_like(y, sigma_y)
    popt, pcov = curve_fit(
        linear_model, x, y, sigma=sigma_array, absolute_sigma=True
    )
    a_fit, b_fit = popt
    sigma_a, sigma_b = np.sqrt(np.diag(pcov))

    y_pred = linear_model(x, a_fit, b_fit)
    residuals = y - y_pred
    chi2_val = np.sum((residuals / sigma_y) ** 2)

    n_data = len(x)
    n_params = len(popt)
    ndof = n_data - n_params  # degrees of freedom (11 - 2 = 9)
    chi2_red = chi2_val / ndof
    p_value = chi2.sf(chi2_val, ndof)
    print(f"\n[{name}]")
    print(f"  Slope (a)        : {a_fit:.4f} +/- {sigma_a:.4f}")
    print(f"  Intercept (b)    : {b_fit:.4f} +/- {sigma_b:.4f}")
    print(f"  Chi^2            : {chi2_val:.4f}")
    print(f"  N_dof            : {ndof}")
    print(f"  Reduced Chi^2    : {chi2_red:.4f}")
    print(f"  P(Chi^2, N_dof)  : {p_value:.4f}")
    ax.errorbar(
        x,
        y,
        yerr=sigma_y,
        fmt="o",
        ecolor="gray",
        capsize=3,
        label="Measurements",
    )
    #overplot
    x_line = np.linspace(2, 20, 100)
    y_line = linear_model(x_line, a_fit, b_fit)
    ax.plot(x_line, y_line, "r--", linewidth=1.5, label="Best-fit line")
    annotation = (
        f"Model: $y = ax + b$\n"
        f"$a = {a_fit:.3f} \\pm {sigma_a:.3f}$\n"
        f"$b = {b_fit:.3f} \\pm {sigma_b:.3f}$\n"
        f"$\\chi^2 = {chi2_val:.2f}$ ($N_{{dof}} = {ndof}$)\n"
        f"$\\chi^2 / N_{{dof}} = {chi2_red:.2f}$\n"
        f"$P(\\chi^2, N_{{dof}}) = {p_value:.3f}$"
    )
    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.4),
    )

    ax.set_title(f"{name} ($\sigma_y = {sigma_y}$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(2, 20)
    ax.set_ylim(2, 14)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

  plt.suptitle(
      f"Linear Model",
      fontsize=14,
  )
  plt.tight_layout()
  plt.show()

run_chi2_analysis(sigma_y=2.5)