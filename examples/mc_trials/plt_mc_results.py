import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt
import cProfile

from navlie import GaussianResultList, State, Measurement
from navlie import monte_carlo, MonteCarloResult
from typing import List, Tuple
import pickle
import os

SHOW_FIGS = True
SAVE_FIGS = True
MC_TRIALS = 50
TRIAL_TIME = 3.5

# Load data
with open(
    f"./data/results/monte_carlo/mc_{MC_TRIALS}_trials_{TRIAL_TIME}s.pkl", "rb"
) as f:
    mc_data = pickle.load(f)

results_map: MonteCarloResult = mc_data["results_map"]
results_gmm: MonteCarloResult = mc_data["results_gmm"]
results_gvi: MonteCarloResult = mc_data["results_gvi"]

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")

fig, ax = nav.plot_nees(
    results=results_map,
    confidence_interval=None,
    label="MAP (Cauchy)",
    color="tab:blue",
    normalize=True,
)
fig, ax = nav.plot_nees(
    results=results_gmm,
    ax=ax,
    confidence_interval=None,
    label="MAP (GMM)",
    color="tab:purple",
    normalize=True,
)
fig, ax = nav.plot_nees(
    results=results_gvi,
    ax=ax,
    confidence_interval=0.997,
    label="ESGVI",
    color="tab:orange",
    normalize=True,
)
ax.set_ylabel(r"Normalized Squared Mahalanobis Distance")
ax.set_xlabel("Time (s)")
# ax.set_title(f"aNEES {MC_TRIALS} trials.")
if SAVE_FIGS:
    plt.savefig(
        f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/monte_carlo/se2_aNEES_{MC_TRIALS}_{int(TRIAL_TIME)}s.pdf"
    )
plt.show()

print("Estimation Performance")
print("   Method    | RMSE (rad) |  RMSE (m)  |  aNEES")
print("------------------------------------------------")
print(
    f" MAP (Cauchy)|  {np.mean(results_map.rmse[:,0]):.4f}   |  {np.mean(results_map.rmse[:,1:]):.4f}   |  {(np.mean(results_map.average_nees/results_map.dof)):.3f}"
)
print(
    f" MAP (GMM)   |  {np.mean(results_gmm.rmse[:,0]):.4f}   |  {np.mean(results_gmm.rmse[:,1:]):.4f}   |  {(np.mean(results_gmm.average_nees/results_gmm.dof)):.3f}"
)

print(
    f" ESGVI       |  {np.mean(results_gvi.rmse[:,0]):.4f}   |  {np.mean(results_gvi.rmse[:,1:]):.4f}   |  {(np.mean(results_gvi.average_nees/results_gvi.dof)):.3f}"
)
print(" ----------------------------------------------")

if results_gvi.num_trials < 15:
    fig, axs = plt.subplots(3, 2)
    axs: List[plt.Axes] = axs
    for result in results_gvi.trial_results:
        nav.plot_error(result, axs=axs)

    fig.suptitle("Estimation error")
    plt.tight_layout()
    plt.show()
