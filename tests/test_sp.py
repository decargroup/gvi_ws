# %%
import numpy as np
import scipy.linalg
import navlie as nav
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Dict

from gvi_ws.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from gvi_ws.models.models import LaserRangeFinder
from gvi_ws.util.psd import force_sym_PSD
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from gvi_ws.util.cubatures import (
    student_t_cubature,
    unscented_cubature,
    spherical_cubature,
    gh_cubature,
    trans_spherical_cubature,
    trans_gh_cubature,
    trans_unscented_cubature,
)


def test_unit_sigmapoints(method="gh", order=3):
    key1 = "x0"
    projection = np.identity(3)
    state_list = [
        SE2State.random(stamp=0, state_id="x0"),
        SE2State.random(stamp=0.1, state_id="x1"),
    ]
    var_slices = {"x0": slice(0, state_list[0].dof)}
    prior_fac = PriorFactor(
        keys=key1,
        prior_state=state_list[0].copy(),
        prior_covariance=np.identity(3),
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )
    nav_method = method
    if method == "spherical":
        nav_method = "cubature"

    nav_sp, nav_w = generate_sigmapoints(dof=state_list[0].dof, method=nav_method)
    nav_sp = np.round(nav_sp, 10)

    gvi_sp, gvi_w = prior_fac._gen_unit_sigma_pts()
    gvi_sp = np.round(gvi_sp, 10)

    covariance = np.identity(3) * 1e-2
    sqrt_covar = np.linalg.cholesky(covariance)

    gvi_vec_sp = [sqrt_covar @ sp_i.reshape((-1, 1)) for sp_i in gvi_sp]
    nav_vec_sp = sqrt_covar @ nav_sp

    print(gvi_vec_sp)
    print(nav_vec_sp.T)

    def create_weight_dict(points, weights):
        weight_dict = {}
        for point, weight in zip(points, weights):
            weight = round(
                weight, 8
            )  # Ensure floating point precision issues don't cause mismatches
            if weight not in weight_dict:
                weight_dict[weight] = []
            weight_dict[weight].append(
                point.tolist()
            )  # Convert to list for easy comparison

        # Sort points within each weight group to ensure order-independent comparison
        for weight in weight_dict:
            weight_dict[weight].sort()

        return weight_dict

    dict_nav = create_weight_dict(nav_sp.T, nav_w)
    dict_gvi = create_weight_dict(gvi_sp, gvi_w)
    matching = dict_gvi == dict_nav
    assert matching
    if matching:
        print(
            "Passed: Unit sigma points and weights are equal for navlie and personal."
        )
    if not matching:
        print("Failed: Unit sigma points and weights are not equal.")
        print("Navlie Dictionary:", dict_nav)
        print("My Dictionary:", dict_gvi)


def test_transformed_sigmapoints(method="gh", dof=2):
    if method == "gh":
        trans_sp, trans_w = trans_gh_cubature(state_dof=dof, order_p=3)
        sp, w = gh_cubature(state_dof=dof, order_p=3)
    elif method == "spherical":
        trans_sp, trans_w = trans_spherical_cubature(state_dof=dof)
        sp, w = spherical_cubature(state_dof=dof)
    else:
        trans_sp, trans_w = trans_unscented_cubature(state_dof=dof)
        sp, w = unscented_cubature(state_dof=dof, order_p=3)

    def create_weight_dict(points, weights):
        weight_dict = {}
        for point, weight in zip(points, weights):
            weight = round(
                weight, 8
            )  # Ensure floating point precision issues don't cause mismatches
            if weight not in weight_dict:
                weight_dict[weight] = []
            weight_dict[weight].append(
                point.tolist()
            )  # Convert to list for easy comparison

        # Sort points within each weight group to ensure order-independent comparison
        for weight in weight_dict:
            weight_dict[weight].sort()

        return weight_dict

    def compare_weight_dict(dict_1: Dict, dict_2: Dict):
        for key in dict_1.keys():
            sp_1 = np.array(dict_1[key])
            if key not in dict_2.keys():
                return False
            sp_2 = np.array(dict_2[key])
            if not np.allclose(sp_1, sp_2):
                return False
        return True

    dict_trans = create_weight_dict(trans_sp, trans_w)
    dict_sph = create_weight_dict(sp, w)
    matching = compare_weight_dict(dict_sph, dict_trans)
    # assert matching
    if matching:
        print(
            f"Passed: Unit sigma points and weights are equal for {method} and transformed points with n={dof}"
        )
        print("Transformed:", dict_trans)
        print(f"{method}:", dict_sph)

    if not matching:
        print(f"Failed: Sigma points and Weights are not equal for n={dof}.")
        print("transformed:", dict_trans)
        print(f"{method}:", dict_sph)


def test_student_t_sigmapoints():
    unit_sp, w = student_t_cubature(state_dof=2, order_p=5)
    print(unit_sp, w)
    unit_sp_unscented, w_unscented = unscented_cubature(state_dof=2)
    print(unit_sp_unscented, w_unscented)
    unit_sp_gh, w_gh = gh_cubature(state_dof=2, order_p=3)
    print(unit_sp_gh, w_gh)
    fig, ax = plt.subplots(1, 1)
    ax: plt.Axes = ax

    ax.scatter(
        unit_sp_gh[:, 0], unit_sp_gh[:, 1], color="tab:orange", label="Gauss-Hermite"
    )
    # ax.scatter(
    #     unit_sp_unscented[:, 0],
    #     unit_sp_unscented[:, 1],
    #     color="tab:blue",
    #     label="Unscented",
    # )
    ax.scatter(unit_sp[:, 0], unit_sp[:, 1], color="tab:red", label="Student's T")
    circle = Circle(
        (0, 0),
        radius=10,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
        linestyle="-",
    )
    ax.add_patch(circle)

    # Make sure the aspect ratio is equal so the circle looks like a circle
    ax.set_aspect("equal", "box")
    ax.legend()
    # plt.savefig(
    #     f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/sigmapoint_comp_gh.pdf"
    # )
    plt.show()


if __name__ == "__main__":
    VERBOSE = False
    METHOD = "gh"
    ORDER = 2
    # test_unit_sigmapoints(method=METHOD, order=ORDER)
    # test_student_t_sigmapoints()
