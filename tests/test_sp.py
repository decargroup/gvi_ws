import os
import sys

# Get the absolute path of the project root (one level above "test")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Change the working directory to the project root
os.chdir(PROJECT_ROOT)

# Add project root to sys.path so Python finds 'src'
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import scipy.linalg
import navlie as nav
from src.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from src.models.models import LaserRangeFinder
from src.util.psd import force_sym_PSD
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from src.util.cubatures import (
    student_t_cubature,
    unscented_cubature,
    spherical_cubature,
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

    m = np.zeros(3)
    for i, w in enumerate(gvi_w):
        m += w * gvi_sp[i]
    print(m)


def test_student_t_sigmapoints():
    pass


if __name__ == "__main__":
    VERBOSE = False
    METHOD = "unscented"
    ORDER = 2

    test_unit_sigmapoints(method=METHOD, order=ORDER)
