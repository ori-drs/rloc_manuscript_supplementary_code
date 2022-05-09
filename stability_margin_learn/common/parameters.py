import random
import numpy as np

from jet_leg_common.jet_leg.computational_geometry.math_tools import Math
from jet_leg_common.jet_leg.robots.robot_model_interface import RobotModelInterface


def stance_feet(low=0, high=1):
    """ stanceFeet vector contains 1 if the foot is on the ground and 0 if it is in the air """
    swing_legs_index = random.randint(low, high)

    stance_feet_list = [1, 1, 1, 1]

    if swing_legs_index == 1:
        index = random.randint(0, 3)
        stance_feet_list[index] = 0

    elif swing_legs_index == 2:
        index_1 = random.randint(0, 1)
        if index_1 == 0:
            index_2 = 3
        elif index_1 == 1:
            index_2 = 2

        stance_feet_list[index_1] = 0
        stance_feet_list[index_2] = 0

    elif swing_legs_index == 3:
        index = random.randint(0, 3)

        stance_feet_list = [0, 0, 0, 0]
        stance_feet_list[index] = 1

    return stance_feet_list


def base_euler(roll_dist=None, pitch_dist=None, yaw_dist=None):
    if roll_dist is None:
        roll_dist = [-np.pi/24.0, np.pi/24.0]

    if pitch_dist is None:
        pitch_dist = [-np.pi/24.0, np.pi/24.0]

    if yaw_dist is None:
        yaw_dist = [-np.pi/18.0, np.pi/18.0]

    return np.array([
        np.random.uniform(roll_dist[0], roll_dist[1]),
        np.random.uniform(pitch_dist[0], pitch_dist[1]),
        np.random.uniform(yaw_dist[0], yaw_dist[1])
    ]).flatten()

def com_positions(robot_name):
    model = RobotModelInterface(robot_name)
    scaling_factor = 0.0

    com_pos = np.array([
        np.random.uniform(-scaling_factor*model.max_dev_from_nominal[0], scaling_factor*model.max_dev_from_nominal[0]),
        np.random.uniform(-scaling_factor*model.max_dev_from_nominal[1], scaling_factor*model.max_dev_from_nominal[1]),
        np.random.uniform(-scaling_factor*model.max_dev_from_nominal[2], scaling_factor*model.max_dev_from_nominal[2])
    ]).flatten()

    return com_pos

def feet_positions(robot_name):
    model = RobotModelInterface(robot_name)
    scaling_factor = 1.5

    lf_foot_pos = np.array([
        np.random.uniform(model.nominal_stance_LF[0] - scaling_factor*model.max_dev_from_nominal[0], model.nominal_stance_LF[0] + scaling_factor*model.max_dev_from_nominal[0]),
        np.random.uniform(model.nominal_stance_LF[1] - scaling_factor*model.max_dev_from_nominal[1], model.nominal_stance_LF[1] + scaling_factor*model.max_dev_from_nominal[1]),
        np.random.uniform(model.nominal_stance_LF[2] - scaling_factor*model.max_dev_from_nominal[2], model.nominal_stance_LF[2] + scaling_factor*model.max_dev_from_nominal[2])
    ]).flatten()

    rf_foot_pos = np.array([
        np.random.uniform(model.nominal_stance_RF[0] - scaling_factor*model.max_dev_from_nominal[0], model.nominal_stance_RF[0] + scaling_factor*model.max_dev_from_nominal[0]),
        np.random.uniform(model.nominal_stance_RF[1] - scaling_factor*model.max_dev_from_nominal[1], model.nominal_stance_RF[1] + scaling_factor*model.max_dev_from_nominal[1]),
        np.random.uniform(model.nominal_stance_RF[2] - scaling_factor*model.max_dev_from_nominal[2], model.nominal_stance_RF[2] + scaling_factor*model.max_dev_from_nominal[2])
    ]).flatten()

    lh_foot_pos = np.array([
        np.random.uniform(model.nominal_stance_LH[0] - scaling_factor*model.max_dev_from_nominal[0], model.nominal_stance_LH[0] + scaling_factor*model.max_dev_from_nominal[0]),
        np.random.uniform(model.nominal_stance_LH[1] - scaling_factor*model.max_dev_from_nominal[1], model.nominal_stance_LH[1] + scaling_factor*model.max_dev_from_nominal[1]),
        np.random.uniform(model.nominal_stance_LH[2] - scaling_factor*model.max_dev_from_nominal[2], model.nominal_stance_LH[2] + scaling_factor*model.max_dev_from_nominal[2])
    ]).flatten()

    rh_foot_pos = np.array([
        np.random.uniform(model.nominal_stance_RH[0] - scaling_factor*model.max_dev_from_nominal[0], model.nominal_stance_RH[0] + scaling_factor*model.max_dev_from_nominal[0]),
        np.random.uniform(model.nominal_stance_RH[1] - scaling_factor*model.max_dev_from_nominal[1], model.nominal_stance_RH[1] + scaling_factor*model.max_dev_from_nominal[1]),
        np.random.uniform(model.nominal_stance_RH[2] - scaling_factor*model.max_dev_from_nominal[2], model.nominal_stance_RH[2] + scaling_factor*model.max_dev_from_nominal[2])
    ]).flatten()

    return np.vstack((lf_foot_pos, rf_foot_pos, lh_foot_pos, rh_foot_pos))


def contact_normals(roll_dist=None, pitch_dist=None, math_tools=Math()):
    if roll_dist is None:
        roll_dist = [0, np.pi / 16]

    if pitch_dist is None:
        pitch_dist = [0, np.pi / 16]

    contact_normals_list = []

    for _ in range(4):
        angles = np.array([
            np.random.normal(roll_dist[0], roll_dist[1]),
            np.random.normal(pitch_dist[0], pitch_dist[1]),
            0.0
        ]).flatten()

        contact_normals_list.append(
            np.transpose(np.transpose(math_tools.rpyToRot(angles[0], angles[1], angles[2])).dot(np.array([0, 0, 1]))))

    return contact_normals_list


def base_external_force(distribution=None):
    if distribution is None:
        distribution = [0, 0, 0]

    return np.array([
        np.random.normal(0, distribution[0]),
        np.random.normal(0, distribution[1]),
        np.random.normal(0, distribution[2]),
    ]).flatten()


def friction_coeff(limit_range=None):
    if limit_range is None:
        limit_range = [0.3, 0.7]

    return np.random.uniform(limit_range[0], limit_range[1])


def linear_velocity(x_dist=None, y_dist=None, z_dist=None):
    if x_dist is None:
        x_dist = [0.0, 0.2]

    if y_dist is None:
        y_dist = [0.0, 0.2]

    if z_dist is None:
        z_dist = [0, 0.05]

    return np.array([
        np.random.normal(x_dist[0], x_dist[1]),
        np.random.normal(y_dist[0], y_dist[1]),
        np.random.normal(z_dist[0], z_dist[1])
    ]).flatten()


def linear_acceleration(x_dist=None, y_dist=None, z_dist=None):
    if x_dist is None:
        x_dist = [-4, 4]

    if y_dist is None:
        y_dist = [-4, 4]

    if z_dist is None:
        z_dist = [-0, 0]

    return np.array([
        np.random.uniform(x_dist[0], x_dist[1]),
        np.random.uniform(y_dist[0], y_dist[1]),
        np.random.uniform(z_dist[0], z_dist[1])
    ]).flatten()


def angular_acceleration(x_dist=None, y_dist=None, z_dist=None):
    if x_dist is None:
        x_dist = [0.0, 0.2]

    if y_dist is None:
        y_dist = [0.0, 0.2]

    if z_dist is None:
        z_dist = [0, 0.3]

    return np.array([
        np.random.normal(x_dist[0], x_dist[1]),
        np.random.normal(y_dist[0], y_dist[1]),
        np.random.normal(z_dist[0], z_dist[1])
    ]).flatten()


def base_external_torque(distribution=None):
    if distribution is None:
        distribution = [0, 0, 0]

    return np.array([
        np.random.normal(0, distribution[0]),
        np.random.normal(0, distribution[1]),
        np.random.normal(0, distribution[2]),
    ]).flatten()
