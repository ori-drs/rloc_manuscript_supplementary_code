import numpy as np

from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters

import eigenpy
eigenpy.switchToNumpyMatrix()


def compute_stability(comp_dyn=ComputationalDynamics('anymal_coyote'),
                      params=IterativeProjectionParameters(robot_name='anymal_coyote'), comp_geom=ComputationalGeometry(),
                      constraint_mode='FRICTION_AND_ACTUATION', com=None, com_euler=None, com_lin_vel=None,
                      com_lin_acc=None, com_ang_acc=None, ext_force=None, ext_torque=None, feet_position=None, mu=0.5,
                      stance_feet=None, contact_normals=None):
    if com is None:
        com = np.array([0., 0., 0.0])

    if com_euler is None:
        com_euler = np.array([0.0, 0.0, 0.0])

    if com_lin_vel is None:
        com_lin_vel = np.array([0., 0., 0.])

    if com_lin_acc is None:
        com_lin_acc = np.array([0., 0., 0.])

    if com_ang_acc is None:
        com_ang_acc = np.array([0., 0., 0.])

    if ext_force is None:
        ext_force = np.array([0., 0., 0.])

    if ext_torque is None:
        ext_torque = np.array([0., 0., 0.])

    if feet_position is None:
        lf_foot_pos = np.array([0.36, 0.21, -0.47])
        rf_foot_pos = np.array([0.36, -0.21, -0.47])
        lh_foot_pos = np.array([-0.36, 0.21, -0.47])
        rh_foot_pos = np.array([-0.36, -0.21, -0.47])

        """ contact points in the World Frame"""
        feet_position = np.vstack((lf_foot_pos, rf_foot_pos, lh_foot_pos, rh_foot_pos))

    if stance_feet is None:
        ''' stanceFeet vector contains 1 is the foot is on the ground and 0 if it is in the air'''
        stance_feet = [1, 1, 1, 1]

    if contact_normals is None:
        contact_normals = [np.array([0, 0, 1])] * 4
    normals = np.vstack([contact_normals[0], contact_normals[1], contact_normals[2], contact_normals[3]])

    constraint_mode_ip = [constraint_mode] * 4

    ''' extForceW is an optional external pure force (no external torque for now) applied on the CoM of the robot.'''
    ext_centroidal_wrench = np.hstack([ext_force, ext_torque])

    '''You now need to fill the 'params' object with all the relevant 
        informations needed for the computation of the IP'''
    params.setContactsPosWF(feet_position)
    params.setEulerAngles(com_euler)
    params.externalCentroidalWrench = ext_centroidal_wrench
    params.setCoMPosWF(com)
    params.comLinVel = com_lin_vel  # [+- 2.0m/s, +- 2.0m/s, 0.5m/s]
    params.setCoMLinAcc(com_lin_acc)  # [+- 5m/s^2,+- 5m/s^2,+- 5m/s^2]
    params.setCoMAngAcc(com_ang_acc)  # [+- 1rad/s^2,+- 1rad/s^2,+- 1rad/s^2]
    params.setTorqueLims(comp_dyn.robotModel.robotModel.joint_torque_limits)
    params.setActiveContacts(stance_feet)
    params.setConstraintModes(constraint_mode_ip)
    params.setContactNormals(normals)
    params.setFrictionCoefficient(mu)
    params.setTotalMass(comp_dyn.robotModel.robotModel.trunkMass)

    ''' compute iterative projection
    Outputs of "iterative_projection_bretl" are:
    IP_points = resulting 2D vertices
    actuation_polygons = these are the vertices of the 3D force polytopes (one per leg)
    computation_time = how long it took to compute the iterative projection
    '''
    ip_points, force_polytopes, ip_computation_time = comp_dyn.iterative_projection_bretl(params)

    facets = comp_geom.compute_halfspaces_convex_hull(ip_points)
    reference_point = comp_dyn.getReferencePoint(params, "COM")
    point_feasibility, margin = comp_geom.isPointRedundant(facets, reference_point)

    #point_feasibility, margin = comp_dyn.computeMargin(params, "COM")

    return margin
