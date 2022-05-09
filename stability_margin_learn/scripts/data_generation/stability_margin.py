import os
import time
import string

from common.parameters import *
from common.jet_leg_interface import compute_stability
from common.paths import ProjectPaths

from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters

CONSTRAINTS = 'FRICTION_AND_ACTUATION'
MAX_ITERATIONS = 1000 iterations correspond to approximately 58.2106878757s
COMPUTE_JACOBIAN = False
STORE_BINARY_MATRIX = False


def main():
    paths = ProjectPaths()
    math_tools = Math()
    robot_name = 'anymal_coyote'
    comp_dyn = ComputationalDynamics(robot_name)
    params = IterativeProjectionParameters(robot_name=robot_name)
    comp_geom = ComputationalGeometry()

    save_path = paths.DATA_PATH + '/stability_margin/' + paths.INIT_DATETIME_STR + '/' + ''.join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    file_name = 'data_'
    file_suffix = 0

    try:
        os.makedirs(save_path)
    except OSError:
        pass

    exceptions = 0
    start_time = time.time()

    iteration = 0

    for _ in range(MAX_ITERATIONS):
        print ('\n\n----\nIteration', iteration + 1, 'of', MAX_ITERATIONS, '| Exceptions:', exceptions, \
            ' | Elapsed Time:', time.time() - start_time, 'seconds')

        stance_feet_list = stance_feet(high=2)  # Get random stance configuration
        com_world = com_positions(robot_name)  # Center of Mass in world frame
        com_lin_vel = linear_velocity()  # Random Linear Velocity
        com_lin_acc = linear_acceleration()  # Random COM linear acceleration
        com_ang_acc = angular_acceleration()  # Random COM angular acceleration
        friction = friction_coeff()  # Gazebo uses 0.8
        feet_pos = feet_positions(robot_name)  # Random positions of foot contacts
        feet_contact_normals = contact_normals(math_tools=math_tools)  # List of Normal Rotations for each contact
        ext_force_world = base_external_force()  # External force applied to base in world frame
        ext_torque_world = base_external_torque()
        com_euler = base_euler()

        try:
            stability_margin = compute_stability(
                comp_dyn=comp_dyn,
                params=params,
                comp_geom=comp_geom,
                constraint_mode=CONSTRAINTS,
                com=com_world,
                com_euler=com_euler,
                com_lin_vel=com_lin_vel,
                com_lin_acc=com_lin_acc,
                com_ang_acc=com_ang_acc,
                ext_force=ext_force_world,
                ext_torque=ext_torque_world,
                feet_position=feet_pos,
                mu=friction,
                stance_feet=stance_feet_list,
                contact_normals=feet_contact_normals
            )

            print ('Stability Margin:', stability_margin)

            parameters = np.concatenate([
                np.array(com_euler).flatten(),
                np.array(com_lin_vel).flatten(),
                np.array(com_lin_acc).flatten(),
                np.array(com_ang_acc).flatten(),
                np.array(ext_force_world).flatten(),
                np.array(ext_torque_world).flatten(),
                np.array(feet_pos).flatten(),
                np.array([friction]).flatten(),
                np.array(stance_feet_list).flatten(),
                np.array(feet_contact_normals).flatten(),
            ])

            parameters = np.array(parameters, dtype=np.float)

            results = np.concatenate([
                np.array([stability_margin]).flatten()
            ])

            results = np.array(results, dtype=np.float)

            ''' Save to a file. Create it if it doesn't exist. '''
            f = open(save_path + '/' + file_name + str(file_suffix).zfill(4) + '.csv', 'ab')
            np.savetxt(f, np.concatenate([parameters, results]).flatten(), delimiter=',', fmt='%1.6f', newline=',')
            f.seek(-1, os.SEEK_END)
            f.truncate()
            f.write(b"\n")
            f.close()

            iteration += 1

            # Create a new file after every 50k samples
            if iteration % 50000 == 0:
                file_suffix += 1

        except Exception as e:
            exceptions += 1
            print ('Exception Occurred ', e)

    print ('\n\nTotal Execution Time:', time.time() - start_time, 'seconds\n')


if __name__ == '__main__':
    main()
