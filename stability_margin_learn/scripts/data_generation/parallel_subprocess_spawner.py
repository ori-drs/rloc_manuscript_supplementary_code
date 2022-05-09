import os
import psutil
import thread

from common.paths import ProjectPaths


def main():
    num_cpu = psutil.cpu_count(True)

    project_paths = ProjectPaths()
    script_path = project_paths.SCRIPTS_PATH + '/data_generation/respawning_margin_data_generator.sh'

    for _ in range(num_cpu):
        thread.start_new_thread(os.system, ('sh ' + script_path,))

    while True:
        pass


if __name__ == '__main__':
    main()
