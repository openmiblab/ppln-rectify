import os, sys
import logging
from pathlib import Path
import argparse



def setup_logging(build, pipeline):
    dir_logs = os.path.join(build, pipeline)
    os.makedirs(dir_logs, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{dir_logs}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    

def stage_output_dir(build, pipeline, module):

    # Outputs of the stage
    stage = Path(module).name[:-3]
    dir_output = os.path.join(build, pipeline, stage)
    os.makedirs(dir_output, exist_ok=True)
    return dir_output


def run_script(run, default_build, pipeline):
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=default_build, help="Build folder")
    args = parser.parse_args()

    setup_logging(args.build, pipeline)

    run(args.build)