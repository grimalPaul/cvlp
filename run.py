import argparse
import os


parser = argparse.ArgumentParser()

path_database = '/scratch_global/stage_pgrimal/data/ViQuAE'
path_output_dir = "/scratch_global/stage_pgrimal/results"
parser.add_argument("--database", type = str, default=path_database)
parser.add_argument("--output_dir", type = str, default = path_output_dir)

arg = parser.parse_args()

# if not os.path.exists(arg.output_dir): os.mkdir(arg.output_dir)