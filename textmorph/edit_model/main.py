import argparse

from gtd.io import save_stdout
from gtd.log import set_log_level
from gtd.utils import Config
from textmorph.edit_model.training_run import EditTrainingRuns

set_log_level('DEBUG')


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('exp_id', nargs='+')
arg_parser.add_argument('-c', '--check_commit', default='strict')
arg_parser.add_argument('-p', '--profile', action='store_true')
arg_parser.add_argument('seed', type=int, default=0)
arg_parser.add_argument('--num-iter', default=15)
arg_parser.add_argument('--eps', default=0.0001)
arg_parser.add_argument('--momentum', default=0.5)
args = arg_parser.parse_args()

# create experiment
experiments = EditTrainingRuns(check_commit=(args.check_commit=='strict'))

exp_id = args.exp_id
if exp_id == ['default']:
    # new default experiment
    exp = experiments.new()
elif len(exp_id) == 1 and exp_id[0].isdigit():
    # reload old experiment
    exp = experiments[int(exp_id[0])]
else:
    # new experiment according to configs
    soup = args.seed
    num_iter = args.num_iter
    eps = args.eps
    momentum = args.momentum
    config = Config.from_file(exp_id[0], soup, [num_iter, eps, momentum])
    config.seed = args.seed
    config.num_iter = args.num_iter
    for filename in exp_id[1:]:
        config = Config.merge(config, Config.from_file(filename))
    exp = experiments.new(config)  # new experiment from config

# start training
exp.workspace.add_file('stdout', 'stdout.txt')
exp.workspace.add_file('stderr', 'stderr.txt')


with save_stdout(exp.workspace.root):
    exp.train()
