import functools

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


# Registers all the ViZDoom environments
def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)

# Sample Factory allows the registration of a custom Neural Network architecture
# See https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/doom_model.py for more details
def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()

# parse the command line args and create a config
def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

## Start the training, this should take around 15 minutes
register_vizdoom_components()

# The scenario we train on today is health gathering
# other scenarios include:
# "doom_basic", "doom_two_colors_easy", "doom_dm", "doom_dwango5", "doom_my_way_home"
# "doom_deadly_corridor", "doom_defend_the_center", "doom_defend_the_line"
env = "doom_health_gathering_supreme"
cfg = parse_vizdoom_cfg(argv=[f"--env={env}",
                              "--num_workers=8",
                              "--num_envs_per_worker=4",
                              "--train_for_env_steps=50000000"
                              ])

status = run_rl(cfg)

from sample_factory.enjoy import enjoy

cfg = parse_vizdoom_cfg(argv=[f"--env={env}",
                              "--num_workers=1",
                              "--save_video",
                              "--no_render",
                              "--max_num_episodes=10"],
                        evaluation=True)
status = enjoy(cfg)

# get HF token
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--huggingface-token", type=str, help="Hugging Face token to push to repo.")
args = parser.parse_args()


from huggingface_hub import login
login(args.huggingface_token)

hf_username = "toinsson" # insert your HuggingFace username here
cfg = parse_vizdoom_cfg(argv=[f"--env={env}", "--num_workers=1", "--save_video", "--no_render", "--max_num_episodes=10", "--max_num_frames=100000", "--push_to_hub", f"--hf_repository={hf_username}/rl_course_vizdoom_health_gathering_supreme"], evaluation=True)
status = enjoy(cfg)