from manabot.env import VectorEnv
from manabot.ppo.model import Model, Hypers
from manabot.ppo.experiment import Experiment

from argparse import ArgumentParser

def parse_args() -> tuple[Experiment, Hypers]:
    parser = ArgumentParser()
    Experiment.add_args(parser)
    Hypers.add_args(parser)
    return Experiment.from_args(parser.parse_args()), Hypers.from_args(parser.parse_args())

def main():
    experiment, hypers = parse_args()
    run_name, writer = experiment.setup(hypers)

    env = VectorEnv(hypers)
    model = Model(hypers, experiment.device)

    try:
    model.train(experiment, writer)
    finally:
        env.close()
        writer.close()

if __name__ == "__main__":
    main()