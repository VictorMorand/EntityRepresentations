#Author : Victor MORAND
# 
import logging
from typing import List, Optional
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launchers.slurm import SlurmLauncher

# import task
from LabelExtractor import LearnLabelExtractor

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

# Launchers, here we specify what we need for a task.

launcher = SlurmLauncher()
gpulauncher = launcher.config(gpus=1, 
                              mem_per_gpu= 22 * 1024, 
                              time="60")

# Configuration of the whole experiment
@configuration
class Configuration(ConfigurationBase):
    epochs: int = 1
    model_name: str = "gpt2-small"
    layers: List[int] = [2]

def run( helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    logging.info(f"Launching Tasks using launcher: {gpulauncher}")

    for layer in cfg.layers:
        LearnLabelExtractor(
            model_name=tag(cfg.model_name), 
            layer=tag(layer), 
            epochs=cfg.epochs
            ).submit(launcher=gpulauncher)
    
    