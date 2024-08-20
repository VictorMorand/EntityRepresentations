#Author : Victor MORAND
# 
import logging
from typing import List, Optional
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experimaestro.launchers.slurm import SlurmLauncher

# import task
from LabelExtractor import LearnLabelExtractor

logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.DEBUG) # in order to set experimaestro to debug

# Launchers, here we specify what we need for a task.

# Manual setting
# launcher = SlurmLauncher()
# gpulauncher = launcher.config(gpus=1, 
#                               mem_per_gpu= 22 * 1024, 
#                               time="60")


# Configuration of the whole experiment
@configuration
class Configuration(ConfigurationBase):
    epochs: int = 1
    launcher: str =  """duration=3h & cuda(mem=11G)*1 & cpu(cores=8)"""
    lr: float = 1e-2
    batch_size: int = 64
    logs_per_epoch: int = 2
    with_context: bool = False
    model_name: str = "gpt2-small"
    dataset_name: str = "webNLG"
    layers: dict = {'from':0, 'to':0}

def run( helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])

    logging.info(f"Launching Tasks using launcher: {gpulauncher}")

    tasks = {}
    layers = range(cfg.layers["from"],cfg.layers["to"] + 1)
    logging.info(f"will launch jobs for layers {layers} of {cfg.model_name} ")
    for layer in layers:
        task = LearnLabelExtractor(
                        model_name= tag(cfg.model_name), 
                        dataset_name= tag(cfg.dataset_name), 
                        with_context= cfg.with_context,
                        layer=tag(layer), 
                        epochs=cfg.epochs,
                        lr=cfg.lr,
                        logs_per_epoch = cfg.logs_per_epoch,
                        batch_size = cfg.batch_size,
                        )
        tasks[tagspath(task)] =  task.submit(launcher=gpulauncher).jobpath

    # Build a central "runs" directory to plot easily the metrics
    runpath = helper.xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)
    
    for key, jobath in tasks.items():
        path = (runpath / key)
        if path.exists():
            path.unlink()
        path.symlink_to(jobath)
