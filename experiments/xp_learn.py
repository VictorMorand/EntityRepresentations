#Author : Victor MORAND
# 
import os, logging
from typing import List, Optional
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experimaestro.launchers.slurm import SlurmLauncher

# import task
from LabelExtractor import LearnLabelExtractor, LEARNER_VERSION

logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.DEBUG) # in order to set experimaestro to debug


# Configuration of the whole experiment
@configuration
class Configuration(ConfigurationBase):
    epochs: int = 1
    launcher: str =  """duration=3h & cuda(mem=11G)*1 & cpu(cores=8)"""
    lr: float = 1e-2
    batch_size: int = 64
    n_runs: int = 1
    logs_per_epoch: int = 2
    with_context: bool = False
    extraction_method: str = "after_context"
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
        for run in range(cfg.n_runs):
            task = LearnLabelExtractor(
                        model_name= tag(cfg.model_name), 
                        dataset_name= tag(cfg.dataset_name), 
                        extraction_method = tag(cfg.extraction_method),
                        with_context= cfg.with_context,
                        layer=tag(layer), 
                        epochs=cfg.epochs,
                        lr=cfg.lr,
                        logs_per_epoch = cfg.logs_per_epoch,
                        batch_size = cfg.batch_size,
                        run = run,
                        )
            tasks[tagspath(task)] =  task.submit(launcher=gpulauncher).jobpath

    # Build a central "runs" directory to plot easily the metrics
    runpath = helper.xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)
    
    for key, jobath in tasks.items():
        path = (runpath / key)
        if path.exists():
            # remove the old symlink
            os.remove(path)
        path.symlink_to(jobath)
