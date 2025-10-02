# Author : Victor MORAND
#
import logging, os
from pathlib import Path
from typing import List, Optional
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, Constant
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experimaestro.launchers.slurm import SlurmLauncher

# import task
from LabelExtractor import EvalLabelExtractor, METRIC_VERSION
from processResults import loadResults

logging.basicConfig(level=logging.DEBUG)


# Configuration of the whole experiment
@configuration
class Configuration(ConfigurationBase):
    jobs_path: str = "/home/"
    hashs: Optional[List[str]] = None
    model_names: List[str] = ["gpt2-small"]
    launchers: List[str] = ["""duration=3h & cuda(mem=11G)*1 & cpu(cores=8)"""]
    layers_to_evaluate: dict[str, int] = {"from": 0, "to": 1}
    batch_size: int = 32
    with_context: bool = False
    dataset_name: str = "webNLG"
    max_ent_length: int = 20
    max_len: int = 200  # max length of the input sequence, used for
    metrics_v: Constant[float] = METRIC_VERSION


def run(helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    if len(cfg.launchers) < len(cfg.model_names):
        raise ValueError(
            f"Got {len(cfg.launchers)} launchers for {len(cfg.model_names)} models to evaluate"
        )

    results = loadResults(cfg.jobs_path)

    for i, model in enumerate(cfg.model_names):

        filtered_results = results[
            (results["model_name"] == model)
            & (results["dataset_name"] == cfg.dataset_name)
            & (results["with_context"] == cfg.with_context)
        ]
        if cfg.hashs is not None:
            filtered_results = filtered_results[
                filtered_results["hash"].isin(cfg.hashs)
            ]

        logging.info(f"got  {len(filtered_results)} with config: {cfg}")

        # get launcher for current model name.
        gpulauncher = find_launcher(cfg.launchers[i], tags=["slurm"])
        logging.info(f"Launching Tasks for {model} using launcher: {gpulauncher}")

        layers = range(
            cfg.layers_to_evaluate.get("from", 0),
            cfg.layers_to_evaluate.get("to", 0) + 1,
        )

        if len(layers) == 0:
            logging.info("No layers to evaluate, skipping...")
            continue
        eval_jobs = []

        for _, row in filtered_results.iterrows():
            # find taskVec
            files = [file for file in os.listdir(row["path"]) if file.endswith(".pth")]
            # get the taskVec file
            if len(files) == 0:
                logging.info("No taskVec file found")
                continue
            else:
                taskVecPath = row["path"] / files[0]

            for layer in layers:
                logging.info(
                    f"launching evaluation of {taskVecPath}... at layer {layer} for model {model}"
                )

                task = EvalLabelExtractor.C(
                    job_path=str(row["path"]),
                    TaskVec_path=str(taskVecPath),
                    TaskVecLayer=row["layer"],
                    #data
                    dataset_name=tag(cfg.dataset_name),
                    max_length=tag(cfg.max_len),
                    max_ent_length= tag(cfg.max_ent_length),
                    # model
                    model_name=tag(model),
                    layer=tag(layer),
                    batch_size=cfg.batch_size,
                    with_context=tag(cfg.with_context),
                )
                task.submit(launcher=gpulauncher)

                eval_jobs.append({"path": task.jobpath, "layer": layer})

        helper.xp.wait()
        logging.info(f"Finished evaluation for {model}, {len(eval_jobs)} jobs done.")
        
        for eval in eval_jobs:
            path = Path(eval["path"])
            layer = eval["layer"]
            # logging.info(f"Evaluating job at path: {path.name} for layer {layer}")
            #print last two lines of the log file
            log_files = Path(path).glob("*.err") #generator 

            log_file = next(log_files, None)
            if log_file is None:
                logging.info(f"No log file found in {path}")
                continue

            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    logging.info(lines[-2].strip())
            else:
                logging.info(f"Log file {log_file} does not exist.")

