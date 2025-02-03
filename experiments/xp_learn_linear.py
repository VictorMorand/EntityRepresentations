#Author : Victor MORAND
# 
import logging, os
from typing import List, Optional
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, Constant
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher

# import task
from LabelExtractor import LearnLinearFilter, METRIC_VERSION
from processResults import loadResults

logging.basicConfig(level=logging.DEBUG)

# Configuration of the whole experiment
@configuration
class Configuration(ConfigurationBase):
    jobs_path: str = "/home/morand/experiments_JeanZay/jobs/labelextractor.learnlabelextractor/"
    model_names: List[str] = ["gpt2-small"]
    launchers: List[str] =  ["""duration=3h & cuda(mem=11G)*1 & cpu(cores=8)"""]
    batch_size: int = 100
    epochs: int = 10
    lr: float = 1e-2
    logs_per_epoch: int = 4
    with_context: bool = False
    dataset_name: str = "webNLG"
    extraction_method: str = "in_context"
    metrics_v: Constant[float] = METRIC_VERSION

def run( helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    if len(cfg.launchers) < len(cfg.model_names):
        raise ValueError(f"Got {len(cfg.launchers)} launchers for {len(cfg.model_names)} models to evaluate")
    
    results = loadResults(cfg.jobs_path)
    
    #extract metric 
    results["metric"] = results["Eval"].apply(lambda x: None if x is None else x["Exact Match"])

    for i, model in enumerate(cfg.model_names):

        #get launcher for current model name.
        gpulauncher = find_launcher(cfg.launchers[i], tags=["slurm"])

        filtered_results = results[
                  (results["model_name"] == model) &
                  (results["dataset_name"]==cfg.dataset_name) &
                  (results["extraction_method"]==cfg.extraction_method) &
                  (results["with_context"]==cfg.with_context) ]
        
        logging.info(f"got  {len(filtered_results)} with config: {cfg}")
        
        if len(filtered_results) == 0:
            print(f"No results for {model}, {cfg.dataset_name}, {cfg.with_context}, {cfg.method}")
            continue
        
        ################################################
        #TODO get best job instead of running on everything.. OR run on everything so that we get variance ?
        ################################################

        # idx = 0
        # layer = filtered_results.loc[idx]["layer"]
        # logging.info(f"Some layer for {cfg.model_name}: {layer}, exact match: {filtered_results.loc[idx]['metric']:.3f}, extraction method: {filtered_results.loc[idx]['extraction_method']}")
        
        logging.info(f"Launching Tasks for {model} using launcher: {gpulauncher}")

        for _ , row in filtered_results.iterrows():
            #find taskVec
            files = [file for file in os.listdir(row["path"]) if file.endswith(".pth")]
            #get the taskVec file
            if len(files) == 0:
                logging.info("No taskVec file found")
                continue
            else:
                taskVecPath = row["path"] / files[0]
            logging.info(f"launching Linear Filter Learning for {taskVecPath}...")

            task = LearnLinearFilter( 
                job_path =  str(row["path"]),
                TaskVec_path =  str(taskVecPath),
                model_name =  tag(model),
                dataset_name =  tag(cfg.dataset_name),
                layer =  tag(row["layer"]),
                batch_size = cfg.batch_size,
                epochs=cfg.epochs,
                logs_per_epoch=cfg.logs_per_epoch,
                lr= cfg.lr,
                with_context =  tag(cfg.with_context),
                extraction_method = tag(cfg.extraction_method),
            )
            task.submit(launcher=gpulauncher)


    
