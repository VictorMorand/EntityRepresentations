import pathlib, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def loadResults(xp_path):
    """ load all results from a given experimaestro experiment directory
    xp_path: pathlib.Path, path to the jobs to load.
    """
    if type(xp_path) is not Path:
         xp_path = Path(xp_path)
    
    jobs = os.listdir(xp_path)
    # print(f"available jobs: {jobs}")

    results = []
    for job in tqdm(jobs):
        jobPath = xp_path / job
        job_data = {"path": jobPath,}
        with open( jobPath / "params.json") as json_file:
            params = json.load(json_file)

        params = params["objects"][0]["fields"]
        #add params to job_data
        job_data.update(params)
        if "with_context" not in job_data:
            job_data["with_context"] = False
        if "dataset_name" not in job_data:
            job_data["dataset_name"] = "WebNLG"
        if "extraction_method" not in job_data:
            job_data["extraction_method"] = "after_context"

        # params = params["params"]
        # print(job_data)
        hist_path = jobPath / "history.json"
        if not hist_path.exists():
            # print(f"missing history for {job}")
            continue
        
        with open( hist_path) as json_file:
            history = json.load(json_file)

        job_data["history"] = history

        # Find Evaluation files
        eval_file = sorted([f for f in os.listdir(jobPath) if f.startswith("Evaluation")])
        # print("found eval files:", eval_file)
        if len(eval_file) == 0:
            job_data["Eval"] = None
        else:
            with open( jobPath / eval_file[-1] ) as json_file:
                job_data["Eval"] = json.load(json_file)

        # Find Inference files
        inf_files = [f for f in os.listdir(jobPath) if f.startswith("Inference")]
        if len(inf_files) == 0:
            job_data["inference"] = None
        else:
             # Sort files by modification time, most recent first
            inf_files.sort(key=lambda f: os.path.getmtime(jobPath / f), reverse=True)
            # Select the most recent file
            job_data["inference"] = jobPath / inf_files[0]

        results.append(job_data)

    return pd.DataFrame(results)

def get_inference_res(results, 
                      model_name, 
                      layer, 
                      dataset_name=None, 
                      with_context = False,
                      extraction_method="in_context",
                      verbose=False):
    results = results[(results["model_name"] == model_name) 
                      & (results["layer"] == layer) 
                      & (results["with_context"] == with_context)
                      & (results["extraction_method"] == extraction_method)
                      ]
    if dataset_name:
        results = results[results["dataset_name"] == dataset_name]
    if verbose: print(f"found {len(results)} results for layer {layer} of {model_name} {'with' if with_context else 'without'} context {'on ' + dataset_name if dataset_name else ''}")
    #get first row dict
    if len(results) == 0:
        return None
    res = results.iloc[0].to_dict()  
    inference_file = res["path"] / res["inference"]
    if verbose: print(f"got inference files: {inference_file}")
    #get inference in results for layer
    with open(inference_file) as json_file:
        inference = json.load(json_file)
    return inference 

def get_taskVec(results, model_name, layer, dataset_name=None, with_context = False):
    results = results[(results["model_name"] == model_name) & (results["layer"] == layer) & (results["with_context"] == with_context)]
    if dataset_name:
        results = results[results["dataset_name"] == dataset_name]
    print(f"found {len(results)} results for layer {layer} of {model_name} {'with' if with_context else 'without'} context {'on ' + dataset_name if dataset_name else ''}")
    #get first row dict
    if len(results) == 0:
        return None
    res = results.iloc[0].to_dict()  
    files = [file for file in os.listdir(res["path"]) if file.endswith(".pth")]
    #get the taskVec file
    print(f"found {files} ")
    if len(files) == 0:
        print("No taskVec file found")
        return None
    else:
        return res["path"] / files[0]