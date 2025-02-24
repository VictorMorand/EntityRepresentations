# On the Representations of Entities in Auto-regressive Large Language Models
*Anonymzed Repository for the Double Blind Peer reviewing process.*


This is the repository for the Entity representation project. 
Our code is based on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens/tree/main), an interpretability library that allows to harmonize various LLM hooks and loadings.

## Installation

### Environment
this project is built on Pytorch, which can be installed from [here](https://pytorch.org/get-started/locally/)
then the environment can be set up with pip 

Optionnally create a special env for
```
python -m venv env 
source env/bin/activate
pip install --upgrade pip
```
and then install the requirements

```
pip install -r requirements.txt
```

## Demo Usage
We provide several Notebooks for interactive manipulation of our code, among which:
- `Demo.ipynb` presents a walkthrough as well as a demo of the Entity Lens Method.
- `EntityRepresentations.ipynb` contains the raw code for many experimentatations.

