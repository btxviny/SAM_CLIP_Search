# CLIP_Search Environment Setup

An environment.yaml file is provided with all the required dependencies. To create the Conda environment, run:

```bash
   conda env create -f environment.yaml


To run the experiments with different SAM parameters run:
```bash
   python sam_parameter_search.py

To launch the MLFLOW server UI run:
```bash
   mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5001