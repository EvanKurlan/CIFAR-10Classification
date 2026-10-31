# CIFAR-10 Classification

Starter project for investigating which CIFAR-10 classes are hardest to distinguish and what contributes to classification errors. The project currently loads and audits the included CIFAR-10 Python archive and evaluates the required majority-class baseline.

## Setup

The workflow expects `cifar-10-python.tar.gz` in the repository root. If it is not present after cloning, download the official CIFAR-10 Python archive:

```powershell
Invoke-WebRequest -Uri https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -OutFile .\cifar-10-python.tar.gz
```

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How to run the project

From the repository root:

```powershell
python .\src\cifar_baseline.py
```

The script prints the data audit and baseline metrics, and writes the complete results to `results/baseline_results.json`. To use another copy of the archive:

```powershell
python .\src\cifar_baseline.py --archive C:\path\to\cifar-10-python.tar.gz --output results\baseline_results.json
```

## Project notes

- [PROJECT_STARTING_POINT.md](PROJECT_STARTING_POINT.md) summarizes the question, audit, preprocessing/split design, baseline results, and remaining work.
- [requirements.txt](requirements.txt) lists the Python dependencies.
- `src/cifar_baseline.py` contains the executable loading, audit, and baseline workflow.

Dataset source: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton, University of Toronto, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). 

## Assistance declaration

Generative AI assistance was used to create the initial project scaffold, README, and analysis code.
