# chem_oracle
This repository contains the implementation of the Chemical Oracle described in our paper _Digitizing chemical discovery with a Bayesian explorer for interpreting reactivity data_ ([preprint]).

## Requirements
* Python 3.10 or higher
* Poetry ([installation instructions][poetry_installation])

## Installation
1. Edit the `pyproject.toml` file to choose whether to use the `cpu` or `gpu` version of `jax`.
```toml
[tool.poetry.dependencies]
jaxlib = {source = "jax-gpu"} # or "jax-cpu"
```

2. Install the dependencies using Poetry.
```bash
poetry install # to install dev dependencies, poetry install --with dev
# to enter the virtual environment with all requirements installed
poetry shell
```

## Usage
Interfacing with analytical data is handled by the `ExperimentManager` class in the `chem_oracle.experiment` namespace. Whilst this class can be used directly by feeding analytical data manually, the Oracle can also be launched in monitoring mode using the code in `chem_oracle.main` namespace, which automatically processes new analytical data as it is added to the `ExperimentManager`'s data directory.

```python
from chem_oracle.main import main
from chem_oracle.experiment import ExperimentManager

# The containing directory of xlsx_file is used as data_dir by default
manager = ExperimentManager(xlsx_file = "experiment_outcomes.xlsx")
main(manager)
```

## Bayesian model
The Bayesian model can be found in the `chem_oracle.model` namespace. Variants of this model using a different set of assumptions and prior distributions can be examined by checking out the other branches of this repository.

## Delphi model repository system
Instead of creating separate git branches for each model variant, the model itself can be deposited and versioned using the [_Delphi_][delphi] system.

[poetry_installation]: https://python-poetry.org/docs/#installation
[preprint]: https://chemrxiv.org/engage/chemrxiv/article-details/63607930ecdad5caaff4f734
[delphi]: https://github.com/hessammehr/delphi