# WQN algorithm supporting code (ACHA)

Supporting code for "The WQN algorithm to adaptively correct artifacts in the EEG signal" by Matteo Dora, Stéphane Jaffard, David Holcman.

## Setup

The code was written for Python 3.10 (but can probably run on previous versions without modification). The dependencies can be installed via [poetry](https://python-poetry.org/). After installing `poetry`, the project can be set up with:

```sh
poetry install
```

Otherwise, one can manually install the dependencies listed in `pyproject.toml` file with their preferred tool, e.g:

```sh
pip install scipy numpy matplotlib PyWavelets ipykernel fbm h5py pandas tqdm
```

## Figures and results

The folder `acha_scripts` contains Python scripts to generate all figures and results. They can be run like this:

```sh
PYTHONPATH=. poetry run python acha_scripts/[name_of_the_script].py 
```

The generated output files will be placed in the `./output` folder.

## Help

If you need help running the code don't hesitate to contact the author at matteo.dora@ieee.org.
