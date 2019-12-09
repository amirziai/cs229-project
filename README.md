# Smoothing multi-stage fine-tuning in multi-task NLP applications
CS229 Project

## Running the code
Use Python 3.7+

This project is built on top of [jiant](https://github.com/nyu-mll/jiant). Follow the installations [here](https://github.com/nyu-mll/jiant/blob/master/tutorials/setup_tutorial.md) to setup your environment. We have copied a fork of the project into `jiant/` and reference it in the main driver file `experiments.py`.

Install dependencies on top of `jiant`:
```bash
pip3 install -r requirements.txt
```

Run the experiments:
```bash
PYTHONPATH=jiant/ python3 experiments.py
```

## Changes to `jiant`
The following files are changed or added to enable bandits for selecting batches in the main training loop:
- `jiant/jiant/trainer.py`
- `jiant/jiant/bandits.py`

## Resources
- Final report
- Poster
- Video

## References
- [jiant](https://github.com/nyu-mll/jiant)
