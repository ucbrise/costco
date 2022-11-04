# CostCO
WARNING: This is an academic proof-of-concept prototype and has not received
careful code review. This implementation is NOT ready for production use.

## Setup
Set up a python virtual environment (recommended) and install the dependencies
in `requirements.txt`.

## Cost Modeling a Protocol
### Screening Step
[TODO] pbd.py

### Getting the Cost Model

TL;DR: run appropriate circuit in circuits file; and feed the results to
cost_modeler.py.

### Specific Example (ABY)
Cherry-pick [this
commit](https://github.com/vivi/ABY/commit/659fb849aff4f72edb84b59caa7d91b7cab979ec) into your ABY directory and
rebuild ABY.

[TODO] Fix cost_modeler.py to differentiate between ABY/AgMPC

## Generating Circuits
For convenience, you can use the predefined circuits in `circuits`.

However, if you would like to generate your own circuits, use `circuit.py`:

```
python3 circuit.py [-w=width] [-g=max_gates] [spec_file] [experiment_type]"
```

Experiment type can be either `pbd` or `ccd`.

For CCD, if width (circuit depth) is not a feature that was deemed significant from the PBD
experiment, fix call `circuit.py` with `w=0`.

To exclude features from the CCD experiment, delete the relevant gates in the
spec file.

## Running the Compiler
```
python3 run_compiler.py [input_file] [output_file]
```
