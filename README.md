[TODO] License blurb

## Setup
Set up a python virtual environment (recommended) and install the dependencies
in `requirements.txt`.

## Cost Modeling a Protocol
(under construction)

### ABY
Cherry-pick this commit [TODO: link to patch] into your ABY directory and
rebuild ABY.

[TODO] Fix cost_modeler.py to differentiate between ABY/AgMPC

TL;DR: run appropriate circuit in circuits file; and feed the results to foba.

## Generating Circuits
For convenience, you can use the predefined circuits in `circuits`.

However, if you would like to generate your own circuits, use `circuit.py`:

```
python3 circuit.py [-w=width] [spec_file] [experiment_type]"
```

[TODO] Add argparse to circuit.py

Experiment type can be either `pbd` or `ccd`.

For CCD, if width is not a feature that was deemed significant from the PBD
experiment, fix call `circuit.py` with `w=0`.

To exclude features from the CCD experiment, delete the relevant gates in the
spec file.

## Running the Compiler
```
python3 run_compiler.py [input_file] [output_file]
```
