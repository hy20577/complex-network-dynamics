# MATLAB reference data

The tests in `tests/test_matlab_reference.py` check the Python ports against the
original MATLAB code from the PhD thesis
([hy20577/PhD_thesis_all_codes](https://github.com/hy20577/PhD_thesis_all_codes),
commit `d0e88a9`).

## What is here

| Path | Contents |
|------|----------|
| `matlab/original/` | Unmodified thesis functions: `MIR`, `Grid`, `location`, `corr_decay`, `Log_generator`, `RK4`, `Euler`, `lorenz_ext` |
| `matlab/shims/` | Small stand-ins for MATLAB-only functions Octave lacks: `graph`, `distances` (breadth-first search) and `waitbar` (does nothing) |
| `matlab/generate_reference.m` | Runs the original functions on the inputs and writes the outputs |
| `make_inputs.py` | Writes the fixed input data (`data/in_*.txt`) |
| `data/` | Inputs (`in_*.txt`) and MATLAB outputs (`ref_*.txt`), comma-separated, 17 significant digits |

## Regenerating

Only needed when inputs change or more MATLAB functions are added. Requires
[Octave](https://octave.org) (`brew install octave`), or MATLAB.

```bash
python tests/reference/make_inputs.py
octave --no-gui tests/reference/matlab/generate_reference.m
```

## Comparing chaotic systems

Chaotic trajectories amplify rounding differences, so the tests compare only
the first steps of a trajectory exactly, and compare Lyapunov exponents with
a tolerance.
