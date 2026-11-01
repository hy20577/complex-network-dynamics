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

## Behaviour matched to MATLAB

The Python code follows the MATLAB code in these details:

- `location`: cells are half-open from below, like `ceil()`, so a point exactly on a grid line belongs to the lower cell.
- `MIR`: a node's self-information is divided by the decay time of the pair (node, node).
- `Grid`: halves round up, like MATLAB's `round()`.
- Coupled maps: a node with no links evolves as (1 − α) f(xᵢ).

## Known issue in the MATLAB code

The Gram-Schmidt routine inside `Log_generator.m`, `Circle_generator.m` and
`Composite_log_generator.m` uses `if a ~= 0` on a vector. MATLAB treats that as
true only when *every* entry is nonzero, so the projection is skipped whenever
a tangent vector has a zero entry. For connected networks this makes no
difference (≤ 5e-15 on the 12-node network over 3,000 steps). For networks
with an isolated node or separate components, the vectors stay partly zero and
the Lyapunov exponents come out wrong: on the 3-node test network MATLAB gives
(0.613, 0.516, −0.234), while the correct values, which Python gives and which
MATLAB also gives with the `if` removed, are (0.613, −0.234, −0.361). The tests
therefore compare the orbits of that case with MATLAB, but check its exponents
against the determinant rule. `RK4.m` and `Euler.m` do not have this check.

## Comparing chaotic systems

Chaotic trajectories amplify rounding differences, so the tests compare only
the first steps of a trajectory exactly, and compare Lyapunov exponents with
a tolerance.
