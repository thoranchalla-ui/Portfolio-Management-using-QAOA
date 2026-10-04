# Quantum Portfolio Picker (QAOA)

Pick exactly **K of N assets** that balance expected return against risk, using a QAOA circuit
built in Qiskit, with brute force and greedy search as classical baselines.

## Run it (about 5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python portfolio_qaoa.py --selftest      # 4 checks; check 4 compares the Qiskit circuit to the fast engine
python portfolio_qaoa.py                 # CLI demo
streamlit run app.py                     # the demo UI
```

Files: `portfolio_qaoa.py` (QUBO, Ising, QAOA, baselines), `viz.py` (charts), `app.py` (Streamlit UI).

## How it works

1. **QUBO:** minimise `risk * x^T Sigma x - mu^T x`, with `x` a bitstring of N asset choices and exactly K ones.
2. **Ising + circuit:** the QUBO becomes Z and ZZ terms. Each QAOA layer is RZ/RZZ (cost) then a mixer.
3. **XY ring mixer:** starts in a state with K assets selected and conserves their number, so every sample is a valid portfolio (no penalty tuning).
4. **CVaR objective:** the classical optimiser tunes the angles to minimise the mean of the best 10% of shots. Plain expected energy was clearly worse in testing (see below).
5. **Answer:** the lowest-energy valid portfolio among the sampled shots.

Only Qiskit *core* is used (`QuantumCircuit`, `Statevector`), so it avoids `qiskit-algorithms` API churn.
The optimiser loop runs a numpy copy of the same circuit for speed; the final state is evaluated on the
real Qiskit circuit when available and cross-checked (`--selftest`, and the Circuit tab in the app).

## Measured results (numpy engine, N=8, K=4, p=3, 2048 shots)

On the 8 seeds (out of 40 tried) where the greedy baseline is **not** optimal:

| Method | Optimum found | Mean gap to optimum |
|---|---|---|
| Greedy | 0/8 | 4.0% |
| QAOA, expected-energy objective | 3/8 | 7.3% |
| QAOA, CVaR objective (default) | 8/8 | 0.0% |

QAOA output is 100% valid portfolios (random bitstrings: 27%). Probability on the true optimum averages about 20%.

## Be upfront with judges

- **No quantum advantage.** N=8 is 70 candidate portfolios; brute force is instant. The contribution is a
  verified hybrid pipeline, a constraint-preserving circuit, and honest benchmarking.
- **Instances were selected** where greedy fails, to make the comparison informative. On easy instances greedy is also exact.
- **It does not scale yet.** At N=10, K=5 (seed 2), p=3 missed the optimum (6% gap) while greedy was exact; p=4 found it
  but took ~19 s to optimise.
- Equal weights only, synthetic data by default (you can upload a prices CSV).
- Results above come from the numpy engine in my sandbox; Qiskit could not be installed there, so
  `--selftest` step 4 is the first time the Qiskit circuit is compared to it on your machine.

## 2-minute pitch outline

Problem (choose K of N under risk) - why quantum is *plausible* (combinatorial, QUBO-shaped) - live demo with the
solution map - the honest benchmark table - what's next (IBM hardware via `qiskit-ibm-runtime`, real price data, weights).
