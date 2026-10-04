from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
def make_problem(n: int = 8, seed: int = 7):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.20, n)
    a = rng.normal(size=(n, n)) * 0.1
    sigma = a @ a.T + np.diag(rng.uniform(0.01, 0.05, n))
    return mu, sigma
def problem_from_prices(prices, periods_per_year: int = 252):
    prices = np.asarray(prices, dtype=float)
    rets = prices[1:] / prices[:-1] - 1.0
    return rets.mean(axis=0) * periods_per_year, np.cov(rets, rowvar=False) * periods_per_year
def all_bits(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    return (idx[:, None] >> np.arange(n)) & 1
@dataclass
class Qubo:
    Q: np.ndarray
    const: float
    k: int
    mu: np.ndarray
    sigma: np.ndarray
    risk: float
    penalty: float
    _energies: np.ndarray | None = field(default=None, repr=False)
    @property
    def n(self) -> int:
        return len(self.mu)
    def energy_all(self) -> np.ndarray:
        if self._energies is None:
            b = all_bits(self.n)
            self._energies = np.einsum("bi,ij,bj->b", b, self.Q, b) + self.const
        return self._energies
    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.const)
    def feasible_mask(self) -> np.ndarray:
        return all_bits(self.n).sum(axis=1) == self.k
    def normalized(self, feasible_only: bool = False, span: float = 1.0) -> "Qubo":
        e = self.energy_all()
        ref = e[self.feasible_mask()] if feasible_only else e
        scale = span / float(ref.max() - ref.min())
        return Qubo(
            Q=self.Q * scale,
            const=(self.const - float(ref.min())) * scale,
            k=self.k, mu=self.mu, sigma=self.sigma, risk=self.risk,
            penalty=self.penalty,
        )
def build_qubo(mu, sigma, k: int, risk: float = 0.5, penalty: float | None = None) -> Qubo:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    if penalty is None:
        penalty = 1.5 * (mu.max() + risk * np.abs(sigma).sum() / n)
    q = risk * sigma + penalty * np.ones((n, n))
    q[np.diag_indices(n)] += -mu - 2.0 * penalty * k
    return Qubo(Q=q, const=penalty * k * k, k=k, mu=mu, sigma=sigma, risk=risk, penalty=penalty)
def to_ising(q: Qubo):
    Q = (q.Q + q.Q.T) / 2.0
    n = len(Q)
    h = -0.5 * Q.sum(axis=1)
    J = np.triu(Q, 1) * 0.5
    offset = q.const + 0.25 * Q.sum() + 0.25 * np.trace(Q)
    return h, J, float(offset)
def ising_energy_all(h, J, offset, n) -> np.ndarray:
    z = 1 - 2 * all_bits(n)
    return offset + z @ h + np.einsum("bi,ij,bj->b", z, J, z)
def brute_force(q: Qubo):
    e = np.where(q.feasible_mask(), q.energy_all(), np.inf)
    b = int(np.argmin(e))
    return b, float(e[b])
def greedy(q: Qubo):
    x = np.zeros(q.n, dtype=int)
    for _ in range(q.k):
        best, best_e = None, np.inf
        for i in range(q.n):
            if x[i]:
                continue
            x[i] = 1
            e = q.risk * x @ q.sigma @ x - q.mu @ x
            x[i] = 0
            if e < best_e:
                best, best_e = i, e
        x[best] = 1
    return bits_to_int(x), q.energy(x)
def bits_to_int(x) -> int:
    return int(sum(int(v) << i for i, v in enumerate(x)))
def int_to_bits(b: int, n: int) -> np.ndarray:
    return np.array([(b >> i) & 1 for i in range(n)])
def portfolio_stats(x, mu, sigma) -> dict:
    x = np.asarray(x)
    w = x / max(x.sum(), 1)
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ sigma @ w))
    return {"return": ret, "risk": vol, "sharpe_like": ret / vol if vol > 0 else float("nan")}
@dataclass
class QAOAResult:
    params: np.ndarray
    expectation: float
    probs: np.ndarray
    history: list
    n_evals: int
class QAOA:
    def __init__(self, qubo: Qubo, p: int = 2, backend: str = "qiskit", mixer: str = "xy",
                 span: float = 2 * np.pi, objective: str = "cvar", alpha: float = 0.1):
        if backend not in ("qiskit", "numpy"):
            raise ValueError("backend must be 'qiskit' or 'numpy'")
        if mixer not in ("x", "xy"):
            raise ValueError("mixer must be 'x' or 'xy'")
        if objective not in ("expectation", "cvar"):
            raise ValueError("objective must be 'expectation' or 'cvar'")
        self.mixer = mixer
        self.objective = objective
        self.alpha = alpha
        self._source = qubo
        self.k = qubo.k
        if mixer == "xy":
            base = build_qubo(qubo.mu, qubo.sigma, qubo.k, qubo.risk, penalty=0.0)
            self.qubo = base.normalized(feasible_only=True, span=span)
        else:
            self.qubo = qubo.normalized(feasible_only=True, span=span)
        self.span = span
        self.p = p
        self.n = qubo.n
        self.backend = backend
        self.energies = self.qubo.energy_all()
        self._order = np.argsort(self.energies)
        self.h, self.J, self.offset = to_ising(self.qubo)
        self.pairs = [(i, (i + 1) % self.n) for i in range(self.n)] if self.n > 2 else [(0, 1)]
        self.circuit = None
        if backend == "qiskit":
            self._build_circuit()
    def _build_circuit(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        n, p = self.n, self.p
        self.gam = ParameterVector("gamma", p)
        self.bet = ParameterVector("beta", p)
        qc = QuantumCircuit(n)
        if self.mixer == "xy":
            for i in range(self.k):
                qc.x(i)
        else:
            qc.h(range(n))
        for layer in range(p):
            g, b = self.gam[layer], self.bet[layer]
            for i in range(n):
                qc.rz(2 * float(self.h[i]) * g, i)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(self.J[i, j]) > 1e-12:
                        qc.rzz(2 * float(self.J[i, j]) * g, i, j)
            qc.barrier()
            if self.mixer == "xy":
                for i, j in self.pairs:
                    qc.rxx(2 * b, i, j)
                    qc.ryy(2 * b, i, j)
            else:
                for i in range(n):
                    qc.rx(2 * b, i)
        self.circuit = qc
    def probabilities(self, params) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        gammas, betas = params[: self.p], params[self.p:]
        if self.backend == "qiskit":
            from qiskit.quantum_info import Statevector
            binding = {self.gam[i]: float(gammas[i]) for i in range(self.p)}
            binding.update({self.bet[i]: float(betas[i]) for i in range(self.p)})
            bound = self.circuit.assign_parameters(binding)
            return np.asarray(Statevector(bound).probabilities(), dtype=float)
        return self._numpy_probs(gammas, betas)
    def _numpy_probs(self, gammas, betas) -> np.ndarray:
        n = self.n
        N = 2**n
        idx = np.arange(N)
        if self.mixer == "xy":
            psi = np.zeros(N, dtype=complex)
            psi[(1 << self.k) - 1] = 1.0
        else:
            psi = np.full(N, 1 / np.sqrt(N), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.energies)
            if self.mixer == "xy":
                c, s = np.cos(2 * b), -1j * np.sin(2 * b)
                for i, j in self.pairs:
                    differ = ((idx >> i) & 1) != ((idx >> j) & 1)
                    partner = idx ^ ((1 << i) | (1 << j))
                    new = psi.copy()
                    new[differ] = c * psi[differ] + s * psi[partner[differ]]
                    psi = new
            else:
                c, s = np.cos(b), -1j * np.sin(b)
                psi = psi.reshape((2,) * n)
                for ax in range(n):
                    a0 = np.take(psi, 0, axis=ax)
                    a1 = np.take(psi, 1, axis=ax)
                    psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
                psi = psi.reshape(-1)
        return np.abs(psi) ** 2
    def expectation(self, params) -> float:
        return float(self.probabilities(params) @ self.energies)
    def cvar(self, probs) -> float:
        ps = probs[self._order]
        before = np.cumsum(ps) - ps
        w = np.clip(self.alpha - before, 0.0, ps)
        return float(w @ self.energies[self._order] / self.alpha)
    def _np_expectation(self, params) -> float:
        d = len(params) // 2
        pr = self._numpy_probs(np.asarray(params[:d]), np.asarray(params[d:]))
        return self.cvar(pr) if self.objective == "cvar" else float(pr @ self.energies)
    @staticmethod
    def _interp(x: np.ndarray) -> np.ndarray:
        d = len(x)
        xe = np.concatenate([[0.0], x, [0.0]])
        return np.array([(i - 1) / d * xe[i - 1] + (d - i + 1) / d * xe[i] for i in range(1, d + 2)])
    def run(self, restarts: int = 3, maxiter: int = 200, seed: int = 0) -> QAOAResult:
        rng = np.random.default_rng(seed)
        n_evals = 0
        def local(x0, hist=None):
            nonlocal n_evals
            def f(x):
                nonlocal n_evals
                n_evals += 1
                v = self._np_expectation(x)
                if hist is not None:
                    hist.append(v)
                return v
            return minimize(f, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        k = 2 * np.pi / self.span
        grid = [(g * k, b) for g in np.linspace(0.05, 1.0, 12) for b in np.linspace(0.05, 1.5, 10)]
        scored = sorted(grid, key=lambda gb: self._np_expectation(np.array(gb)))
        n_evals += len(grid)
        best_x, best_f, best_hist = None, np.inf, []
        for g0, b0 in scored[:restarts]:
            res = local(np.array([g0, b0]))
            x, hist = res.x, [res.fun]
            for depth in range(2, self.p + 1):
                gam, bet = x[: depth - 1], x[depth - 1:]
                interp = np.concatenate([self._interp(gam), self._interp(bet)])
                padded = np.concatenate([gam, [0.0], bet, [0.0]])
                starts = [interp, padded] + [interp + rng.normal(0, 0.15, 2 * depth) for _ in range(3)]
                runs = []
                for st in starts:
                    h: list = []
                    runs.append((local(st, h), h))
                res, hist = min(runs, key=lambda rh: rh[0].fun)
                x = res.x
            if res.fun < best_f:
                best_x, best_f, best_hist = x, float(res.fun), hist
        probs = self.probabilities(best_x)
        return QAOAResult(best_x, float(probs @ self.energies), probs, best_hist, n_evals)
    def verify_with_qiskit(self, params) -> float:
        twin = QAOA(self._source, p=self.p, backend="qiskit", mixer=self.mixer, span=self.span,
                    objective=self.objective, alpha=self.alpha)
        return float(np.max(np.abs(twin.probabilities(params) - self._numpy_probs(
            np.asarray(params[: self.p]), np.asarray(params[self.p:])))))
    def sample(self, probs, shots: int = 2048, seed: int = 1) -> dict:
        rng = np.random.default_rng(seed)
        probs = probs / probs.sum()
        draws = rng.choice(len(probs), size=shots, p=probs)
        vals, counts = np.unique(draws, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}
def bitstring(b: int, n: int) -> str:
    return "".join(str((b >> i) & 1) for i in range(n))
def solve(n=8, k=4, risk=0.5, p=3, restarts=3, shots=2048, seed=2,
          backend="qiskit", mixer="xy", objective="cvar", alpha=0.1, data=None) -> dict:
    mu, sigma = data if data is not None else make_problem(n, seed)
    n = len(mu)
    q = build_qubo(mu, sigma, k, risk)
    opt_b, opt_e = brute_force(q)
    greedy_b, greedy_e = greedy(q)
    qaoa = QAOA(q, p=p, backend=backend, mixer=mixer, objective=objective, alpha=alpha)
    res = qaoa.run(restarts=restarts, seed=seed)
    counts = qaoa.sample(res.probs, shots=shots, seed=seed)
    feas = q.feasible_mask()
    measured_feasible = [b for b in counts if feas[b]]
    qaoa_b = min(measured_feasible, key=lambda b: q.energy_all()[b]) if measured_feasible else None
    order = np.where(feas)[0][np.argsort(q.energy_all()[feas])]
    n_feas = int(feas.sum())
    uniform_over = n_feas if mixer == "xy" else 2**n
    qaoa_e = q.energy_all()[qaoa_b] if qaoa_b is not None else None
    return {
        "p_top5": float(res.probs[order[:5]].sum()),
        "p_random_top5": min(5, n_feas) / n_feas,
        "gap_qaoa": (qaoa_e - opt_e) / abs(opt_e) if qaoa_e is not None else None,
        "gap_greedy": (greedy_e - opt_e) / abs(opt_e),
        "mu": mu, "sigma": sigma, "qubo": q, "qaoa": qaoa, "result": res, "counts": counts,
        "optimal": opt_b, "optimal_energy": opt_e,
        "greedy": greedy_b, "greedy_energy": greedy_e,
        "qaoa_best": qaoa_b,
        "qaoa_best_energy": q.energy_all()[qaoa_b] if qaoa_b is not None else None,
        "p_optimal": float(res.probs[opt_b]),
        "p_feasible": float(res.probs[feas].sum()),
        "p_random_optimal": 1.0 / uniform_over,
        "p_random_feasible": float(feas.mean()),
    }
def selftest():
    print("1) QUBO <-> Ising energies agree ...", end=" ")
    mu, sigma = make_problem(6, 3)
    q = build_qubo(mu, sigma, 3)
    h, J, off = to_ising(q)
    assert np.allclose(ising_energy_all(h, J, off, 6), q.energy_all())
    print("ok")
    print("2) Brute-force optimum is feasible ...", end=" ")
    b, _ = brute_force(q)
    assert int_to_bits(b, 6).sum() == 3
    print("ok")
    print("3) numpy simulator matches a dense-matrix reference (both mixers) ...", end=" ")
    x = np.array([0.7, 1.3, 0.4, 0.9])
    for mixer in ("x", "xy"):
        a = QAOA(q, p=2, backend="numpy", mixer=mixer)
        assert np.allclose(a.probabilities(x), _reference_probs(a, x), atol=1e-9), mixer
    print("ok")
    print("4) qiskit circuit matches numpy simulator (both mixers) ...", end=" ")
    try:
        import qiskit
    except ImportError:
        print("skipped (qiskit not installed)")
    else:
        for mixer in ("x", "xy"):
            a = QAOA(q, p=2, backend="numpy", mixer=mixer)
            c = QAOA(q, p=2, backend="qiskit", mixer=mixer)
            assert np.allclose(a.probabilities(x), c.probabilities(x), atol=1e-8), mixer
        print("ok")
def _reference_probs(qaoa: "QAOA", params) -> np.ndarray:
    from scipy.linalg import expm
    n, p = qaoa.n, qaoa.p
    gammas, betas = np.asarray(params)[:p], np.asarray(params)[p:]
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]])
    def op(single, i):
        m = np.array([[1.0]])
        for q in reversed(range(n)):
            m = np.kron(m, single if q == i else I2)
        return m
    N = 2**n
    if qaoa.mixer == "xy":
        psi = np.zeros(N, dtype=complex)
        psi[(1 << qaoa.k) - 1] = 1
    else:
        psi = np.full(N, 1 / np.sqrt(N), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = expm(-1j * g * np.diag(qaoa.energies)) @ psi
        if qaoa.mixer == "xy":
            for i, j in qaoa.pairs:
                psi = expm(-1j * b * (op(X, i) @ op(X, j) + op(Y, i) @ op(Y, j))) @ psi
        else:
            for i in range(n):
                psi = expm(-1j * b * op(X, i)) @ psi
    return np.abs(psi) ** 2
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--backend", default="qiskit", choices=["qiskit", "numpy"])
    ap.add_argument("--mixer", default="xy", choices=["x", "xy"])
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        r = solve(n=a.n, k=a.k, p=a.p, backend=a.backend, mixer=a.mixer, seed=a.seed)
        n = a.n
        print(f"Brute force optimum : {bitstring(r['optimal'], n)}  energy={r['optimal_energy']:.4f}")
        print(f"Greedy baseline     : {bitstring(r['greedy'], n)}  energy={r['greedy_energy']:.4f}")
        print(f"QAOA best measured  : {bitstring(r['qaoa_best'], n)}  energy={r['qaoa_best_energy']:.4f}")
        print(f"P(optimal) QAOA={r['p_optimal']:.3f} vs random={r['p_random_optimal']:.4f}")
        print(f"P(feasible) QAOA={r['p_feasible']:.3f} vs random={r['p_random_feasible']:.3f}")
