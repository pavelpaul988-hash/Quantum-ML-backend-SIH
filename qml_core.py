"""
qml_core.py
Shared module: import this identically in the training notebook
AND in the backend service so pickled model objects can be unpickled
(pickle stores where a class was *defined* — this file needs to be the
single source of truth for both sides).
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def _apply_gate(state, gate, qubit, n_qubits):
    state = state.reshape([2] * n_qubits)
    state = np.tensordot(gate, state, axes=([1], [qubit]))
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(-1)


def _ry(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.array(
        [
            [np.exp(-1j * theta / 2.0), 0],
            [0, np.exp(1j * theta / 2.0)],
        ],
        dtype=complex,
    )


def _cnot(state, control, target, n_qubits):
    state = state.reshape([2] * n_qubits)
    state = np.moveaxis(state, [control, target], [0, 1])
    out = state.copy()
    out[1, 0] = state[1, 1]
    out[1, 1] = state[1, 0]
    out = np.moveaxis(out, [0, 1], [control, target])
    return out.reshape(-1)


class EnhancedVQC:
    """Parameterized Quantum Circuit with Entanglement & Multi-Observable Readout."""

    def __init__(self, n_qubits=4, n_layers=3, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_layers, n_qubits, 2))

    def _forward_circuit(self, x, params):
        n = self.n_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0

        for q in range(n):
            state = _apply_gate(state, _ry(x[q]), q, n)
            state = _apply_gate(state, _rz(x[q]), q, n)

        for l in range(self.n_layers):
            for q in range(n):
                state = _apply_gate(state, _ry(params[l, q, 0]), q, n)
                state = _apply_gate(state, _rz(params[l, q, 1]), q, n)
            for q in range(n):
                state = _cnot(state, q, (q + 1) % n, n)
        return state

    def extract_quantum_features(self, X, params=None):
        params = self.params if params is None else params
        features = []
        n = self.n_qubits

        for x in X:
            state = self._forward_circuit(x, params)
            probs = (np.abs(state) ** 2).reshape([2] * n)
            sample_feats = []

            for q in range(n):
                z_exp = probs.take(0, axis=q).sum() - probs.take(1, axis=q).sum()
                sample_feats.append(z_exp)

            for q1 in range(n):
                for q2 in range(q1 + 1, n):
                    adjusted_q2_axis = q2 - 1

                    probs_q1_0 = probs.take(0, axis=q1)
                    probs_q1_1 = probs.take(1, axis=q1)

                    prob_00 = probs_q1_0.take(0, axis=adjusted_q2_axis).sum()
                    prob_01 = probs_q1_0.take(1, axis=adjusted_q2_axis).sum()
                    prob_10 = probs_q1_1.take(0, axis=adjusted_q2_axis).sum()
                    prob_11 = probs_q1_1.take(1, axis=adjusted_q2_axis).sum()

                    p_even = prob_00 + prob_11
                    p_odd = prob_01 + prob_10
                    sample_feats.append(p_even - p_odd)

            features.append(sample_feats)
        return np.array(features)

    def predict_proba(self, X, params=None):
        feats = self.extract_quantum_features(X, params)
        z0 = feats[:, 0]
        prob = (1.0 - z0) / 2.0
        return np.clip(prob, 1e-6, 1.0 - 1e-6)

    def fit(self, X, y, epochs=15, lr=0.35, batch_size=32, verbose=True):
        rng = np.random.default_rng(42)
        for epoch in range(epochs):
            epoch_start_time = time.time()
            idx = rng.permutation(len(X))
            X_shuf, y_shuf = X[idx], y[idx]

            for start in range(0, len(X_shuf), batch_size):
                xb, yb = X_shuf[start : start + batch_size], y_shuf[start : start + batch_size]
                grad = np.zeros_like(self.params)
                shift = np.pi / 2.0
                for l in range(self.n_layers):
                    for q in range(self.n_qubits):
                        for k in range(2):
                            p_plus = self.params.copy()
                            p_plus[l, q, k] += shift
                            p_minus = self.params.copy()
                            p_minus[l, q, k] -= shift

                            pred_plus = self.predict_proba(xb, p_plus)
                            pred_minus = self.predict_proba(xb, p_minus)
                            loss_plus = -np.mean(
                                yb * np.log(pred_plus) + (1 - yb) * np.log(1 - pred_plus)
                            )
                            loss_minus = -np.mean(
                                yb * np.log(pred_minus) + (1 - yb) * np.log(1 - pred_minus)
                            )
                            grad[l, q, k] = (loss_plus - loss_minus) / 2.0
                self.params -= lr * grad

            p_all = self.predict_proba(X)
            loss = -np.mean(y * np.log(p_all) + (1 - y) * np.log(1 - p_all))
            acc = accuracy_score(y, (p_all >= 0.5).astype(int))
            epoch_time = time.time() - epoch_start_time
            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"[Quantum VQC] Epoch {epoch+1:02d}/{epochs} | Loss: {loss:.4f} | Train Acc: {acc*100:.2f}% | Time: {epoch_time:.2f}s"
                )


class HybridQuantumClassicalClassifier:
    """True Stacking Hybrid Quantum-Classical Architecture."""

    def __init__(self, n_qubits=4, n_layers=2):
        self.vqc = EnhancedVQC(n_qubits=n_qubits, n_layers=n_layers)
        self.classical_base = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
        self.meta_fusion = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)

    def fit(self, X_c, X_q, y):
        print("\n--- Training Phase 1: Variational Quantum Circuit ---")
        self.vqc.fit(X_q, y, epochs=5, lr=0.35, batch_size=32, verbose=True)

        print("\n--- Training Phase 2: Classical Base Model (Logistic Regression) ---")
        self.classical_base.fit(X_c, y)

        print("\n--- Training Phase 3: Stacking Meta-Fusion ---")
        c_probs_train = self.classical_base.predict_proba(X_c)[:, 1:2]
        q_probs_train = self.vqc.predict_proba(X_q).reshape(-1, 1)

        hybrid_train_features = np.hstack([c_probs_train, q_probs_train, X_c[:, :3]])
        self.meta_fusion.fit(hybrid_train_features, y)
        print("Hybrid Stacking Complete.")

    def predict_proba(self, X_c, X_q):
        c_probs = self.classical_base.predict_proba(X_c)[:, 1:2]
        q_probs = self.vqc.predict_proba(X_q).reshape(-1, 1)
        hybrid_features = np.hstack([c_probs, q_probs, X_c[:, :3]])
        return self.meta_fusion.predict_proba(hybrid_features)[:, 1]

    def predict(self, X_c, X_q, threshold=0.5):
        return (self.predict_proba(X_c, X_q) >= threshold).astype(int)
