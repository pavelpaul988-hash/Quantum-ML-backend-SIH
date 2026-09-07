
import os
import joblib
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "breast_cancer_models.pkl"

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
]


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("       BreastCancerQML - Hybrid Model Training")
print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Wisconsin Breast Cancer dataset...")

data = load_breast_cancer()

X = data.data[:, :10]
y = data.target

# sklearn labels:
# 0 = malignant
# 1 = benign
#
# We keep these labels because the frontend/backend already
# expects malignant/benign classification.

print("\nDataset loaded successfully.")
print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])

print("\nFeatures used:")
for feature in FEATURE_NAMES:
    print(" -", feature)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("                    DATA SPLIT")
print("=" * 60)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# CLASSICAL MACHINE LEARNING
# ============================================================

print("\n" + "=" * 60)
print("              CLASSICAL MACHINE LEARNING")
print("=" * 60)

print("\nScaling classical features...")

classical_scaler = StandardScaler()

X_train_classical = classical_scaler.fit_transform(X_train)
X_test_classical = classical_scaler.transform(X_test)

print("Creating Random Forest model...")

classical_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

print("Training Random Forest...")

classical_model.fit(
    X_train_classical,
    y_train
)

print("Random Forest training completed.")

print("\nEvaluating Classical ML model...")

classical_predictions = classical_model.predict(X_test_classical)

classical_accuracy = accuracy_score(
    y_test,
    classical_predictions
)

print("\nClassical ML Accuracy:")
print(f"{classical_accuracy * 100:.2f}%")

print("\nClassical Classification Report:")
print(
    classification_report(
        y_test,
        classical_predictions,
        target_names=["Malignant", "Benign"]
    )
)

print("Classical Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        classical_predictions
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = {}

for name, value in zip(
    FEATURE_NAMES,
    classical_model.feature_importances_
):
    feature_importance[name] = float(value)

print("\nFeature importance calculated.")


# ============================================================
# QUANTUM PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("                QUANTUM PREPROCESSING")
print("=" * 60)

print("\nScaling quantum input features...")

quantum_scaler = StandardScaler()

X_train_scaled = quantum_scaler.fit_transform(X_train)
X_test_scaled = quantum_scaler.transform(X_test)

print("\nApplying PCA...")

pca = PCA(
    n_components=4,
    random_state=42
)

X_train_quantum = pca.fit_transform(X_train_scaled)
X_test_quantum = pca.transform(X_test_scaled)

explained_variance = np.sum(
    pca.explained_variance_ratio_
)

print("\nOriginal features:", X.shape[1])
print("Quantum features:", X_train_quantum.shape[1])
print(
    "Explained variance:",
    f"{explained_variance * 100:.2f}%"
)


# ============================================================
# QUANTUM MACHINE LEARNING
# ============================================================

print("\n" + "=" * 60)
print("              QUANTUM MACHINE LEARNING")
print("=" * 60)

print("\nCreating 4-qubit ZZ feature map...")

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1
)

print("Creating Real Amplitudes ansatz...")

ansatz = real_amplitudes(
    num_qubits=4,
    reps=1
)

print("Creating StatevectorSampler...")

sampler = StatevectorSampler(
    seed=42
)

print("Creating Qiskit VQC...")

optimizer = COBYLA(
    maxiter=25
)

quantum_model = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

print("\nQuantum circuit created successfully.")
print("Qubits: 4")
print("Feature map: ZZFeatureMap")
print("Ansatz: RealAmplitudes")
print("Optimizer: COBYLA")


# ============================================================
# TRAIN VQC
# ============================================================

print("\n" + "=" * 60)
print("                TRAINING QUANTUM VQC")
print("=" * 60)

print("\nStarting VQC training...")
print("Please wait...\n")

quantum_model.fit(
    X_train_quantum,
    y_train
)

print("\nQuantum VQC training completed successfully.")


# ============================================================
# QUANTUM EVALUATION
# ============================================================

print("\nEvaluating Quantum ML model...")

quantum_predictions = quantum_model.predict(
    X_test_quantum
)

quantum_predictions = np.asarray(
    quantum_predictions
).reshape(-1)

quantum_predictions = quantum_predictions.astype(int)

quantum_accuracy = accuracy_score(
    y_test,
    quantum_predictions
)

print("\nQuantum ML Accuracy:")
print(f"{quantum_accuracy * 100:.2f}%")

print("\nQuantum Classification Report:")
print(
    classification_report(
        y_test,
        quantum_predictions,
        target_names=["Malignant", "Benign"]
    )
)

print("Quantum Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        quantum_predictions
    )
)


# ============================================================
# SAVE QUANTUM PARAMETERS
# ============================================================
#
# IMPORTANT:
# We DO NOT save the complete VQC object with joblib.
#
# Qiskit VQC contains an internal function that cannot be
# serialized by Python pickle/joblib.
#
# Instead, save:
#   - trained VQC weights
#   - feature map configuration
#   - ansatz configuration
#
# The Flask backend can reconstruct the VQC later.
# ============================================================

print("\n" + "=" * 60)
print("              PREPARING QUANTUM MODEL")
print("=" * 60)

try:
    quantum_weights = np.asarray(
        quantum_model.weights
    ).tolist()

    print("\nQuantum weights extracted successfully.")
    print("Number of parameters:", len(quantum_weights))

except Exception as error:
    print("\nERROR extracting quantum weights:")
    print(error)
    raise


# ============================================================
# MODEL PACKAGE
# ============================================================

model_package = {

    "classical_model": classical_model,

    "classical_scaler": classical_scaler,

    "quantum_scaler": quantum_scaler,

    "pca": pca,

    "quantum_weights": quantum_weights,

    "feature_importance": feature_importance,

    "classical_accuracy": float(
        classical_accuracy
    ),

    "quantum_accuracy": float(
        quantum_accuracy
    ),

    "feature_names": FEATURE_NAMES,

    "quantum_features": 4,

    "qubits": 4,

    "feature_map": "ZZFeatureMap",

    "ansatz": "RealAmplitudes",

    "optimizer": "COBYLA",

    "quantum_framework": "Qiskit VQC",

    "sampler": "StatevectorSampler",

    "hybrid_method": (
        "70% Classical ML + 30% Quantum ML"
    ),

    "dataset": (
        "Wisconsin Diagnostic Breast Cancer"
    ),

    "version": "1.0",

}


# ============================================================
# SAVE PACKAGE
# ============================================================

print("\n" + "=" * 60)
print("                  SAVING MODELS")
print("=" * 60)

print("\nSaving hybrid model package...")

joblib.dump(
    model_package,
    MODEL_FILE
)

print("\nModel package saved successfully!")

print("\nFile:")
print(
    os.path.abspath(MODEL_FILE)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("                 TRAINING COMPLETE")
print("=" * 60)

print("\nClassical ML Accuracy:")
print(f"  {classical_accuracy * 100:.2f}%")

print("\nQuantum ML Accuracy:")
print(f"  {quantum_accuracy * 100:.2f}%")

print("\nQuantum configuration:")
print("  Qubits: 4")
print("  Feature Map: ZZFeatureMap")
print("  Ansatz: RealAmplitudes")
print("  Optimizer: COBYLA")
print("  Sampler: StatevectorSampler")

print("\nSaved:")
print("  breast_cancer_models.pkl")

print("\nBreastCancerQML model training finished.")
print("=" * 60)

