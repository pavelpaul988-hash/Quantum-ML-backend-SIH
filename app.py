# ============================================================
# BREASTCANCERQML
# Flask Backend API
# Hybrid Classical + Quantum Machine Learning
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import joblib

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

MODEL_FILE = "breast_cancer_models.pkl"


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "meanRadius",
    "meanTexture",
    "meanPerimeter",
    "meanArea",
    "meanSmoothness",
    "meanCompactness",
    "meanConcavity",
    "meanConcavePoints",
    "meanSymmetry",
    "meanFractalDimension"
]

DISPLAY_FEATURE_NAMES = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension"
]


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

classical_model = None
classical_scaler = None

quantum_scaler = None
pca = None

quantum_model = None
quantum_weights = None

feature_importance = {}

classical_accuracy = 0.0
quantum_accuracy = 0.0


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

try:

    print("\nLoading model package...")

    model_package = joblib.load(MODEL_FILE)

    classical_model = model_package["classical_model"]
    classical_scaler = model_package["classical_scaler"]

    quantum_scaler = model_package["quantum_scaler"]
    pca = model_package["pca"]

    quantum_accuracy = model_package.get(
        "quantum_accuracy",
        0.0
    )

    classical_accuracy = model_package.get(
        "classical_accuracy",
        0.0
    )

    feature_importance = model_package.get(
        "feature_importance",
        {}
    )

    # --------------------------------------------------------
    # LOAD SAVED QUANTUM WEIGHTS
    # --------------------------------------------------------

    quantum_weights = np.asarray(
        model_package["quantum_weights"],
        dtype=float
    )

    print("Rebuilding 4-qubit Quantum VQC...")

    feature_map = zz_feature_map(
        feature_dimension=4,
        reps=1
    )

    ansatz = real_amplitudes(
        num_qubits=4,
        reps=1
    )

    sampler = StatevectorSampler(
        seed=42
    )

    quantum_model = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        initial_point=quantum_weights
    )

    print(
        "Quantum VQC weights restored successfully."
    )

    print("\nModel package loaded successfully.")
    print(
        f"Classical accuracy: "
        f"{classical_accuracy * 100:.2f}%"
    )

    print(
        f"Quantum accuracy: "
        f"{quantum_accuracy * 100:.2f}%"
    )

    print(
        f"Quantum parameters: "
        f"{len(quantum_weights)}"
    )

except Exception as error:

    print("\nERROR loading model package:")
    print(error)

    classical_model = None
    classical_scaler = None
    quantum_scaler = None
    pca = None
    quantum_model = None
    quantum_weights = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_features(data):

    values = []

    for feature in FEATURE_NAMES:

        if feature not in data:
            raise ValueError(
                f"Missing feature: {feature}"
            )

        try:

            value = float(
                data[feature]
            )

        except Exception:

            raise ValueError(
                f"Invalid value for {feature}"
            )

        if not np.isfinite(value):

            raise ValueError(
                f"Invalid numeric value for {feature}"
            )

        values.append(value)

    return np.array(
        values,
        dtype=float
    ).reshape(1, -1)


def safe_probability(value):

    try:

        value = float(value)

    except Exception:

        return 0.5

    return max(
        0.0,
        min(1.0, value)
    )


def label_from_prediction(prediction):

    prediction = int(prediction)

    if prediction == 0:
        return "malignant"

    return "benign"


def selected_feature_names():

    return [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension"
    ]


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "status": "online",

        "project": "BreastCancerQML",

        "description":
            "Hybrid Classical and Quantum Machine "
            "Learning for Breast Cancer Classification",

        "classical_model":
            "Random Forest",

        "quantum_model":
            "Qiskit VQC",

        "qubits": 4,

        "features": 10,

        "message":
            "BreastCancerQML backend is running.",

        "warning":
            "Research and educational prototype. "
            "Not a medical diagnosis."

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "online",

        "project": "BreastCancerQML",

        "description":
            "Hybrid Classical and Quantum Machine "
            "Learning for Breast Cancer Classification",

        "model_loaded":
            classical_model is not None
            and quantum_model is not None,

        "classical_model":
            "Random Forest",

        "quantum_model":
            "Qiskit VQC",

        "classical_accuracy":
            classical_accuracy,

        "quantum_accuracy":
            quantum_accuracy,

        "qubits": 4,

        "features": 10,

        "warning":
            "Research and educational prototype. "
            "Not a medical diagnosis."

    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # CHECK MODELS
        # ----------------------------------------------------

        if classical_model is None:

            return jsonify({

                "status": "error",

                "success": False,

                "error":
                    "Classical model is not loaded."

            }), 500


        if quantum_model is None:

            return jsonify({

                "status": "error",

                "success": False,

                "error":
                    "Quantum model is not loaded."

            }), 500


        # ----------------------------------------------------
        # READ REQUEST
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "status": "error",

                "success": False,

                "error":
                    "No JSON data received."

            }), 400


        # ----------------------------------------------------
        # SUPPORT BOTH FORMATS
        #
        # { "features": {...} }
        #
        # OR
        #
        # { "meanRadius": ..., ... }
        # ----------------------------------------------------

        if "features" in data:

            feature_data = data["features"]

        else:

            feature_data = data


        # ----------------------------------------------------
        # CONVERT FEATURES
        # ----------------------------------------------------

        X = convert_features(
            feature_data
        )


        # ====================================================
        # CLASSICAL MACHINE LEARNING
        # ====================================================

        X_classical = (
            classical_scaler.transform(X)
        )

        classical_prediction_raw = (
            classical_model.predict(
                X_classical
            )
        )

        classical_prediction = int(
            np.asarray(
                classical_prediction_raw
            ).reshape(-1)[0]
        )

        classical_label = (
            label_from_prediction(
                classical_prediction
            )
        )


        # ----------------------------------------------------
        # CLASSICAL PROBABILITY
        # ----------------------------------------------------

        classical_probability_raw = (
            classical_model.predict_proba(
                X_classical
            )
        )

        classical_probability_raw = (
            np.asarray(
                classical_probability_raw
            )
        )


        if (
            classical_probability_raw.ndim == 2
            and classical_probability_raw.shape[1] >= 2
        ):

            malignant_probability = (
                safe_probability(
                    classical_probability_raw[0][0]
                )
            )

            benign_probability = (
                safe_probability(
                    classical_probability_raw[0][1]
                )
            )

        else:

            malignant_probability = 0.5
            benign_probability = 0.5


        classical_confidence = max(
            malignant_probability,
            benign_probability
        )


        # ====================================================
        # QUANTUM MACHINE LEARNING
        # ====================================================

        X_quantum_scaled = (
            quantum_scaler.transform(X)
        )

        X_quantum = (
            pca.transform(
                X_quantum_scaled
            )
        )


        # ----------------------------------------------------
        # IMPORTANT FIX
        #
        # Do NOT use:
        #
        # quantum_model.predict()
        #
        # because the rebuilt VQC object has not been
        # fitted in this process.
        #
        # Instead, use the saved trained weights directly
        # through the neural network.
        # ----------------------------------------------------

        quantum_output = (
            quantum_model.neural_network.forward(
                X_quantum,
                quantum_weights
            )
        )


        quantum_output = np.asarray(
            quantum_output
        ).reshape(-1)


        # ----------------------------------------------------
        # QUANTUM PROBABILITIES
        # ----------------------------------------------------

        if len(quantum_output) >= 2:

            quantum_malignant_probability = (
                safe_probability(
                    quantum_output[0]
                )
            )

            quantum_benign_probability = (
                safe_probability(
                    quantum_output[1]
                )
            )

        elif len(quantum_output) == 1:

            value = safe_probability(
                quantum_output[0]
            )

            quantum_malignant_probability = value

            quantum_benign_probability = (
                1.0 - value
            )

        else:

            quantum_malignant_probability = 0.5
            quantum_benign_probability = 0.5


        # ----------------------------------------------------
        # NORMALIZE QUANTUM PROBABILITIES
        # ----------------------------------------------------

        quantum_total = (
            quantum_malignant_probability
            + quantum_benign_probability
        )


        if quantum_total > 0:

            quantum_malignant_probability /= (
                quantum_total
            )

            quantum_benign_probability /= (
                quantum_total
            )


        # ----------------------------------------------------
        # QUANTUM CLASSIFICATION
        # ----------------------------------------------------

        if (
            quantum_malignant_probability
            >= quantum_benign_probability
        ):

            quantum_prediction = 0

            quantum_label = "malignant"

        else:

            quantum_prediction = 1

            quantum_label = "benign"


        quantum_confidence = max(
            quantum_malignant_probability,
            quantum_benign_probability
        )


        # ====================================================
        # HYBRID MODEL
        # ====================================================

        hybrid_malignant_probability = (

            0.70 * malignant_probability

            +

            0.30 * quantum_malignant_probability

        )


        hybrid_benign_probability = (

            0.70 * benign_probability

            +

            0.30 * quantum_benign_probability

        )


        # ----------------------------------------------------
        # NORMALIZE HYBRID
        # ----------------------------------------------------

        hybrid_total = (

            hybrid_malignant_probability

            +

            hybrid_benign_probability

        )


        if hybrid_total > 0:

            hybrid_malignant_probability /= (
                hybrid_total
            )

            hybrid_benign_probability /= (
                hybrid_total
            )


        # ----------------------------------------------------
        # HYBRID CLASSIFICATION
        # ----------------------------------------------------

        if (
            hybrid_malignant_probability
            >= hybrid_benign_probability
        ):

            hybrid_prediction = 0

            hybrid_label = "malignant"

        else:

            hybrid_prediction = 1

            hybrid_label = "benign"


        hybrid_confidence = max(

            hybrid_malignant_probability,

            hybrid_benign_probability

        )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        feature_importance_result = {}

        if feature_importance:

            for index, feature in enumerate(
                DISPLAY_FEATURE_NAMES
            ):

                key = FEATURE_NAMES[index]

                value = feature_importance.get(
                    key,
                    feature_importance.get(
                        feature,
                        0.0
                    )
                )

                feature_importance_result[
                    feature
                ] = float(value)


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response = {

            # ------------------------------------------------
            # FRONTEND COMPATIBILITY
            # ------------------------------------------------

            "status": "success",

            "success": True,


            # ------------------------------------------------
            # MAIN CLASSIFICATION
            # ------------------------------------------------

            "classification":
                hybrid_label,

            "prediction":
                hybrid_prediction,

            "confidence":
                hybrid_confidence,


            # ------------------------------------------------
            # CLASSICAL ML
            # ------------------------------------------------

            "classical_ml": {

                "prediction":
                    classical_prediction,

                "classification":
                    classical_label,

                "confidence":
                    classical_confidence,

                "malignant_probability":
                    malignant_probability,

                "benign_probability":
                    benign_probability,

                "accuracy":
                    classical_accuracy

            },


            # ------------------------------------------------
            # QUANTUM ML
            # ------------------------------------------------

            "quantum_ml": {

                "prediction":
                    quantum_prediction,

                "classification":
                    quantum_label,

                "confidence":
                    quantum_confidence,

                "malignant_probability":
                    quantum_malignant_probability,

                "benign_probability":
                    quantum_benign_probability,

                "accuracy":
                    quantum_accuracy,

                "qubits": 4

            },


            # ------------------------------------------------
            # HYBRID AI
            # ------------------------------------------------

            "hybrid": {

                "prediction":
                    hybrid_prediction,

                "classification":
                    hybrid_label,

                "confidence":
                    hybrid_confidence,

                "malignant_probability":
                    hybrid_malignant_probability,

                "benign_probability":
                    hybrid_benign_probability,

                "classical_weight":
                    0.70,

                "quantum_weight":
                    0.30

            },


            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            "feature_importance":
                feature_importance_result,


            # ------------------------------------------------
            # INPUT FEATURES
            # ------------------------------------------------

            "features":
                feature_data,


            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            "metadata": {

                "project":
                    "BreastCancerQML",

                "model":
                    "Hybrid Classical + Quantum ML",

                "classical_model":
                    "Random Forest",

                "quantum_model":
                    "Qiskit VQC",

                "qubits":
                    4,

                "feature_count":
                    10

            },


            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            "disclaimer":
                "Research and educational prototype. "
                "The classification is not a medical diagnosis. "
                "Please consult a qualified medical professional "
                "for medical evaluation."

        }


        return jsonify(
            response
        ), 200


    # ========================================================
    # VALIDATION ERROR
    # ========================================================

    except ValueError as error:

        print(
            "\nPREDICTION VALIDATION ERROR:"
        )

        print(error)

        return jsonify({

            "status": "error",

            "success": False,

            "error":
                str(error)

        }), 400


    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as error:

        print(
            "\nPREDICTION ERROR:"
        )

        print(
            type(error).__name__
        )

        print(
            str(error)
        )

        return jsonify({

            "status": "error",

            "success": False,

            "error":
                str(error),

            "error_type":
                type(error).__name__

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("             BreastCancerQML Backend")
    print("=" * 60)

    print("\nServer:")
    print("http://127.0.0.1:5000")

    print("\nAPI endpoints:")
    print("GET  /")
    print("GET  /api/health")
    print("POST /api/predict")

    print("\nQuantum:")
    print("4-qubit Qiskit VQC")

    print("\nFrontend can connect to:")
    print(
        "http://127.0.0.1:5000/api/predict"
    )

    print("\n")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )