from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
import numpy as np
import cv2
from PIL import Image

global qann, ansatz

def loadQANN():
    global qann, ansatz
    # 1. Define a Quantum Feature Map
    num_features = 8
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    # 2. Construct a Quantum Circuit (QANN) with an ansatz
    ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
    qc = QuantumCircuit(num_features)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    # Define an observable for feature extraction (e.g., Z-measurement on all qubits)
    observable = QuantumCircuit(num_features, name='observable')
    for i in range(num_features):
        observable.z(i)
    # 3. Create a QNN Instance
    estimator = Estimator()
    qann = EstimatorQNN(circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters, estimator=estimator, observables=[observable])
    return qann

def getQANNFeatures(img_path, qann):
    global ansatz
    img = cv2.imread(img_path)
    img = cv2.resize(img, (8, 8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 5. Extract "features" (expectation values in this case)
    # The forward pass of the QNN with input data and initial weights (e.g., random)
    # will yield the "features" represented by the expectation values of the observables.
    initial_weights = [0.47249139226894876, 0.4770202391038413, 0.32068646301478265, 0.23300215217830522, 0.20925357404918843, 0.9127976893516987, 0.7181644015440662,
                       0.04767789875059503, 0.2728995913455713, 0.10277537615091126,
                       0.9799993421687556, 0.7016872494296539, 0.6039389322210846, 0.11189121858759932, 0.7573894527668734, 0.09017659012296453]
    extracted_features = qann.forward(input_data=img, weights=initial_weights)
    cv2.imwrite("test.jpg", extracted_features)
    extracted_features = cv2.imread("test.jpg")
    extracted_features = cv2.resize(extracted_features, (128, 128))
    extracted_features = Image.open(img_path).convert('L')
    extracted_features = np.array(extracted_features)
    U, S, V = np.linalg.svd(extracted_features)
    S = np.diag(S)
    k = 50
    #calculate optimial values
    extracted_features = U[:, :k] @ S[0:k,:k] @ V[:k, :]
    extracted_features = Image.fromarray(extracted_features.astype(np.uint8))
    extracted_features.save('test.jpg')
    extracted_features = cv2.imread('test.jpg')
    extracted_features = cv2.resize(extracted_features, (128, 128), cv2.INTER_LANCZOS4)
    return extracted_features

#loadQANN()
#getQANNFeatures("Dataset/001.jpeg")
