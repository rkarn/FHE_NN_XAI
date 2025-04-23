## Explainability in Encrypted Neural Networks

Investigating how to provide insights into model decisions in FHE settings. This could involve creating mathematical frameworks that allow for the extraction of interpretable features from encrypted models, 
enhancing trust in secure machine learning applications. The notebook `09_Neural_network_MNIST.ipynb` uses SHAP to do so using IBM-FHE toolkit. Way to run it is given in the file `notes.txt` Another FHE example is given in 
`fhe_neuralnetwork_mnist.py` file.



### What we're Proposing:
FHE-Inference: Run inference on encrypted MNIST data using an encrypted neural network, then decrypt the result and compare accuracy with plaintext inference.


SHAP Explainability (Encrypted): Train a SHAP explainer on raw Xtrain. Use FHE-encrypted Xtest to compute SHAP values homomorphically. Decrypt those SHAP values and compare them with the original SHAP values.


Feasibility Breakdown:
Inference on Encrypted Data: This is a common use case in FHE and is supported by frameworks like Concrete-ML (Zama), HEAAN, SEAL, etc. It's computationally expensive but doable for small networks and simple datasets like MNIST.

SHAP on Encrypted Data: This is very challenging. SHAP explainability involves:
- Model perturbation.
- Feature contribution estimation through marginal expectation calculations.
- A lot of floating-point operations and conditional branching.

These operations are not FHE-friendly:
- FHE schemes typically support only a subset of operations (additions, multiplications).
- No native support for branching or floating-point arithmetic (unless approximated).
- SHAP involves significant model re-evaluation, which is costly under FHE.