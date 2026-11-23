# Use the Windows certificate store for downloads (needed because AVG
# Antivirus scans HTTPS traffic with its own certificate)
import truststore
truststore.inject_into_ssl()

import deepchem as dc
# Imported directly: in this DeepChem nightly build, dc.models only exposes
# MultitaskRegressor when TensorFlow is installed
from deepchem.models.fcnet import MultitaskRegressor

# 1. Load & featurize the Delaney (ESOL) solubility dataset
print("Loading and featurizing dataset...")
tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer="ECFP",
    splitter="scaffold"
)
train_dataset, valid_dataset, test_dataset = datasets

# 2. Initialize a PyTorch Multi-Layer Perceptron (MLP) model
model = MultitaskRegressor(
    n_tasks=len(tasks),
    n_features=1024,
    layer_sizes=[1000],
    dropouts=0.2,
    learning_rate=0.001
)

# 3. Train the model
print("Training model...")
model.fit(train_dataset, nb_epoch=30)

# 4. Evaluate performance
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
train_score = model.evaluate(train_dataset, [metric], transformers)
test_score = model.evaluate(test_dataset, [metric], transformers)
print(f"Train Pearson R^2 Score: {train_score['pearson_r2_score']:.3f}")
print(f"Test Pearson R^2 Score:  {test_score['pearson_r2_score']:.3f}")

# 5. Predict solubility on new molecules (SMILES)
new_smiles = [
    "CCO",                         # Ethanol
    "CC(=O)OC1=CC=CC=C1C(=O)O",    # Aspirin
    "C1=CC=C2C=CC=CC2=C1"          # Naphthalene
]

featurizer = dc.feat.CircularFingerprint(size=1024)
features = featurizer.featurize(new_smiles)
predictions = model.predict_on_batch(features)
real_predictions = dc.trans.undo_transforms(predictions, transformers)

print("\nPredicted Log Solubility (mol/L):")
for smiles, pred in zip(new_smiles, real_predictions):
    print(f"  {smiles:<26} -> {float(pred.ravel()[0]):.2f}")
