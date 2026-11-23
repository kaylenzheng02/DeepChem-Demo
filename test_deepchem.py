import truststore
truststore.inject_into_ssl()

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import deepchem as dc
# imported directly - in this dc build, dc.models only exposes
# multitalkregressor when tensorflow installed
from deepchem.models.fcnet import MultitaskRegressor

#fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# load & featurize the esol dataset
print("loading and featurizing esol dataset...")
tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer="ECFP",
    splitter="scaffold"
)
train_dataset, valid_dataset, test_dataset = datasets

# initialize pyttorch mlp model my little pony 
model = MultitaskRegressor(
    n_tasks=len(tasks),
    n_features=1024,
    layer_sizes=[1000],
    dropouts=0.2,
    learning_rate=0.001
)

# 3 wow train model 
print("training model...")
model.fit(train_dataset, nb_epoch=30)

# 4 eval neural network w metrics RMSE, MAE, R2
metrics = [
    dc.metrics.Metric(dc.metrics.rms_score, name="RMSE"),
    dc.metrics.Metric(dc.metrics.mae_score, name="MAE"),
    dc.metrics.Metric(dc.metrics.r2, name="R2"),
]
train_scores = model.evaluate(train_dataset, metrics, transformers)
test_scores = model.evaluate(test_dataset, metrics, transformers)

#5 load original esol dataset as a baseline for comp
csv_path = os.path.join(dc.utils.data_utils.get_data_dir(), "delaney-processed.csv")
raw = pd.read_csv(csv_path).drop_duplicates(subset="smiles").set_index("smiles")

#keep only molecules in test set in the same order
test_smiles = list(test_dataset.ids)
esol_pred = raw.loc[test_smiles, "ESOL predicted log solubility in mols/L"].values
measured = raw.loc[test_smiles, "measured log soluability in mols/L"].values

esol_scores = {
    "RMSE": np.sqrt(mean_squared_error(measured, esol_pred)),
    "MAE": mean_absolute_error(measured, esol_pred),
    "R2": r2_score(measured, esol_pred),
}

# 6 comp table
print(f"\nTest set results (scaffold split, {len(test_smiles)} molecules)")
print(f"{'Method':<28}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
print("-" * 52)
print(f"{'ESOL formula (2004)':<28}{esol_scores['RMSE']:>8.3f}{esol_scores['MAE']:>8.3f}{esol_scores['R2']:>8.3f}")
print(f"{'Neural net + ECFP':<28}{test_scores['RMSE']:>8.3f}{test_scores['MAE']:>8.3f}{test_scores['R2']:>8.3f}")
print(f"\n(Neural net train RMSE: {train_scores['RMSE']:.3f}, for checking overfitting)")

# 7 predict solubility on new molecules (SMILES)
new_smiles = [
    "CCO",                          #ethanol
    "CC(=O)OC1=CC=CC=C1C(=O)O",     #aspirin
    "C1=CC=C2C=CC=CC2=C1",          #naphthalene
]
featurizer = dc.feat.CircularFingerprints(size=1024)
features = featurizer.featurize(new_smiles)
predictions = model.predict_on_batch(features)
real_predictions = dc.trans.undo_transforms(predictions, transformers)

print(f"\nPredicted Log Solubility (mol/L):")
for smiles, pred in zip(new_smiles, real_predictions):
    print(f"  {smiles:<26} -> {float(pred.ravel()[0]):.2f}")

try:
    import matplotlib
    matplotlib.use("Agg")  # save to file without opening a window
    import matplotlib.pyplot as plt

    nn_pred = dc.trans.undo_transforms(model.predict(test_dataset), transformers).ravel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(measured, esol_pred, alpha=0.6, label="ESOL formula")
    ax.scatter(measured, nn_pred, alpha=0.6, label="Neural net + ECFP")
    lo = min(measured.min(), esol_pred.min(), nn_pred.min()) - 0.5
    hi = max(measured.max(), esol_pred.max(), nn_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Measured log solubility (mol/L)")
    ax.set_ylabel("Predicted log solubility (mol/L)")
    ax.set_title("Test set: predicted vs. measured")
    ax.legend()
    fig.savefig("predicted_vs_measured.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to predicted_vs_measured.png")
except ImportError:
    print("\n(Skipping plot: run 'pip install matplotlib' to enable it)")