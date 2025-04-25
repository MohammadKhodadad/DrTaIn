#Final Model

import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import MinMaxScaler

# Convert SMILES to Morgan Fingerprints
def smiles_to_fingerprint(smiles, radius=4, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        generator = GetMorganGenerator(radius=radius, fpSize=nBits)
        return np.array(generator.GetFingerprint(mol))
    else:
        return None

# Load dataset from CSV
def load_data(csv_file):
    df = pd.read_csv(csv_file) 

    print("First 5 Rows of the Dataset:")
    print(df.head())  # Check raw data
    
    # Convert SMILES to fingerprints
    fingerprints = [smiles_to_fingerprint(smiles) for smiles in df['smiles']]
    valid_indices = [i for i, fp in enumerate(fingerprints) if fp is not None]

    # Keep only valid entries
    X = np.array([fingerprints[i] for i in valid_indices])
    y = np.array([df['affinity'].iloc[i] for i in valid_indices])  

    print("Raw Affinity Values Range Before Scaling: ", np.min(y), np.max(y))  

    # Apply MinMaxScaler before train_test_split
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    print("Scaled Affinity Values Range: ", np.min(y), np.max(y))  

    return X, y, scaler  # Return scaler to use later

# Load data from CSV
csv_file = 'C:/Users/ssadmin/Desktop/projects/DD/Model/docked_leadlike.csv'  #file path
X, y, scaler = load_data(csv_file)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check before training
print("Training Actual Values Range BEFORE Training: ", np.min(y_train), np.max(y_train))
print("Test Actual Values Range BEFORE Training: ", np.min(y_test), np.max(y_test))

# Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Regularization
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1) # Output layer predicting binding affinity
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss=tf.keras.losses.Huber(), metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Reverse scaling correctly
y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Compute R²
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")

# Check after training
print("Training Actual Values Range AFTER Training: ", np.min(y_train), np.max(y_train))
print("Test Actual Values Range AFTER Training: ", np.min(y_test), np.max(y_test))

# Test with an example molecule
def generate_molecule_suggestion(input_smiles):
    fp = smiles_to_fingerprint(input_smiles)
    if fp is not None:
        fp = np.array(fp).reshape(1, -1)
        predicted_affinity = model.predict(fp)
        predicted_affinity = scaler.inverse_transform(predicted_affinity.reshape(-1, 1)).flatten()[0]
        return predicted_affinity
    else:
        return None

new_molecule_1 = 'O=C(NC1CC[NH+](Cc2ccccc2)CC1)c1oc(CN2C(=O)COc3ccccc23)cc1'  # Expected: ~  -8.5175
predicted_affinity = generate_molecule_suggestion(new_molecule_1)
print(f"Predicted Binding Affinity for {new_molecule_1}: {predicted_affinity}")
print(f"Actual Binding Affinity for    {new_molecule_1}: -8.5175")

new_molecule_2 = 'FC(F)(F)c1c(C(=O)[O-])c(C(=O)[O-])ccc1F'  # Expected: ~  -2.7969611
predicted_affinity = generate_molecule_suggestion(new_molecule_2)
print(f"Predicted Binding Affinity for {new_molecule_2}: {predicted_affinity}")
print(f"Actual Binding Affinity for    {new_molecule_2}: -2.7970")

# Final check on actual values
print("Final Training Actual Values Sample: ", y_train[:5])
print("Final Training Predicted Values Sample: ", y_train_pred[:5])
print("Final Test Actual Values Sample: ", y_test[:5])
print("Final Test Predicted Values Sample: ", y_test_pred[:5])

# Fit scaler on training labels
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

# Save the trained scaler
scaler_path = 'C:/Users/ssadmin/Desktop/projects/DD/Model/scaler.pkl'
joblib.dump(scaler, scaler_path)

# Plot R2
def plot_r2(y_actual, y_pred, dataset_name, r2_value):
    errors = np.abs(y_actual - y_pred)  # Calculate absolute deviation
    norm = plt.Normalize(errors.min(), errors.max())  # Normalize errors for colormap
    cmap = plt.cm.viridis  # Use a perceptually uniform colormap

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(y_actual, y_pred, c=errors, cmap=cmap, norm=norm, alpha=0.7, edgecolors='black')
    plt.colorbar(sc, label='Absolute Error')  # Show error scale
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', linestyle='dashed', label="y = x")
    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{dataset_name} Set: Predicted vs Actual (R² = {r2_value:.4f})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for Training Data
plot_r2(y_train, y_train_pred, "Training", r2_train)

# Plot for Validation Data
plot_r2(y_test, y_test_pred, "Validation", r2_test)

# Define common bin edges for training set
bins_train = np.histogram_bin_edges(np.concatenate([y_train, y_train_pred]), bins=30)

# Plot histograms for training data with shared bins
plt.hist(y_train, bins=bins_train, alpha=0.5, label='Training Actual', color='red')
plt.hist(y_train_pred, bins=bins_train, alpha=0.5, label='Training Predicted', color='blue')
plt.legend()
plt.title("Distribution of Actual and Predicted (Training Data)")
plt.show()

# Define common bin edges for test set
bins_test = np.histogram_bin_edges(np.concatenate([y_test, y_test_pred]), bins=30)

# Plot histograms for test data with shared bins
plt.hist(y_test, bins=bins_test, alpha=0.5, label='Test Actual', color='red')
plt.hist(y_test_pred, bins=bins_test, alpha=0.5, label='Test Predicted', color='blue')
plt.legend()
plt.title("Distribution of Actual and Predicted (Test Data)")
plt.show()

# Calculate and compare AUC (Area Under Curve)
bins = np.linspace(min(y_test.min(), y_test_pred.min()), 
                   max(y_test.max(), y_test_pred.max()), 50)

hist_actual, _ = np.histogram(y_test, bins=bins, density=True)
hist_predicted, _ = np.histogram(y_test_pred, bins=bins, density=True)

# Compute AUC using the trapezoidal rule
auc_actual = np.trapz(hist_actual, bins[:-1])
auc_predicted = np.trapz(hist_predicted, bins[:-1])

print(f"Area under actual distribution: {auc_actual:.4f}")
print(f"Area under predicted distribution: {auc_predicted:.4f}")

# Compute residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Define common bin edges for residuals
bins_residuals = np.histogram_bin_edges(np.concatenate([train_residuals, test_residuals]), bins=30)

# Plot residual histogram for training set
plt.hist(train_residuals, bins=bins_residuals, alpha=0.5, label="Training Residuals", color="green")
# Plot residual histogram for test set
plt.hist(test_residuals, bins=bins_residuals, alpha=0.5, label="Test Residuals", color="purple")

plt.legend()
plt.title("Residual Distribution (Training & Test)")
plt.show()

#residuals vs. predicted values
plt.scatter(y_test_pred, test_residuals, alpha=0.5, label="Test Residuals", color="purple")
plt.axhline(y=0, color='black', linestyle='dashed')  
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.legend()
plt.show()