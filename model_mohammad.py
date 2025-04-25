import joblib
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load and scale target data from CSV
#    (replace 'E_score2' with whichever numeric column you want as the label)
csv_file = "docked_leadlike.csv"
df = pd.read_csv(csv_file)
print("Available columns:", df.columns.tolist())

smiles_list = df["smiles"].astype(str).tolist()
# Use E_score2 as the target affinity/docking score
affinities = df["E_score2"].astype(float).values.reshape(-1, 1)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(affinities).flatten()

# 2. Set up tokenizer & max length
checkpoint = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
max_length = 256

class AffinityDataset(Dataset):
    def __init__(self, smiles, targets, tokenizer, max_length):
        self.smiles = smiles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        t = self.targets[idx]
        toks = self.tokenizer(
            s,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = toks['input_ids'].squeeze(0)
        attention_mask = toks['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(t, dtype=torch.float)

# 3. Split data
train_smiles, test_smiles, train_y, test_y = train_test_split(
    smiles_list, y_scaled, test_size=0.3, random_state=42
)

train_ds = AffinityDataset(train_smiles, train_y, tokenizer, max_length)
test_ds  = AffinityDataset(test_smiles,  test_y,  tokenizer, max_length)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# 4. Model definition
class ChemBERTaRegressor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = outputs.pooler_output  # CLS token pooler
        return self.regressor(pooled).squeeze(-1)

# Instantiate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChemBERTaRegressor(checkpoint).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.SmoothL1Loss()  # Huber-like

epochs = 5
# 5. Training loop
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, targets in tqdm.tqdm(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs} - Training Loss: {avg_loss:.4f}")

# 6. Evaluation & predictions
def evaluate(loader):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm.tqdm(loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            preds = model(input_ids, attention_mask).cpu().numpy()
            all_preds.append(preds)
            all_true.append(targets.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    return y_true, y_pred

train_true_scaled, train_pred_scaled = evaluate(train_loader)
test_true_scaled,  test_pred_scaled  = evaluate(test_loader)

# inverse scale
train_true = scaler.inverse_transform(train_true_scaled.reshape(-1,1)).flatten()
train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1,1)).flatten()
test_true  = scaler.inverse_transform(test_true_scaled.reshape(-1,1)).flatten()
test_pred  = scaler.inverse_transform(test_pred_scaled.reshape(-1,1)).flatten()

r2_train = r2_score(train_true, train_pred)
r2_test  = r2_score(test_true,  test_pred)
print(f"R² (train): {r2_train:.4f}")
print(f"R² (test) : {r2_test:.4f}")

# 7. Save model & scaler
torch.save(model.state_dict(), "chemberta_affinity_torch.pt")
joblib.dump(scaler, "affinity_scaler.pkl")

# 8. Visualization
def plot_r2(actual, pred, title, r2):
    errors = np.abs(actual - pred)
    norm = plt.Normalize(errors.min(), errors.max())
    plt.figure(figsize=(6,6))
    sc = plt.scatter(actual, pred, c=errors, cmap='viridis', norm=norm, alpha=0.7, edgecolors='black')
    plt.colorbar(sc, label='Absolute Error')
    mn, mx = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    plt.plot([mn,mx],[mn,mx],'r--',label='y = x')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f"{title} (R² = {r2:.4f})")
    plt.legend(); plt.grid(True); plt.show()

plot_r2(train_true, train_pred, 'Training Set', r2_train)
plot_r2(test_true,  test_pred,  'Validation Set', r2_test)

# Histograms
def plot_hist(actual, pred, title):
    bins = np.histogram_bin_edges(np.concatenate([actual, pred]), bins=30)
    plt.figure(); plt.hist(actual, bins=bins, alpha=0.5, label='Actual')
    plt.hist(pred,   bins=bins, alpha=0.5, label='Predicted')
    plt.title(f'Distribution: {title}'); plt.legend(); plt.show()

plot_hist(train_true, train_pred, 'Training')
plot_hist(test_true,  test_pred,  'Validation')

# AUC comparison
bins_auc = np.linspace(min(test_true.min(), test_pred.min()), max(test_true.max(), test_pred.max()), 50)
auc_true = np.trapz(np.histogram(test_true, bins=bins_auc, density=True)[0], bins_auc[:-1])
auc_pred = np.trapz(np.histogram(test_pred, bins=bins_auc, density=True)[0], bins_auc[:-1])
print(f"AUC actual:    {auc_true:.4f}")
print(f"AUC predicted: {auc_pred:.4f}")

# Residuals
def plot_residuals(actual, pred, title):
    res = actual - pred
    bins = np.histogram_bin_edges(res, bins=30)
    plt.figure(); plt.hist(res, bins=bins, alpha=0.7); plt.title(f'Residuals: {title}'); plt.show()
    plt.figure(); plt.scatter(pred, res, alpha=0.7); plt.axhline(0, linestyle='--',color='black');
    plt.xlabel('Predicted'); plt.ylabel('Residual'); plt.title(f'Residuals vs Predicted: {title}'); plt.show()

plot_residuals(train_true, train_pred, 'Training')
plot_residuals(test_true,  test_pred,  'Validation')