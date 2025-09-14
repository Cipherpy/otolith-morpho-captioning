# ---------- NEW: Analysis utilities on embeddings ----------
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet152
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

# =========== CONFIG ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TEST_DIR = '/home/reshma/Otolith/otolith_Final/model_data/test'
OOD_ROOT = '/home/reshma/Otolith/otolith_Final/OOD'
MODEL_PATH = '/home/reshma/Otolith/otolith_Final/src/outputs/20250515_095917/models/resnet152_no_augment_color_size512_best.pth'
# =================================

# -------- 1. Data Transform -------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------- 2. Load Model -----------
model = resnet152(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 30)  # Adapt to your #classes if different
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.fc = nn.Identity()  # Remove classifier, get embeddings only
model.to(DEVICE)
model.eval()

# -------- 3. Load In-Distribution (Test) Data ----------
test_dataset = ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes

test_embeddings = []
test_labels = []

with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Test set"):
        images = images.to(DEVICE)
        feats = model(images)
        test_embeddings.append(feats.cpu())
        test_labels.extend(targets.cpu().numpy())

test_embeddings = torch.cat(test_embeddings).numpy()
test_labels = [class_names[l] for l in test_labels]  # Use class names, not indices

# -------- 4. Load OOD Data ------------
def get_ood_image_paths_and_labels(ood_root):
    ood_paths = []
    ood_labels = []
    ood_sources = []
    for habitat in os.listdir(ood_root):
        habitat_path = os.path.join(ood_root, habitat)
        if not os.path.isdir(habitat_path):
            continue
        for species in os.listdir(habitat_path):
            species_path = os.path.join(habitat_path, species)
            if not os.path.isdir(species_path):
                continue
            for fname in os.listdir(species_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
                    ood_paths.append(os.path.join(species_path, fname))
                    ood_labels.append(f'OOD_{habitat}_{species}')
                    ood_sources.append('OOD')
    return ood_paths, ood_labels, ood_sources

ood_paths, ood_labels, ood_sources = get_ood_image_paths_and_labels(OOD_ROOT)

# -------- 5. Extract OOD Embeddings -----------
ood_embeddings = []
with torch.no_grad():
    for img_path in tqdm(ood_paths, desc="OOD set"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(DEVICE)
            feat = model(img_t).cpu().numpy().squeeze()
            ood_embeddings.append(feat)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

ood_embeddings = np.stack(ood_embeddings)



# ID-only arrays
X_id = test_embeddings  # shape [N_id, D]
y_id = np.array(test_labels)

# 1) PCA subspace projection (fit on ID only)
K = 50  # number of PCs; try 32/64/128 based on D and N
pca = PCA(n_components=K, random_state=42)
X_id_p = pca.fit_transform(X_id)               # projected ID (K-d)
X_id_rec = pca.inverse_transform(X_id_p)       # back to D
id_residual = np.linalg.norm(X_id - X_id_rec, axis=1)

# Apply to OOD:
X_ood = ood_embeddings
X_ood_p = pca.transform(X_ood)
X_ood_rec = pca.inverse_transform(X_ood_p)
ood_residual = np.linalg.norm(X_ood - X_ood_rec, axis=1)

# 2) Class centroids + (diagonal) covariance and Mahalanobis distance
#    (Diagonal covariance is robust and avoids singular matrices; use full cov if you have data)
classes = np.unique(y_id)
centroids = {}
diag_cov_inv = {}  # inverse of diagonal covariance
eps = 1e-6

for c in classes:
    Xc = X_id[y_id == c]
    mu = Xc.mean(axis=0)
    var = Xc.var(axis=0) + eps
    centroids[c] = mu
    diag_cov_inv[c] = 1.0 / var

def maha_diag(x, mu, inv_var):
    # Mahalanobis with diagonal covariance: sum((x-mu)^2 * inv_var)
    diff = x - mu
    return np.sqrt(np.sum(diff * diff * inv_var, axis=1))

# Distances to all class centroids; keep min and argmin
def min_maha_and_argmin(X):
    dists = []
    for c in classes:
        d = maha_diag(X, centroids[c], diag_cov_inv[c])
        dists.append(d[:, None])
    D = np.hstack(dists)          # [N, C]
    argmin = D.argmin(axis=1)     # nearest class index
    minval = D.min(axis=1)        # smallest mahala distance
    return minval, argmin

id_maha_min, id_maha_arg = min_maha_and_argmin(X_id)
ood_maha_min, ood_maha_arg = min_maha_and_argmin(X_ood)

# 3) kNN density proxy: mean distance to k nearest ID neighbors
k = 50
nn = NearestNeighbors(n_neighbors=min(k, len(X_id)), algorithm='auto')
nn.fit(X_id)
id_knn_dist, _  = nn.kneighbors(X_id)
ood_knn_dist, _ = nn.kneighbors(X_ood)
id_knn_mean  = id_knn_dist.mean(axis=1)
ood_knn_mean = ood_knn_dist.mean(axis=1)

# ---------- Build a unified DataFrame with new metrics ----------
id_df = pd.DataFrame({
    'path': [p for p, _ in test_dataset.samples],
    'source': 'Test',
    'true_label': y_id,
    'assign_class': classes[id_maha_arg],
    'maha_min': id_maha_min,
    'pca_residual': id_residual,
    'knn_mean': id_knn_mean
})

ood_df = pd.DataFrame({
    'path': ood_paths,
    'source': 'OOD',
    'true_label': ood_labels,         # OOD_* labels you built
    'assign_class': classes[ood_maha_arg],
    'maha_min': ood_maha_min,
    'pca_residual': ood_residual,
    'knn_mean': ood_knn_mean
})

metrics_df = pd.concat([id_df, ood_df], ignore_index=True)
metrics_df.to_csv("embedding_metrics_id_ood.csv", index=False)

print("Summary (lower is 'more ID-like'):")
print(metrics_df.groupby('source')[['maha_min', 'pca_residual', 'knn_mean']].describe())

# ---------- Optional: 2D embedding on PCA-projected data ----------
# Use your existing TSNE but on [X_id_p; X_ood_p] for cleaner geometry
from sklearn.manifold import TSNE

X_all_proj2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(
    np.vstack([X_id_p, X_ood_p])
)
src = np.array(['Test']*len(X_id_p) + ['OOD']*len(X_ood_p))
assign = np.concatenate([classes[id_maha_arg], classes[ood_maha_arg]])
pca_res = np.concatenate([id_residual, ood_residual])
knn_m = np.concatenate([id_knn_mean, ood_knn_mean])

viz_df = pd.DataFrame({
    'x': X_all_proj2d[:,0],
    'y': X_all_proj2d[:,1],
    'source': src,
    'assign_class': assign,
    'pca_residual': pca_res,
    'knn_mean': knn_m
})
viz_df.to_csv("tsne_on_pca_projected.csv", index=False)

# Plotly: color by assigned class, symbol by source, size by residual
import plotly.express as px
fig2 = px.scatter(
    viz_df, x='x', y='y',
    color='assign_class', symbol='source',
    size='pca_residual', hover_data=['knn_mean', 'pca_residual'],
    title='t-SNE on PCA-Projected Space (size = PCA residual)'
)
fig2.update_traces(marker=dict(opacity=0.85))
fig2.write_html("tsne_pca_projected.html")
fig2.show()
