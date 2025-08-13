import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi Augmentasi
def augment_landmarks_3d(coords, mode='rotate'):
    coords = np.array(coords).reshape(-1, 3)
    if mode == 'rotate':
        angles = np.random.uniform(-15, 15, size=3)
        rot = R.from_euler('xyz', angles, degrees=True)
        coords = rot.apply(coords)
    elif mode == 'scale':
        scale = np.random.uniform(0.9, 1.2)
        coords *= scale
    elif mode == 'jitter':
        coords += np.random.normal(0, 0.003, coords.shape)
    return coords.flatten()

# Load Dataset
dataset_dir = "dataset"
all_data = []

for file in sorted(os.listdir(dataset_dir)):
    if file.endswith(".csv") and len(file) == 5:
        label = file[0]
        df = pd.read_csv(os.path.join(dataset_dir, file))
        if df.shape[1] != 64:
            print(f"⚠️ Lewati {file} karena kolom ≠ 64 (expect 63 fitur + 1 label)")
            continue
        all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
print("📊 Dataset awal (tangan kanan):", df.shape)

# Bersihkan Data
feature_cols = [col for col in df.columns if col != 'class']
df_clean = df[~(df[feature_cols].sum(axis=1) == 0)].reset_index(drop=True)
print("🧹 Setelah cleaning:", df_clean.shape)

# Distribusi Huruf
print("\n📊 Distribusi Data per Huruf (sebelum augmentasi):")
print(df_clean['class'].value_counts().sort_index())

# Augmentasi
augmented = []
for _, row in df_clean.iterrows():
    original = row[feature_cols].values
    label = row['class']
    for mode in ["rotate", "scale", "jitter"]:
        aug = augment_landmarks_3d(original, mode=mode)
        augmented.append(np.append(aug, label))

aug_df = pd.DataFrame(augmented, columns=feature_cols + ['class'])
full_df = pd.concat([df_clean, aug_df], ignore_index=True)
print("🆕 Dataset setelah augmentasi:", full_df.shape)

# Split Data
X = full_df.drop('class', axis=1)
y = full_df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Training Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("\n✅ Akurasi:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Buat confusion matrix
labels_sorted = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

# Visualisasi Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.xlabel('Prediksi')
plt.ylabel('Kelas Asli')
plt.title('Confusion Matrix - Deteksi Huruf SIBI (Tangan Kanan)')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("📸 Confusion matrix disimpan sebagai 'confusion_matrix.png'")

# Simpan model dengan kompresi
joblib.dump(model, "sibi_rf_model.pkl", compress=3)
