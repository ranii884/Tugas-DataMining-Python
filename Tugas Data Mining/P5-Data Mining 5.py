import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Generate berbagai jenis dataset
datasets = {
    'Linear': make_classification(n_samples=500, n_features=10, n_informative=8, random_state=42),
    'Moons': make_moons(n_samples=500, noise=0.1, random_state=42),
    'Circles': make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
}

# 2. Model yang akan dibandingkan
models = {
    'Naive Bayes': GaussianNB(),
    'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'k-NN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'k-NN (k=15)': KNeighborsClassifier(n_neighbors=15)
}

# 3. Bandingkan performa
results = []

for dataset_name, (X, y) in datasets.items():
    # Normalisasi untuk k-NN
    X_scaled = StandardScaler().fit_transform(X)

    for model_name, model in models.items():
        # Logika pemilihan data: k-NN butuh scaling, NB tidak wajib untuk data ini
        if 'k-NN' in model_name:
            X_use = X_scaled
        else:
            X_use = X

        # Cross-validation
        scores = cross_val_score(model, X_use, y, cv=5, scoring='accuracy')

        results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Mean Accuracy': scores.mean(),
            'Std': scores.std()
        })

# 4. Tampilkan hasil dalam DataFrame
results_df = pd.DataFrame(results)
print("Hasil Performa Lengkap:")
print(results_df.round(4))

# 5. Pivot table untuk perbandingan yang lebih mudah dibaca
pivot = results_df.pivot(index='Dataset', columns='Model', values='Mean Accuracy')
print("\nPerbandingan Akurasi Rata-rata (Pivot):")
print(pivot.round(4))