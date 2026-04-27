import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# NOMOR 1
# =========================
print("=== NOMOR 1 ===")

P = np.array([1, 2, 3])
Q = np.array([4, 5, 6])

euclid = np.linalg.norm(P - Q)
manhat = np.abs(P - Q).sum()

print(f"Euclidean Distance : {euclid:.2f}")
print(f"Manhattan Distance : {manhat:.2f}")


# =========================
# NOMOR 2
# =========================
print("\n=== NOMOR 2 ===")

X = np.array([1, 0, 1, 0, 1, 1])
Y = np.array([1, 1, 0, 0, 1, 0])

m11 = np.sum((X == 1) & (Y == 1))
m00 = np.sum((X == 0) & (Y == 0))

smc = (m11 + m00) / len(X)

intersection = m11
union = np.sum((X == 1) | (Y == 1))
jaccard = intersection / union

print(f"SMC     : {smc:.2f}")
print(f"Jaccard : {jaccard:.2f}")


# =========================
# IMPLEMENTASI JARAK (IRIS + LOOPING)
# =========================
print("\n=== DATA IRIS ===")

iris = load_iris()
data = iris.data
columns = iris.feature_names

df = pd.DataFrame(data, columns=columns)
print(df.head())

# normalisasi (opsional)
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data)

print("\n=== DATA SETELAH NORMALISASI ===")
print(pd.DataFrame(data_norm, columns=columns).head())

# contoh perhitungan
print("\n=== CONTOH JARAK DATA 0 & 1 ===")
print(f"Euclidean : {euclidean(data_norm[0], data_norm[1]):.4f}")
print(f"Manhattan : {cityblock(data_norm[0], data_norm[1]):.4f}")
print(f"Minkowski : {minkowski(data_norm[0], data_norm[1], 3):.4f}")

# looping 5 data pertama
print("\n=== LOOPING 5 DATA PERTAMA ===")
for i in range(5):
    for j in range(i+1, 5):
        print(f"\nData {i} vs {j}")
        print(f"Euclidean : {euclidean(data_norm[i], data_norm[j]):.4f}")
        print(f"Manhattan : {cityblock(data_norm[i], data_norm[j]):.4f}")
        print(f"Minkowski : {minkowski(data_norm[i], data_norm[j], 3):.4f}")


# =========================
# NOMOR 3 (COSINE SIMILARITY)
# =========================
print("\n=== NOMOR 3 ===")

docs = [
    "data science is fun",
    "data mining is fun",
    "machine learning is cool"
]

cv = CountVectorizer()
X_vec = cv.fit_transform(docs)

print("\nVocabulary:")
print(cv.get_feature_names_out())

print("\nBoW Matrix:")
print(X_vec.toarray())

sim_matrix = cosine_similarity(X_vec)

print("\nCosine Similarity:")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        print(f"D{i+1} vs D{j+1} : {sim_matrix[i][j]:.4f}")