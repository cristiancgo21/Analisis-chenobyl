import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE

# ========================
# 1. CARGA Y LIMPIEZA
# ========================
df = pd.read_csv("Chernobyl_Chemical_Radiation.csv")

print("\n=== Info inicial del dataset ===")
print(df.info())
print("\nPrimeras filas:")
print(df.head())

# Valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Imputar nulos: medianas para numéricos, moda para categóricos
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Intentar convertir texto numérico
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col].str.replace(',','.'), errors='ignore')
        except:
            pass

# Eliminar duplicados
df = df.drop_duplicates()

print("\n=== Dataset limpio ===")
print(df.head())

# ========================
# 2. ESCALADO DE VARIABLES
# ========================
numeric_df = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)


# ========================
# 3. K-MEANS (AUTOMÁTICO)
# ========================

sil_scores = []
inertias = []
k_values = range(2, 10)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

# Gráfico método del codo
plt.figure()
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Número de clusters k')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.show()

# Gráfico coeficiente de silueta
plt.figure()
plt.plot(k_values, sil_scores, marker='o')
plt.xlabel('k')
plt.ylabel('Coeficiente de silueta')
plt.title('Coeficiente de silueta vs k')
plt.show()

# Elegir automáticamente el k con mayor silueta
best_k = k_values[np.argmax(sil_scores)]
print(f"\nMejor k según coeficiente de silueta: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# ========================
# 4. CLÚSTER JERÁRQUICO
# ========================
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10,7))
dendrogram(linked)
plt.title("Dendrograma jerárquico")
plt.xlabel("Muestras")
plt.ylabel("Distancia")
plt.show()

# Extraer clusters (ejemplo con 3)
clusters_hier = fcluster(linked, 3, criterion='maxclust')
df['Cluster_Jerarquico'] = clusters_hier

# ========================
# 5. DBSCAN
# ========================

# Estimar eps con gráfico k-dist (4º vecino)
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:,3])
plt.figure()
plt.plot(distances)
plt.ylabel('Distancia al 4º vecino')
plt.xlabel('Puntos ordenados')
plt.title('Gráfico k-dist para estimar eps')
plt.show()

# DBSCAN (ajusta eps según gráfico)
db = DBSCAN(eps=1.0, min_samples=4)
labels_db = db.fit_predict(X_scaled)
df['Cluster_DBSCAN'] = labels_db

# ========================
# 6. RESULTADOS
# ========================
print("\n=== Dataset con clusters ===")
print(df.head())

# Guardar CSV limpio con clusters
df.to_csv("Chernobyl_Chemical_Radiation_clusters.csv", index=False)

# ========================
# 7. PCA para visualización
# ========================

# Ajustar PCA con varianza acumulada ≥90%
pca_full = PCA()
pca_full.fit(X_scaled)
var_acum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(var_acum >= 0.90) + 1
print(f"\nNúmero de componentes para ≥90% varianza: {n_components_90}")

# Reducir a n_components_90
pca = PCA(n_components=n_components_90)
X_pca = pca.fit_transform(X_scaled)

# Para visualización usamos las dos primeras componentes
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_KMeans'], cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters K-Means en espacio PCA')
plt.colorbar(label='Cluster')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_Jerarquico'], cmap='plasma')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Jerárquico en espacio PCA')
plt.colorbar(label='Cluster')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_DBSCAN'], cmap='coolwarm')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters DBSCAN en espacio PCA')
plt.colorbar(label='Cluster')
plt.show()

# ========================
# 8. t-SNE para visualización
# ========================

# Ajustar t-SNE (puedes variar perplexity=30 y random_state)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualizar K-Means en t-SNE
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=df['Cluster_KMeans'], cmap='viridis')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Clusters K-Means en espacio t-SNE')
plt.colorbar(label='Cluster')
plt.show()

# Visualizar Clúster Jerárquico en t-SNE
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=df['Cluster_Jerarquico'], cmap='plasma')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Clusters Jerárquico en espacio t-SNE')
plt.colorbar(label='Cluster')
plt.show()

# Visualizar DBSCAN en t-SNE
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=df['Cluster_DBSCAN'], cmap='coolwarm')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Clusters DBSCAN en espacio t-SNE')
plt.colorbar(label='Cluster')
plt.show()