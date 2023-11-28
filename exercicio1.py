import numpy as np
from sklearn.cluster import DBSCAN

# Dados fornecidos
area = np.array([120, 145, 80, 160, 200, 90, 110, 130, 180, 160])
valor = np.array([300, 450, 550, 600, 350, 420, 550, 780, 360, 575])
dist_praia = np.array([15, 15, 8, 25, 12, 15, 22, 8, 5, 14])

# Criando o array de dados
dados = np.column_stack((area, valor, dist_praia))

# Aplicando o algoritmo DBSCAN
epsilon = 15  # Defina a distância máxima entre os pontos
min_samples = 3  # Número mínimo de amostras em um cluster
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(dados)

# Obtendo os rótulos dos clusters
labels = dbscan.labels_

# Identificando os grupos únicos
unique_clusters = np.unique(labels)

# Analisando o valor médio dos imóveis em cada grupo
for cluster in unique_clusters:
    if cluster != -1:  # Ignora o cluster de ruído (-1)
        cluster_points = dados[labels == cluster]
        mean_value = np.mean(cluster_points[:, 1])  # Calcula o valor médio
        print(f"Grupo {cluster}: Valor médio dos imóveis = {mean_value:.2f}")
    else:
        print("Ruído (sem cluster)")


