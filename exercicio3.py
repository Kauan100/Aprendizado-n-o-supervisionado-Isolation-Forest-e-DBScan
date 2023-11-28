from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dados de entrada
leucocitos = np.array([2000, 4000, 5000, 6500]).reshape(-1, 1)
plaquetas = np.array([100000, 20000, 80000, 145000]).reshape(-1, 1)
linfocitos = np.array([2.3, 4.5, 6.5, 4.4]).reshape(-1, 1)

# Juntando os dados em um array
dados = np.concatenate((leucocitos, plaquetas, linfocitos), axis=1)

# Padronizando os dados
scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(dados)

# Aplicando DBSCAN
epsilon = 0.5  # Escolha um valor adequado para epsilon
min_samples = 2  # Número mínimo de amostras em um cluster
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(dados_padronizados)

# Resultados
print("Resultados do DBSCAN:")
print("Labels dos clusters:", clusters)
