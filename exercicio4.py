from sklearn.ensemble import IsolationForest
import numpy as np

# Dados de hemograma dos pacientes
leucocitos = np.array([2000, 4000, 5000, 6500]).reshape(-1, 1)
plaquetas = np.array([100000, 20000, 80000, 145000]).reshape(-1, 1)
linfocitos = np.array([2.3, 4.5, 6.5, 4.4]).reshape(-1, 1)

# Juntando os dados em uma matriz
hemograma = np.concatenate((leucocitos, plaquetas, linfocitos), axis=1)

# Criando e ajustando o modelo Isolation Forest
modelo = IsolationForest(random_state=42)
modelo.fit(hemograma)

# Identificando anomalias (outliers)
anomalias = modelo.predict(hemograma)

# Exibindo os resultados
for i, resultado in enumerate(anomalias):
    if resultado == -1:
        print(f"Paciente {i+1} apresenta anomalia no hemograma.")
