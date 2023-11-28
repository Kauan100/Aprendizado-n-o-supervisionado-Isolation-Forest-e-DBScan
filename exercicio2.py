import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Dados
corrente_eletrica = np.array([5, 10, 14, 2, 1.5, 6]).reshape(-1, 1)
tempo = np.array([1, 2, 4, 6, 7, 10]).reshape(-1, 1)

# Criando o modelo Isolation Forest
modelo = IsolationForest(contamination=0.1)  # contamination é a proporção de anomalias esperadas

# Treinando o modelo com os dados de corrente elétrica
modelo.fit(corrente_eletrica)

# Predizendo anomalias nos dados de corrente elétrica
anomalias = modelo.predict(corrente_eletrica)

# Plotando os dados
plt.figure(figsize=(10, 6))
plt.scatter(tempo, corrente_eletrica, c='b', label='Dados de corrente elétrica')
plt.scatter(tempo[np.where(anomalias == -1)], corrente_eletrica[np.where(anomalias == -1)], c='r', label='Anomalias')
plt.xlabel('Tempo')
plt.ylabel('Corrente elétrica')
plt.title('Detecção de Anomalias com Isolation Forest')
plt.legend()
plt.show()
