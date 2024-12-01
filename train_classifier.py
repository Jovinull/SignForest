import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Carregar os dados
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divisão em treino e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Criação do modelo Random Forest com hiperparâmetros explícitos
model = RandomForestClassifier(
    n_estimators=100,          # Número de árvores na floresta
    max_depth=None,            # Profundidade máxima das árvores (None significa sem limite)
    min_samples_split=2,       # Número mínimo de amostras necessárias para dividir um nó
    min_samples_leaf=1,        # Número mínimo de amostras em cada folha
    max_features='sqrt',       # Número de características consideradas para cada divisão (raiz quadrada do total)
    bootstrap=True,            # Se as amostras devem ser reamostradas para cada árvore
    criterion='gini',          # Critério para medir a qualidade de uma divisão ('gini' ou 'entropy')
    random_state=42            # Semente para reprodutibilidade
)

# Treinamento do modelo
model.fit(x_train, y_train)

# Predições no conjunto de teste
y_predict = model.predict(x_test)

# Métricas de desempenho
accuracy = accuracy_score(y_test, y_predict)
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')
print("\nRelatório de classificação:")
print(classification_report(y_test, y_predict))

# Salvar o modelo treinado
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
