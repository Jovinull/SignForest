import pickle
import csv
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Carregar os dados
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divisão inicial dos dados
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Hiperparâmetros a explorar
param_grid = {
    "n_estimators": [100, 500, 1000],
    "max_depth": [None, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy", "log_loss"],
    "min_impurity_decrease": [0.0, 0.01, 0.1],
    "ccp_alpha": [0.0, 0.01, 0.1],
    "max_leaf_nodes": [None, 10, 20, 50]
}

# Lista para armazenar resultados
results = []

# Combinações de hiperparâmetros
param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_split"],
    param_grid["min_samples_leaf"],
    param_grid["max_features"],
    param_grid["bootstrap"],
    param_grid["criterion"],
    param_grid["min_impurity_decrease"],
    param_grid["ccp_alpha"],
    param_grid["max_leaf_nodes"]
))

# Total de combinações
total_combinations = len(param_combinations)
print(f"Testando {total_combinations} combinações de hiperparâmetros.")

# Avaliar cada combinação
for i, params in enumerate(param_combinations, start=1):
    (n_estimators, max_depth, min_samples_split, min_samples_leaf, 
     max_features, bootstrap, criterion, min_impurity_decrease, 
     ccp_alpha, max_leaf_nodes) = params

    # Criar o modelo com os hiperparâmetros atuais
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        max_leaf_nodes=max_leaf_nodes,
        random_state=42,
        n_jobs=-1  # Usar todos os threads disponíveis
    )

    # Realizar validação cruzada
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    # Média e desvio padrão das validações
    mean_score = scores.mean()
    std_score = scores.std()

    # Adicionar os resultados na lista
    results.append({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "criterion": criterion,
        "min_impurity_decrease": min_impurity_decrease,
        "ccp_alpha": ccp_alpha,
        "max_leaf_nodes": max_leaf_nodes,
        "mean_accuracy": mean_score,
        "std_accuracy": std_score
    })

    print(f"({i}/{total_combinations}) Testado: {params} -> Acurácia média: {mean_score:.4f}")

# Salvar os resultados em CSV
csv_filename = "hyperparameter_search_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "bootstrap", "criterion", "min_impurity_decrease",
        "ccp_alpha", "max_leaf_nodes", "mean_accuracy", "std_accuracy"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"Resultados salvos em '{csv_filename}'.")
