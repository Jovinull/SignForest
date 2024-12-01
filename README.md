# SignForest - Reconhecimento de Sinais em Libras com Inteligência Artificial

Este projeto utiliza técnicas de visão computacional e aprendizado de máquina para criar um sistema capaz de reconhecer sinais em Libras (Língua Brasileira de Sinais). A base do sistema são landmarks de mãos capturados com MediaPipe e um modelo treinado para realizar as previsões.

---

## Estrutura do Projeto

### 1. `create_dataset.py`
- **Descrição:**
  Este script processa as imagens localizadas na pasta `data/` para extrair landmarks das mãos utilizando a biblioteca MediaPipe. Os landmarks são normalizados e organizados em um formato adequado para treinamento de modelos de aprendizado de máquina.
- **Saída:**
  - Um arquivo `data.pickle` contendo:
    - `data`: Lista com as características extraídas de cada imagem.
    - `labels`: Rótulos associados às imagens.

---

### 2. `train_classifier.py`
- **Descrição:**
  Este script treina um classificador Random Forest com os dados do arquivo `data.pickle`.
- **Entrada:**
  - Arquivo `data.pickle`.
- **Saída:**
  - Modelo treinado salvo como `model.p`.
- **Funcionalidades:**
  - Divide os dados em conjuntos de treino e teste.
  - Avalia o modelo usando métricas como acurácia e relatório de classificação.

---

### 3. `hyperparameter_search.py`
- **Descrição:**
  Realiza uma busca exaustiva pelos melhores hiperparâmetros para o classificador Random Forest.
- **Entrada:**
  - Arquivo `data.pickle`.
- **Saída:**
  - Arquivo `hyperparameter_search_results.csv` contendo os resultados das combinações testadas.
- **Diferenciais:**
  - Explora várias combinações de parâmetros como número de árvores, profundidade máxima e critérios de divisão.
  - Utiliza validação cruzada para avaliar o desempenho de cada combinação.

---

### 4. `inference_classifier.py`
- **Descrição:**
  Realiza inferência em tempo real utilizando a câmera do computador para capturar imagens das mãos e prever o sinal correspondente.
- **Entrada:**
  - Modelo treinado salvo como `model.p`.
- **Funcionamento:**
  - Captura frames de vídeo da webcam.
  - Processa os landmarks das mãos com MediaPipe.
  - Utiliza o modelo Random Forest para prever o sinal correspondente.
- **Saída:**
  - Exibe o frame da webcam com a previsão do sinal e a área delimitada da mão.

---

## Como Rodar o Projeto

### **1. Preparação Inicial**
1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. **Baixe os arquivos necessários:**
   - Acesse a seção de [Releases](https://github.com/seu-usuario/seu-repositorio/releases) do repositório.
   - Faça o download da pasta `data/`.

3. **Crie um ambiente virtual Python:**
   ```bash
   python -m venv venv
   ```

4. **Ative o ambiente virtual:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
     Caso encontre o erro "execução de scripts está desabilitada no sistema", use:
     ```bash
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

5. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

### **2. Criação do Dataset**
- Para criar seu próprio dataset, execute:
  ```bash
  python create_dataset.py
  ```
- Isso processará as imagens em `data/` e gerará um novo arquivo `data.pickle`.

---

### **3. Treinamento do Modelo**
- Para treinar o modelo com os dados fornecidos ou gerados, execute:
  ```bash
  python train_classifier.py
  ```
- O modelo treinado será salvo como `model.p`.

---

### **4. Busca de Hiperparâmetros**
- Para explorar diferentes configurações de hiperparâmetros, execute:
  ```bash
  python hyperparameter_search.py
  ```
- Os resultados serão salvos em um arquivo `hyperparameter_search_results.csv`.

---

### **5. Realização de Inferências**
- Para rodar a inferência em tempo real com a webcam, execute:
  ```bash
  python inference_classifier.py
  ```
- Isso abrirá uma janela de vídeo com a previsão do sinal exibida na tela.

---

## Contribuições
Sinta-se à vontade para abrir issues ou enviar pull requests caso tenha sugestões ou melhorias para o projeto.

---

## Notas Importantes
- Certifique-se de que sua câmera esteja funcionando corretamente antes de rodar o script de inferência.
- Para melhores resultados, utilize um ambiente bem iluminado ao capturar imagens para o dataset ou realizar inferências.