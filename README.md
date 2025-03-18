# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="https://raw.githubusercontent.com/lfusca/templateFiap/main/assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

## 👨‍🎓 Integrantes do Grupo

- RM559800 - [Jonas Felipe dos Santos Lima](https://www.linkedin.com/in/jonas-felipe-dos-santos-lima-b2346811b/)
- RM560173 - [Gabriel Ribeiro](https://www.linkedin.com/in/ribeirogab/)
- RM559926 - [Marcos Trazzini](https://www.linkedin.com/in/mstrazzini/)
- RM560461 - [Jose Antonio Correa Junior](https://www.linkedin.com/in/jacorrea/)
- RM559645 - [Edimilson Ribeiro](https://www.linkedin.com/in/edimilson-ribeiro/)

## 👩‍🏫 Professores

### Tutor(a)

- [Lucas Gomes Moreira](https://www.linkedin.com/in/lucas-gomes-moreira-15a8452a/)

### Coordenador(a)

- [André Godoi](https://www.linkedin.com/in/profandregodoi/)

---

## 📌 Entregas do Projeto

O projeto consiste em duas entregas principais:

1. **Entrega 1 - Machine Learning**: Análise exploratória de dados, clusterização e modelagem preditiva de produtividade agrícola.
2. **Entrega 2 - Computação em Nuvem**: Estimativa de custos para hospedagem da solução na nuvem.

---

## 🛠 **Entrega 1 - Machine Learning**

### 📜 Descrição do Projeto

Este projeto tem como objetivo analisar dados agrícolas e prever a produtividade de culturas utilizando técnicas de Machine Learning. Através de análise exploratória de dados, clustering não supervisionado e modelos preditivos supervisionados, buscamos identificar padrões e fatores que influenciam a produtividade agrícola, fornecendo insights valiosos para otimização da produção.

### 🔍 Detalhes Técnicos

#### Dataset

O projeto utiliza o dataset `crop_yield.csv`, que contém dados sobre produtividade agrícola e diversos fatores ambientais e de cultivo que podem influenciar o rendimento das culturas, incluindo:

- Temperatura
- Umidade
- Precipitação
- pH do solo
- Níveis de nutrientes (N, P, K)
- Tipo de cultura
- Produtividade (yield)

#### Análise Exploratória de Dados (EDA)

A análise exploratória inclui:

- Estatísticas descritivas das variáveis (média, mediana, desvio padrão, valores mínimos e máximos)
- Verificação e tratamento de valores ausentes
- Visualização da distribuição das variáveis através de histogramas e boxplots
- Matriz de correlação para identificar relações entre variáveis
- Gráficos de dispersão para detectar tendências

#### Clustering (Aprendizado Não Supervisionado)

Implementamos o algoritmo K-Means para identificar padrões naturais nos dados de produtividade agrícola:

- Determinação automática do número ideal de clusters usando o Método do Cotovelo e Score de Silhueta
- Visualização dos clusters em gráficos 2D e 3D
- Análise das características de cada cluster
- Identificação e análise de outliers

#### Modelagem Preditiva (Aprendizado Supervisionado)

Desenvolvemos cinco modelos de Machine Learning para prever a produtividade das culturas:

1. **Regressão Linear**: Modelo base para estabelecer uma referência de desempenho
2. **Árvore de Decisão**: Para capturar relações não-lineares nos dados
3. **Random Forest**: Ensemble de árvores de decisão para maior robustez
4. **Gradient Boosting (XGBoost)**: Algoritmo avançado para maximizar a precisão
5. **Rede Neural (MLP Regressor)**: Para capturar padrões complexos nos dados

Cada modelo é avaliado usando as seguintes métricas:

- R² Score (coeficiente de determinação)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

### 🎥 Demonstração no YouTube

WIP

---

### 📁 Estrutura de Pastas

- **`data/`**: Contém o dataset utilizado no projeto
  - `crop_yield.csv`: Dataset principal com dados agrícolas

- **`notebooks/`**: Jupyter Notebooks com análises e modelos
  - `crop_yield_analysis.ipynb`: Notebook principal com todas as análises
  - `crop_yield_analysis.py`: Versão em script Python do notebook

- **`images/`**: Visualizações e gráficos gerados pelas análises
  - Histogramas, boxplots, matrizes de correlação
  - Visualizações de clusters
  - Gráficos de importância de features
  - Comparações de desempenho dos modelos

---

### 🔧 Como Executar

#### Configuração Inicial

1. Clone este repositório:

   ```bash
   git clone https://github.com/FIAP-IA2024/farm-tech-solutions-v5.git
   cd farm-tech-solutions-v5
   ```

2. Crie e ative um ambiente virtual Python:

   - **Linux/macOS:**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - **Windows:**

     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. Instale as dependências Python:

   ```bash
   pip install -r requirements.txt
   ```

#### Executar o Jupyter Notebook

1. Inicie o servidor Jupyter:

   ```bash
   jupyter notebook
   ```

2. No navegador, navegue até a pasta `notebooks` e abra o arquivo `crop_yield_analysis.ipynb`

3. Execute todas as células do notebook para reproduzir as análises e visualizações

---

### 💻 Tecnologias Utilizadas

- **Linguagens de Programação:**
  - Python 3.x

- **Bibliotecas e Frameworks:**
  - **Pandas**: Manipulação e análise de dados
  - **NumPy**: Computação numérica
  - **Matplotlib e Seaborn**: Visualização de dados
  - **Scikit-learn**: Implementação de algoritmos de Machine Learning
  - **XGBoost**: Implementação de Gradient Boosting
  - **Jupyter**: Ambiente interativo para desenvolvimento e documentação

- **Ferramentas:**
  - **Git**: Controle de versão
  - **GitHub**: Hospedagem do repositório

---

## ☁️ **Entrega 2 - Computação em Nuvem**

Com a finalidade de estimar os custos de hospedagem em nuvem, fizemos um estudo utilizando a [calculadora oficial de custos da AWS](https://calculator.aws), onde criamos uma estimativa de custos para uma máquina virtual EC2 `t4g.small` com as seguintes configurações: Instância compartilhada, com 100% de utilização sob demanda, 1vCPU, 2GB de RAM, até 5Gbit de rede e 50GB de armazenamento EBS.

Esta máquina virtual será responsável por rodar a aplicação responsável pela API que receberá os dados dos sensores, e também rodará o modelo de Machine Learning.

Para uma comparação de custos mais abrangente, criamos uma simulação dessa mesma configuração em duas regiões diferentes: US East (N. Virginia) e South America (São Paulo), e os resultados foram os seguintes:

- **Custo mensal/anual**:
  - **US East (N. Virginia)**: 16.26 / 195.12 USD
  - **South America (Sao Paulo)**: 27.16 / 325.92 USD

Com base no escopo dessa comparação, podemos concluir que os custos de hospedagem nos EUA são 40% menores do que no Brazil. Porém, existem outros pontos importantes que devem ser considerados ao decidir qual região hospedar a aplicação que, no nosso caso, são os seguintes:

 - **Latência de rede**: ...
 - **Restrições legais**: ...

---

## 📋 Licença

Este projeto segue o modelo de licença da FIAP e está licenciado sob **Attribution 4.0 International**. Para mais informações, consulte o [MODELO GIT FIAP](https://github.com/agodoi/template).
