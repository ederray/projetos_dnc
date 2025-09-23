# Análise de Dados do Airbnb no Rio de Janeiro

Este projeto realiza a **limpeza, transformação e análise de dados** de anúncios do Airbnb no Rio de Janeiro. O objetivo é transformar um conjunto de dados brutos em uma base de dados limpa, estruturada e pronta para análise exploratória e futuras modelagens preditivas.

-----

### 🎯 Desafio e Contexto

A análise de dados é focada em responder perguntas-chave sobre o mercado de aluguel por temporada no Rio de Janeiro. A partir de um conjunto de dados brutos, o projeto investiga se o **preço, a localização ou as avaliações** são os fatores mais decisivos na escolha de uma acomodação.

O desafio abrange as seguintes etapas:

  * **Carregamento e união** dos dados de anúncios (`listings_cleaned.csv`) e avaliações (`reviews.csv`).
  * **Limpeza e tratamento** de valores ausentes e inconsistências.
  * **Identificação e remoção de outliers** para evitar distorções na análise.
  * **Transformação** de dados categóricos e numéricos para aprimorar a qualidade do dataset.
  * **Análise Exploratória de Dados (EDA)** para descobrir insights e padrões úteis.

-----

### 📂 Estrutura do Projeto

O projeto segue uma estrutura de diretórios organizada, baseada em práticas recomendadas para projetos de ciência de dados.

```
.
├── data/                        # Jupyter Notebooks para exploração e análise
│   ├── raw/                     # Dados brutos
│   ├── processed/               # Dados pré-processados
├── notebooks/                   # Jupyter Notebooks para exploração e análise
│   ├── .eda.ipynb               # Análise Exploratória de Dados
├── references/                  # Imagens sobre metadados
├── reports/                     # Relatórios gerados
│   └── figures/                 # Gráficos e visualizações salvas
├── src/                         # Código-fonte do projeto
│   ├── config/                  # Configurações e logging
│   ├── data/                    # Módulos de carregamento, limpeza e processamento de dados
│   ├── features/                # Engenharia e transformação de features
│   ├── modeling/                # Futura etapa de modelagem preditiva
│       └── run_train.py         # Script  etapa de modelagem preditiva
│   ├── utils/                   # Funções utilitárias e de estatística
│   └── visualize/               # Funções para visualização de dados
│       ├── interactive_plot.py  # Gráficos interativos (Plotly, Ipywidgets)
│       └── static_plot.py       # Gráficos estáticos (Matplotlib, Seaborn)
├── .env                         # Variáveis de ambiente
├── .gitignore                   # Arquivos e diretórios ignorados pelo Git
├── requirements.txt             # Dependências do projeto (gerado a partir do poetry)
├── pyproject.toml               # Configurações do projeto, incluindo dependências e ferramentas
├── README.md                    # Descrição do projeto (este arquivo)
└── Makefile                     # Automação de tarefas
```

-----

### 🛠️ Tecnologias e Ferramentas

Este projeto foi desenvolvido utilizando as seguintes tecnologias e bibliotecas:

  * **Linguagem:** Python 3.11
  * **Gerenciamento de Pacotes:** `poetry`
  * **Análise de Dados:** `pandas`, `numpy`, `sidetable`
  * **Visualização:** `matplotlib`, `seaborn`, `plotly`
  * **Limpeza de Dados:** `missingno`
  * **Estatística:** `scipy`, `statsmodels` (usado para testes de hipótese como **teste t** e **teste de Tukey** para validação estatística)
  * **Outras Ferramenta:** `ipywidgets` (para gráficos interativos), `scikit-learn` (para pré-processamento)

Para garantir a qualidade e a consistência do código, as seguintes ferramentas de linting e formatação são usadas:

  * **`black`**: Formatador de código.
  * **`isort`**: Organização de imports.
  * **`flake8`**: Validação de estilo e erros.

-----

### ⚙️ Como Usar

Para configurar e rodar o projeto localmente, siga os passos abaixo:

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Instale as dependências:**
    Você pode instalar as dependências de duas formas:

      * **Via Poetry (recomendado):** Se você tem o Poetry instalado, execute:
        ```bash
        poetry install
        ```
      * **Via requirements.txt:** Se preferir usar o `pip`, execute:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Execute o pipeline:**
    O `Makefile` automatiza o processo de execução do pipeline de dados. Use o comando `make train` para rodar o arquivo `run_train.py` e iniciar o processamento.

    ```bash
    make train
    ```
