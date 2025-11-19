# 🧠 Unsupervised Spatial Pattern Recognition (CS:GO Case Study)

> **Pipeline de Engenharia de Dados e Deep Learning para detecção automática de estratégias multi-agente utilizando Autoencoders Convolucionais e DBSCAN.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-DBSCAN-yellow)
![Type](https://img.shields.io/badge/Type-Computer_Vision-purple)

## 💼 Visão Geral

Este projeto implementa uma arquitetura de **Aprendizado de Máquina Não Supervisionado** para processar e classificar dados espaciais complexos. Utilizando um dataset de partidas profissionais de E-sports (CS:GO), o sistema é capaz de identificar padrões de posicionamento de equipe sem a necessidade de rotulagem manual prévia.

A solução combina técnicas de **Visão Computacional** para tratamento de coordenadas e **Redução de Dimensionalidade Não Linear** para agrupar comportamentos táticos similares e correlacioná-los com métricas de sucesso (Win Rate).

### 🛠️ Metodologia Técnica

O fluxo de trabalho foi desenhado para transformar dados brutos de telemetria em inteligência acionável:

1.  **Engenharia de Features Espaciais:** Conversão de logs de coordenadas vetoriais em representações matriciais (mapas de densidade 64x64), permitindo o uso de redes neurais convolucionais.
2.  **Compressão de Dados (Deep Learning):** Desenvolvimento de um **Autoencoder Convolucional (CAE)** para aprender a representação latente das táticas. O modelo comprime a entrada (4096 dimensões) em um vetor denso (64 dimensões), preservando a topologia essencial da formação.
3.  **Clusterização Baseada em Densidade:** Aplicação do algoritmo **DBSCAN** sobre o espaço latente. Diferente do K-Means, esta abordagem isola o "ruído" (rodadas atípicas) e consolida apenas as estratégias consistentes ("Táticas Puras").
4.  **Análise de Performance:** Cruzamento dos clusters identificados com o label de vitória (`Y_winner`) para gerar estatísticas de eficácia.

## 📊 Resultados Obtidos

* **Extração de Padrões:** O algoritmo segregou com êxito movimentações aleatórias de táticas coordenadas no mapa *de_mirage*.
* **Rankeamento de Eficácia:** Identificação automática de estratégias de alta performance. O **Cluster 36**, por exemplo, demonstrou uma taxa de conversão de vitória de **85.7%**, validando a relevância do padrão encontrado.
* **Visualização:** Plotagem dos centroides dos clusters sobre o mapa, permitindo a interpretação humana das estratégias descobertas pela máquina.

## 💻 Tecnologias

* **Linguagem:** Python
* **Redes Neurais:** TensorFlow / Keras (Camadas Conv2D, Conv2DTranspose, Dense)
* **Machine Learning Clássico:** Scikit-Learn (DBSCAN, PCA, t-SNE, NearestNeighbors)
* **Manipulação de Dados:** NumPy, Pandas, JSON, LZMA
* **Visualização:** Matplotlib

## 🚀 Execução do Projeto

### 1. Dependências
```bash
pip install numpy matplotlib tensorflow scikit-learn pandas
