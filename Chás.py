#!/usr/bin/env python
# coding: utf-8

# # Análise do conjunto de dados relativos à amostras de Chá. Dados coletados por infravermelho próximo (NIR)
# 
# #### 100 amostras diferentes; 20 amostras de chás argentino verde (1); 20 amostras de chás argentinos pretos (2); 20 amostras de chás brasileiros verdes(3); 20 amostras de chás brasileiros pretos (4); 20 amostras de chás asiáticos (5)

# In[2]:


# Importando bibliotecas 
import matplotlib.pyplot as plt # Gráficos
import pandas as pd # Biblioteca para tratar de conjunto de dados 
import scipy.io # Aqui eu utilizarei para pré-processamentos
import numpy as np # Manipulação de vetores e matrizes


# In[3]:


# LENDO O ARQUIVO .MAT 
caminho = r'C:\Users\felip\OneDrive - Universidade Federal de Pernambuco\Quimiometria\Quimiometria 2\Dados\2 - Análise exploratória\Chás\exemplo_1.mat'
dados_cha = scipy.io.loadmat(caminho) # Aqu


# In[4]:


# Dando uma olhada nos dados importados
print(dados_cha) # Imprime alguma coisa na tela, neste caso os dados importados


# In[5]:


# Importamos um arquivo do tipo DICIONÁRIO, onde temos 'X' que são os dados espectrais, 
#'xaxis' que é um vetor (array) com os comprimentos de onda e 
#'Class' que é um vetor contendo as classes de cada amostra.

# Vamos criar um conjunto de dados (DataFrame) com os dados contidos em 'X'

df_cha = pd.DataFrame(dados_cha['X'])


# In[6]:


# Conferindo se o Dataframe foi criado corretamente 
df_cha.head(1) # Head mostra uma espécie de cabeçalho, por padrão, retorna 5 linhas, porém especifiquei que queria apenas uma.


# In[7]:


# Corrigindo os labels
df_cha = df_cha.set_axis(list(range(1001, 2501)), axis=1, inplace=False) # Modifiquei o DF para ficar com os comprimentos de onda certinhos
df_cha.head(1) 


# In[8]:


# Dando uma olhada nos espectros brutos 
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df_cha.T) # Plotar um gráfico de linhas com os dados especificados entre parênteses e transpostos. 
ax.set_xlabel('Comprimento de onda (nm)', fontsize=12)
ax.set_ylabel('Absorbância (u.a.)', fontsize=12)
plt.show()


# In[9]:


# Percebe-se uma variação de linha de base e espalhamento, tentativa de correção com 1ª Derivada
from scipy.signal import savgol_filter # Função para derivação ou alisamento com filtro de Savitzky-Golay 
df_dx = savgol_filter(df_cha, 75, 2, deriv=1) # Criei uma nova variável com os dados filtrados, aqui escolhi os dados que eu queria alisar, a janela, o grau do polinômio e a derivaida


# In[10]:


# Espectro derivado
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df_dx.T)
ax.set_xlabel('Comprimento de onda (nm)', fontsize=12)
ax.set_ylabel('Absorbância (u.a.)', fontsize=12)
plt.show()


# In[11]:


# Importando pré-processamento para centrar na média e a PCA
from sklearn.preprocessing import scale # Função para Centrar na média
from sklearn.decomposition import PCA # Função PCA


# In[12]:


# Centrando os dados na média
df_dx_center = scale(df_dx, with_std=False) #with_std = False, pois não quero que ele traga para variância unitária. 
df_dx_center = df_dx_center.set_axis(list(range(1001, 2501)), axis=1, inplace=False) 


# In[13]:


# Plotando o gráfico com os dados centrados na média
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df_dx_center.T)
ax.set_xlabel('Comprimento de onda (nm)', fontsize=12)
ax.set_ylabel('Absorbância (u.a.)', fontsize=12)
plt.show()


# In[15]:


# Realizando a PCA
pca = PCA(n_components=4) # Calcular no máximo 6 componentes
pca_model = pca.fit(df_dx_center)
PCA_chas = pd.DataFrame(pca_model.transform(df_dx_center), columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])


# In[16]:


PCA_chas # Só quis conferir como ficaram os dados


# In[17]:


pca_var = pca_model.explained_variance_ratio_ # Percentual de variância explicada
pca_var


# In[18]:


# Explorando os gráficos de variância, Variância explicada e acumulada
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(pca_var)
ax.set_xlabel('Nº de Componentes Principais')
ax.set_ylabel('% de Variância Explicada')
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(10,10))
plt.plot(list(range(1,5)), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(list(range(1,5)))
plt.xlabel('Nº de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.show()


# In[20]:


PCA_chas['Tipo'] = dados_cha['Class'] # Aqui eu adicionei as classes ao conjunto de dados espectrais
# Poderia ser adicionado antes também, mas achei mais fácil trabalhar desse modo


# In[64]:


# Atribuindo variáveis aos autovalores e autovetores
autovetores = pca.components_
autovalores = pca.explained_variance_
autovetores.shape


# In[47]:


# Atribuindo variáveis para Scores e Loadings PC1 x PC2

PC1PC2_scores = PCA_chas.iloc[:, 0:2]
PC1PC2_loadings = autovetores[0:2, :]


# In[69]:


# SCORES e LOADINGS PC1 x PC2
import seaborn as sns # Importando outra biblioteca para gerar gráficos
fig, ax = plt.subplots(1, 2, figsize=(20,10))
colors = {1:'red', 2:'black', 3:'blue', 4:'green', 5:'yellow'} # Criei um dicionário com cores para cada classe 
plt.figure(figsize=(10, 10))
sns.scatterplot(x='PCA1', y='PCA2', data=PCA_chas, hue='Tipo', palette=colors, ax=ax[0]) # Scores, PC1 x PC2 a ser plotado na esquerda
ax[1].plot(PC1PC2_loadings.T) # Loadings plotados na direita
plt.show()


# Algumas coisas podem ser mais simples de serem feitas no programa SPYDER pois temos como observar como estão as variáveis criadas, figuras, dataframes em um modelo parecido com o MATLAB. Resolvi fazer aqui pra ficar melhor a observação do passo-a-paso.
# 
# Preciso melhorar várias questões e adicionar a interpretação de resíduos da PCA. 

# In[ ]:




