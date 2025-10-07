# Importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Ler o arquivo
df = pd.read_csv("ecommerce_preparados.csv")

# 2.Analise Inicial
print(df.head())
print(df.info())
print(df.describe())

# 3. Grafico de Histograma (Preço)
plt.hist(df["Preço"], bins=18, color='blue', edgecolor='black')
plt.title("Distribuição dos Preços")
plt.xlabel("Preço")
plt.ylabel("Frequência")
plt.show()

print(df[['Preço', 'Qtd_Vendidos']].dtypes)

# 4. Grafico de dispersão (Preço x Qtd_Vendidos)
plt.scatter(df['Preço'], df['Qtd_Vendidos'], alpha=0.6)
plt.title("Dispersão: Preço x Qtd_Vendidos")
plt.xlabel("Preço")
plt.ylabel("Qtd_Vendidos")
plt.show()

# 5. Mapa de Calor (correlação numericas)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Mapa de Calor - Correlação entre variaveis numericas")
plt.show()

# 6. Grafico de barras (Gênero)
df['Gênero'].value_counts().plot(kind='bar', color='red')
plt.title("Distribuição dos Gênero")
plt.xlabel("Gênero")
plt.ylabel("Frequência")
plt.show()

# 7. Grafico de pizza (Material do produto)
df['Material'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6,6))
plt.title(" Material  do Produto")
plt.ylabel("")
plt.show()

# 8. Grafico de densidade (Preço)
sns.kdeplot(df['Preço'], shade=True, color='orange')
plt.title("Distribuição de Densidade dos Preços ")
plt.xlabel('Preço')
plt.show()

# Garantir que ambas as colunas são numericas
df['Preço'] = pd.to_numeric(df['Preço'], errors="coerce")
df['Desconto'] = pd.to_numeric(df['Desconto'], errors='coerce')

# Remover linhas com valores nulos nessas colunas
df = df.dropna(subset=['Preço', 'Desconto'])

#9. Grafico de regressão (Preço x Desconto)
sns.regplot(x="Preço", y="Desconto", data=df, scatter_kws={"alpha":0.5})
plt.title("Regressão= Preço x Desconto")
plt.xlabel("Preço")
plt.ylabel("Desconto")
plt.show()