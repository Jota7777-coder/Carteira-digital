import yfinance as yf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ["7201.T", "HMC", "TM"]

dados = yf.download(
    tickers, 
    start="2021-01-01", 
    end="2025-01-01", 
    interval="1mo"
)["Close"]


dividendos = {}
for t in tickers:
    
    dividendos[t] = yf.Ticker(t).dividends.resample("ME").sum()

dividendos_df = pd.DataFrame(dividendos)


retornos = pd.DataFrame()

for t in tickers:
    preco = dados[t]
    
    div = dividendos_df[t].reindex(preco.index).fillna(0)  
    
    
    retorno = ((preco - preco.shift(1)) + div) / preco.shift(1)
    retornos[t] = retorno.dropna()


def media_manual(vetor):
    return sum(vetor) / len(vetor)

def variancia_manual(vetor):
    m = media_manual(vetor)
    return sum((x - m)**2 for x in vetor) / (len(vetor) - 1)

def desvio_manual(vetor):
    return math.sqrt(variancia_manual(vetor))

def covariancia_manual(v1, v2):
    m1, m2 = media_manual(v1), media_manual(v2)
    return sum((x - m1)*(y - m2) for x, y in zip(v1, v2)) / (len(v1) - 1)

def correlacao_manual(v1, v2):
    return covariancia_manual(v1, v2) / (desvio_manual(v1)*desvio_manual(v2))

def coef_variacional(vetor):
    return desvio_manual(vetor) / media_manual(vetor)

resultados = {}

for col in retornos.columns:
    serie = retornos[col].dropna().tolist()
    resultados[col] = {
        "Retorno Médio Mensal (%)": media_manual(serie) * 100,
        "Variância Mensal": variancia_manual(serie),
        "Desvio-padrão Mensal (%)": desvio_manual(serie) * 100,
        "Coef. Variação (%)": coef_variacional(serie) * 100
    }

analise = pd.DataFrame(resultados).T
print("="*50)
print("Estatísticas individuais (2021-2025) — incluindo dividendos")
print("="*50)
print(analise)


print("\n" + "="*50)
print("Correlações 2 a 2")
print("="*50)
for i in range(len(tickers)):
    for j in range(i+1, len(tickers)):
        serie1 = retornos[tickers[i]].dropna().tolist()
        serie2 = retornos[tickers[j]].dropna().tolist()
        rho = correlacao_manual(serie1, serie2)
        print(f"Correlação {tickers[i]} x {tickers[j]} = {rho:.4f}")


rho12 = correlacao_manual(retornos[tickers[0]].dropna().tolist(), retornos[tickers[1]].dropna().tolist())
rho13 = correlacao_manual(retornos[tickers[0]].dropna().tolist(), retornos[tickers[2]].dropna().tolist())
rho23 = correlacao_manual(retornos[tickers[1]].dropna().tolist(), retornos[tickers[2]].dropna().tolist())
rho123 = (rho12 + rho13 + rho23) / 3
print(f"\nCorrelação média entre os 3 ativos = {rho123:.4f}")


capital_inicial = 100000
pesos = np.array([0.3, 0.3, 0.4])  
medias = np.array([media_manual(retornos[col].dropna().tolist()) for col in retornos.columns])
retorno_port = np.dot(pesos, medias)
retorno_reais = capital_inicial * retorno_port

print("\n" + "="*50)
print("Retorno Esperado da Carteira")
print("="*50)
print(f"Retorno esperado da carteira = {retorno_port*100:.2f}% ao mês")
print(f"Retorno esperado sobre R$ {capital_inicial:,.2f} = R$ {retorno_reais:,.2f} ao mês")


cov_matrix = retornos.cov()


var_port = np.dot(pesos.T, np.dot(cov_matrix, pesos))


dp_port = math.sqrt(var_port)

print("\n" + "="*50)
print("Risco da Carteira")
print("="*50)
print(f"Variância mensal da carteira = {var_port:.6f}")
print(f"Desvio-padrão (volatilidade) da carteira = {dp_port*100:.2f}% ao mês")


selic_aa = 0.10
rf_mes = (1 + selic_aa)**(1/12) - 1


sharpe = (retorno_port - rf_mes) / dp_port

print("\n" + "="*50)
print("Índice Sharpe")
print("="*50)
print(f"Taxa livre de risco (Selic) mensal = {rf_mes*100:.4f}%")
print(f"Índice Sharpe da carteira = {sharpe:.4f}")


if sharpe < 0:
    interpretacao = "O portfólio teve um retorno inferior ao ativo livre de risco."
elif 0 <= sharpe < 1:
    interpretacao = "O retorno do portfólio foi maior que o ativo livre de risco, mas o risco pode ser considerado alto para o retorno obtido (relação risco-retorno não é ideal)."
else:  
    interpretacao = "O retorno do portfólio é considerado bom para o nível de risco assumido."
    
print(f"Interpretação: {interpretacao}")


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]


plt.figure()
(dados / dados.iloc[0] * 100).plot()
plt.title("Evolução Normalizada dos Preços de Fechamento (Base 100)")
plt.xlabel("Data")
plt.ylabel("Preço Normalizado (Base 100)")
plt.legend(title="Ativos")
plt.show()

plt.figure()
retornos.plot(kind='line', alpha=0.8)
plt.title("Retornos Mensais (incluindo Dividendos)")
plt.xlabel("Data")
plt.ylabel("Retorno")
plt.axhline(0, color='black', linestyle='--') 
plt.legend(title="Ativos")
plt.show()


plt.figure()
sns.heatmap(retornos.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação dos Retornos Mensais")
plt.show()