import pandas as pd
import yfinance as yf
import numpy as np

def calcular_indicadores_financeiros(tickers, anos):
    """
    Baixa dados financeiros e calcula indicadores para uma lista de empresas e anos.

    Args:
        tickers (list): Lista de tickers das empresas (ex: ["TM", "HMC"]).
        anos (list): Lista de anos para análise (ex: [2021, 2022, 2023, 2024]).

    Returns:
        pandas.DataFrame: Tabela pivotada com os resultados dos indicadores.
    """
    
    resultados_finais = []
    
    
    anos_para_buscar = sorted(list(set(anos + [min(anos) - 1])))

    for ticker in tickers:
        print(f"\nBuscando dados para {ticker}...")
        
        try:
            empresa = yf.Ticker(ticker)
            
           
            balanco = empresa.balance_sheet
            dre = empresa.income_stmt
            
           
            if balanco.empty or dre.empty:
                print(f"-> ⚠️ Aviso: Não foi possível obter dados financeiros para {ticker}. Pulando para o próximo.")
                continue

            
            balanco_t = balanco.T
            dre_t = dre.T
            
            
            balanco_t.index = pd.to_datetime(balanco_t.index).year
            dre_t.index = pd.to_datetime(dre_t.index).year

            for ano in anos:
                ano_anterior = ano - 1
                
             
                try:
                    # Dados do Balanço Patrimonial
                    ativo_circulante = balanco_t.loc[ano, "Current Assets"]
                    estoques = balanco_t.loc[ano, "Inventory"]
                    passivo_circulante = balanco_t.loc[ano, "Current Liabilities"]
                    contas_a_pagar = balanco_t.loc[ano, "Accounts Payable"]
                    ativo_total_atual = balanco_t.loc[ano, "Total Assets"]
                    ativo_total_anterior = balanco_t.loc[ano_anterior, "Total Assets"]
                    
                    # Dados da DRE
                    receita_liquida = dre_t.loc[ano, "Total Revenue"]
                    custo_produtos_vendidos = dre_t.loc[ano, "Cost Of Revenue"]

                 
                    
                    
                    liquidez_corrente = ativo_circulante / passivo_circulante if passivo_circulante else 0
                    
                   
                    liquidez_seca = (ativo_circulante - estoques) / passivo_circulante if passivo_circulante else 0
                    
                   
                    media_ativo_total = (ativo_total_atual + ativo_total_anterior) / 2
                    giro_ativo = receita_liquida / media_ativo_total if media_ativo_total else 0
                    
                    
                    prazo_medio_pagamento = (contas_a_pagar / custo_produtos_vendidos) * 365 if custo_produtos_vendidos else 0
                    
                   
                    resultados_finais.append({
                        "Empresa": ticker,
                        "Ano": ano,
                        "Liquidez Corrente": liquidez_corrente,
                        "Liquidez Seca": liquidez_seca,
                        "Giro do Ativo": giro_ativo,
                        "Prazo Médio de Pagamento (dias)": prazo_medio_pagamento
                    })
                    
                except KeyError as e:
                    print(f"-> ⚠️  Aviso: Não foi possível encontrar o item '{e}' para {ticker} no ano de {ano}. Este ano será ignorado.")
                    continue
                    
        except Exception as e:
            print(f"-> ❌ Erro inesperado ao processar {ticker}: {e}")

    if not resultados_finais:
        print("\nNenhum dado foi processado. Encerrando.")
        return pd.DataFrame()

    
    df_final = pd.DataFrame(resultados_finais)
    tabela_pivotada = df_final.pivot(index='Ano', columns='Empresa')
    
    return tabela_pivotada


if __name__ == "__main__":
   
    tickers_carteira = ["TM", "HMC", "7201.T"] 
    
    anos_analise = [2021, 2022, 2023, 2024]

    
    tabela_resultados = calcular_indicadores_financeiros(tickers_carteira, anos_analise)
    
    if not tabela_resultados.empty:
        print("\n\n" + "="*80)
        print("Tabela Final de Indicadores Financeiros (2021-2024)")
        print("="*80)
        
        
        styled_table = tabela_resultados.style.format("{:.2f}").set_properties(**{'text-align': 'center'})
        
      
        try:
            from IPython.display import display
            display(styled_table)
        except ImportError:
            print(tabela_resultados.round(2))