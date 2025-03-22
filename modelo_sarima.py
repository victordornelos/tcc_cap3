import warnings
import pandas as pd
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm  # Barra de progresso

def run_sarima_grid_search(
    file_path="df_sarima.xlsx",
    output_file="resultados_sarima.xlsx",
    p_values=range(1, 6),
    d_values=[1],
    q_values=range(1, 6),
    P_values=range(0, 6),
    D_values=range(0, 2),
    Q_values=range(0, 6),
    s=12
):
    """
    Executa a busca de hiperparâmetros SARIMA e salva os resultados em um arquivo Excel.

    Parâmetros:
        file_path (str): Caminho do arquivo de entrada com a série temporal.
        output_file (str): Nome do arquivo de saída com os resultados.
        p_values (range): Intervalo para o parâmetro p (autoregressivo).
        d_values (list): Lista para o parâmetro d (diferença).
        q_values (range): Intervalo para o parâmetro q (média móvel).
        P_values (range): Intervalo para o parâmetro P (autoregressivo sazonal).
        D_values (range): Intervalo para o parâmetro D (diferença sazonal).
        Q_values (range): Intervalo para o parâmetro Q (média móvel sazonal).
        s (int): Período sazonal (ex: 12 para dados mensais).
    """
    # Ignorar warnings
    warnings.filterwarnings("ignore")

    # Carregar os dados
    df_sarima = pd.read_excel(file_path)

    # Converter a coluna 'Data' para formato de data e definir como índice
    df_sarima['Data'] = pd.to_datetime(df_sarima['Data'])
    df_sarima.set_index('Data', inplace=True)

    # Definir a frequência da série temporal como mensal
    df_sarima.index.freq = 'MS'

    # Criar todas as combinações de parâmetros
    param_combinations = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values))

    # Criar lista para armazenar os resultados
    results = []

    # Loop com barra de progresso
    for i, (p, d, q, P, D, Q) in enumerate(tqdm(param_combinations, desc="Executando modelos SARIMA"), start=1):
        try:
            # Exibir progresso
            print(f"Testando modelo {i}/{len(param_combinations)} - Parâmetros: (p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q})")

            # Ajustar o modelo SARIMA
            model = SARIMAX(
                df_sarima['Valor'],  
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False)

            # Salvar os resultados
            results.append({
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q,
                'AIC': result.aic,
                'BIC': result.bic,
                'Log-Likelihood': result.llf
            })
        except Exception:
            pass  # Ignorar modelos que não convergem

    # Criar um DataFrame com os resultados
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
        
        # Salvar os resultados em Excel
        results_df.to_excel(output_file, index=False)
        
        print(f"Grid search concluído! Resultados salvos em '{output_file}'.")
    else:
        print("Nenhum modelo foi ajustado com sucesso.")