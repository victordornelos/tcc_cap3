#pip install -r requirements.txt

import warnings
import pandas as pd
import itertools
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed, cpu_count

def run_sarima_grid_search(
    file_path="df_sarima.xlsx",
    output_file="resultados_sarima.xlsx",
    p_values=range(1, 6),
    d_values=[1],
    q_values=range(1, 6),
    P_values=range(0, 6),
    D_values=range(1, 2),
    Q_values=range(0, 6),
    s=12
):
    """
    Executa a busca de hiperparâmetros SARIMA e salva os resultados em um arquivo Excel.
    """
    warnings.filterwarnings("ignore")

    # Carregar os dados
    df_sarima = pd.read_excel(file_path)

    # Converter a coluna 'Data' para formato de data e definir como índice
    df_sarima['Data'] = pd.to_datetime(df_sarima['Data'])
    df_sarima.set_index('Data', inplace=True)
    df_sarima.index.freq = 'MS'

    # Criar todas as combinações de parâmetros
    param_combinations = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values))
    total_combinations = len(param_combinations)

    print(f"📌 Total de combinações a testar: {total_combinations}")
    print(f"🔄 Rodando em até {cpu_count()} núcleos...\n")

    # Função que executa cada modelo SARIMA
    def fit_sarima(params, df, i):
        p, d, q, P, D, Q = params
        try:
            # Exibir progresso com ID do processo
            print(f"[Processo {os.getpid()}] Testando modelo {i}/{total_combinations} - "
                  f"(p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q})")

            model = SARIMAX(
                df['Valor'],  
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False)

            return {
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q,
                'AIC': result.aic,
                'BIC': result.bic,
                'Log-Likelihood': result.llf
            }
        except Exception:
            return None

    # Rodando os modelos em paralelo
    results = Parallel(n_jobs=cpu_count(), backend="loky", verbose=10)(
        delayed(fit_sarima)(params, df_sarima, i) for i, params in enumerate(param_combinations, start=1)
    )

    # Remover modelos que falharam
    results = [r for r in results if r is not None]

    # Criar DataFrame de resultados
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
        results_df.to_excel(output_file, index=False)
        print(f"\n✅ Grid search concluído! Resultados salvos em '{output_file}'.")
    else:
        print("\n❌ Nenhum modelo foi ajustado com sucesso.")

