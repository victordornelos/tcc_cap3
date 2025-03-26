import os
import pandas as pd

# Defina o caminho da pasta de resultados
resultados_dir = 'resultado'

# Inicialize uma lista para armazenar os dataframes
dataframes = []

# Percorra todos os arquivos na pasta de resultados
for filename in os.listdir(resultados_dir):
    if filename.endswith('.xlsx'):
        filepath = os.path.join(resultados_dir, filename)
        df = pd.read_excel(filepath)
        dataframes.append(df)

# Combine todos os dataframes em um único dataframe
consolidated_df = pd.concat(dataframes)

# Ordene o dataframe pela coluna AIC em ordem decrescente
consolidated_df = consolidated_df.sort_values(by='AIC', ascending=True)

# Salve o dataframe consolidado em um novo arquivo Excel
output_filepath = 'resultado_sarima_consolidado.xlsx'
consolidated_df.to_excel(output_filepath, index=False)