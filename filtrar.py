import pandas as pd
import time

caminho_dados_completos = 'DADOS/MICRODADOS_ENEM_2023.csv' 

caminho_dados_filtrados = 'DADOS/MICRODADOS_ENEM_2023_CE_FILTRADO.csv'

codigo_estado_desejado = '23' 

colunas_relevantes = [
    'NU_NOTA_MT',
    'NU_NOTA_CN',
    'NU_NOTA_LC',
    'NU_NOTA_CH',
    'NU_NOTA_REDACAO',
    'Q006',
    'Q002',
    'TP_ESCOLA',
    'TP_COR_RACA',
    'CO_MUNICIPIO_ESC'
]

start_time = time.time()
print(f"Iniciando a leitura e filtragem do arquivo: {caminho_dados_completos}")
print(f"Filtrando pelo código de estado IBGE: '{codigo_estado_desejado}'")

chunk_iterator = pd.read_csv(
    caminho_dados_completos, 
    sep=';', encoding='latin-1', chunksize=100000, usecols=colunas_relevantes
)

dados_filtrados_lista = []
chunk_count = 0

for chunk in chunk_iterator:
    chunk_count += 1
    print(f"Processando pedaço nº {chunk_count}...")
  
    chunk['CO_MUNICIPIO_ESC'] = chunk['CO_MUNICIPIO_ESC'].astype(str)

    chunk_filtrado = chunk[chunk['CO_MUNICIPIO_ESC'].str.startswith(codigo_estado_desejado)]
    
    if not chunk_filtrado.empty:
        dados_filtrados_lista.append(chunk_filtrado)

print("\nTodos os pedaços foram processados. Consolidando o arquivo final...")

if dados_filtrados_lista:
    df_final = pd.concat(dados_filtrados_lista)
    df_final.to_csv(caminho_dados_filtrados, index=False, sep=',')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n--- PROCESSO CONCLUÍDO ---")
    print(f"Arquivo filtrado salvo com sucesso em: '{caminho_dados_filtrados}'")
    print(f"Formato do novo arquivo: {df_final.shape[0]} linhas, {df_final.shape[1]} colunas.")
    print(f"Tempo total de execução: {total_time:.2f} segundos.")
else:
    print("\nNenhum dado encontrado para o código de estado especificado. Verifique o código e o arquivo.")