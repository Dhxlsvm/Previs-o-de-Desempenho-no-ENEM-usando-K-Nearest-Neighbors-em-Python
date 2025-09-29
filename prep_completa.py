# -*- coding: utf-8 -*-
"""
Script Completo: Preparação de Dados e Implementação do K-NN para o ENEM.

Este script executa o processo do início ao fim:
1. Carrega o arquivo de microdados já filtrado.
2. Limpa, transforma e normaliza os dados.
3. Treina um modelo K-NN para cada área do conhecimento.
4. Simula um novo aluno e gera um relatório de previsão completo e comparativo.
"""

# =============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

print("--- INICIANDO SCRIPT DE ANÁLISE DE DESEMPENHO NO ENEM ---")

# =============================================================================
# 2. CONFIGURAÇÃO
# =============================================================================
caminho_dados_filtrados = 'DADOS/MICRODADOS_ENEM_2023_CE_FILTRADO.csv' 

colunas_notas = [
    'NU_NOTA_MT',     
    'NU_NOTA_CN',      
    'NU_NOTA_LC',     
    'NU_NOTA_CH',      
    'NU_NOTA_REDACAO'      
]
colunas_features = [
    'Q006',                
    'Q002',               
    'TP_ESCOLA',          
    'TP_COR_RACA',         
    'CO_MUNICIPIO_ESC' 
]

# =============================================================================
# 3. PREPARAÇÃO DOS DADOS 
# =============================================================================
print(f"\n[FASE 1 de 3] Carregando e preparando os dados de '{caminho_dados_filtrados}'...")

# Carregamento
df = pd.read_csv(caminho_dados_filtrados, sep=',', encoding='latin-1')

# Limpeza
df.dropna(subset=colunas_notas, inplace=True)
df = df[df['NU_NOTA_REDACAO'] != 0]

# Recriar Estado
df['CO_MUNICIPIO_ESC'] = df['CO_MUNICIPIO_ESC'].astype(str)
df['CO_UF_RESIDENCIA'] = df['CO_MUNICIPIO_ESC'].str[:2]
mapa_uf = { '23': 'CE', '35': 'SP', '33': 'RJ', '29': 'BA', '31': 'MG', '26': 'PE' } # Adicione mais se precisar
df['SG_UF_RESIDENCIA'] = df['CO_UF_RESIDENCIA'].map(mapa_uf)
df.drop(columns=['CO_MUNICIPIO_ESC', 'CO_UF_RESIDENCIA'], inplace=True)
df['SG_UF_RESIDENCIA'].fillna('OUTRO', inplace=True) # Preenche estados não mapeados

colunas_categoricas = ['Q006', 'Q002', 'TP_ESCOLA', 'TP_COR_RACA', 'SG_UF_RESIDENCIA']
df_processed = pd.get_dummies(df, columns=colunas_categoricas)

# Separação e Normalização
y_final = df_processed[colunas_notas].reset_index(drop=True)
X = df_processed.drop(columns=colunas_notas)
feature_names = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_final = pd.DataFrame(X_scaled, columns=feature_names)

print("Dados preparados com sucesso!")

# =============================================================================
# 4. TREINAMENTO DOS MODELOS K-NN
# =============================================================================
print("\n[FASE 2 de 3] Treinando os modelos K-NN...")

# Divisão geral dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

modelos_knn = {}

for nota_alvo in colunas_notas:
    print(f" -> Treinando modelo para {nota_alvo}...")
    
    y_alvo_train = y_train[nota_alvo]
    
    knn = KNeighborsRegressor(n_neighbors=9)
    knn.fit(X_train, y_alvo_train)
    
    modelos_knn[nota_alvo] = knn

print("Todos os 5 modelos foram treinados com sucesso!")

# =============================================================================
# 5. PREVISÃO E ANÁLISE COMPARATIVA PARA UM NOVO ALUNO
# =============================================================================
print("\n[FASE 3 de 3] Gerando relatório de previsão para um aluno exemplo...")

# Pega o primeiro aluno do conjunto de teste para simular um "novo aluno"
novo_aluno_features = X_test.iloc[[0]]
notas_reais_aluno = y_test.iloc[0]

# --- INÍCIO DO RELATÓRIO ---
print("\n" + "="*70)
print("RELATÓRIO DE PREVISÃO DE DESEMPENHO NO ENEM")
print("="*70)
print("Este relatório apresenta a nota prevista para um aluno exemplo e a compara")
print("com o desempenho dos 9 alunos de perfil mais similar na base de dados.")
print("-"*70)

# Loop para fazer a previsão e análise para cada matéria
for nome_nota, modelo in modelos_knn.items():
    
    # Dados de treino correspondentes a esta nota
    y_alvo_train = y_train[nome_nota]
    
    # 1. Previsão
    nota_prevista = modelo.predict(novo_aluno_features)[0]
    
    # 2. Encontrar Vizinhos
    distancias, indices = modelo.kneighbors(novo_aluno_features)
    
    # 3. Analisar Vizinhos
    notas_dos_vizinhos = y_alvo_train.iloc[indices[0]]
    media_vizinhos = notas_dos_vizinhos.mean()
    melhor_nota_vizinhos = notas_dos_vizinhos.max()
    pior_nota_vizinhos = notas_dos_vizinhos.min()
    
    # 4. Apresentar resultado para a matéria
    print(f"\n>>> ANÁLISE PARA: {nome_nota} <<<")
    print(f"    Nota Real (referência): {notas_reais_aluno[nome_nota]:.2f}")
    print(f"    PREVISÃO DE NOTA: {nota_prevista:.2f}")
    print("    -----------------------------------------")
    print("    Comparativo com os Vizinhos:")
    print(f"    - Média dos vizinhos: {media_vizinhos:.2f}")
    print(f"    - Faixa de notas (pior a melhor): {pior_nota_vizinhos:.2f} a {melhor_nota_vizinhos:.2f}")

    # 5. Avaliação final
    if nota_prevista > media_vizinhos * 1.05: # 5% acima da média
        print("    [AVALIAÇÃO]: Desempenho previsto é SUPERIOR à média do grupo de referência.")
    elif nota_prevista < media_vizinhos * 0.95: # 5% abaixo da média
        print("    [AVALIAÇÃO]: Desempenho previsto é INFERIOR à média do grupo de referência. [cite: 55]")
    else:
        print("    [AVALIAÇÃO]: Desempenho previsto está NA MÉDIA do grupo de referência.")

print("\n" + "="*70)
print("FIM DO RELATÓRIO")
print("="*70)