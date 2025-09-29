# Autor: Darison Rodrigues da Silva
# Disciplina: Sistemas de Apoio a Tomada de Decisão
# Trabalho: AV1 - Previsão de Desempenho no ENEM

# =============================================================================
# 1. IMPORTAÇÕES
# =============================================================================
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import threading

# =============================================================================
# 2. FUNÇÃO DE CARGA E TREINAMENTO
# =============================================================================


def carregar_dados_e_modelos():
    try:
        caminho_dados = 'DADOS/MICRODADOS_ENEM_2023_CE_FILTRADO.csv'  # VERIFIQUE ESTE NOME

        df = pd.read_csv(caminho_dados, sep=',')

        colunas_notas = ['NU_NOTA_MT', 'NU_NOTA_CN',
                         'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO']

        df.dropna(subset=colunas_notas, inplace=True)
        df = df[df['NU_NOTA_REDACAO'] != 0]

        df['CO_MUNICIPIO_ESC'] = df['CO_MUNICIPIO_ESC'].astype(str)
        df['CO_UF_RESIDENCIA'] = df['CO_MUNICIPIO_ESC'].str[:2]
        mapa_uf = {'23': 'CE', '35': 'SP', '33': 'RJ',
                   '29': 'BA', '31': 'MG', '26': 'PE'}
        df['SG_UF_RESIDENCIA'] = df['CO_UF_RESIDENCIA'].map(mapa_uf)
        df.drop(columns=['CO_MUNICIPIO_ESC', 'CO_UF_RESIDENCIA'], inplace=True)
        df['SG_UF_RESIDENCIA'].fillna('OUTRO', inplace=True)

        colunas_categoricas = ['Q006', 'Q002',
                               'TP_ESCOLA', 'TP_COR_RACA', 'SG_UF_RESIDENCIA']
        df_processed = pd.get_dummies(df, columns=colunas_categoricas)

        y = df_processed[colunas_notas]
        X = df_processed.drop(columns=colunas_notas)

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_final = pd.DataFrame(X_scaled, columns=X.columns)

        modelos_knn = {}
        for nota_alvo in colunas_notas:
            knn = KNeighborsRegressor(n_neighbors=9)
            knn.fit(X_final, y[nota_alvo])
            modelos_knn[nota_alvo] = knn

        # Retorna tudo que a aplicação principal vai precisar
        return modelos_knn, scaler, X.columns, X_final, y
    except Exception as e:
        # Se algo der errado, retorna o erro
        return e

# =============================================================================
# 3. CLASSE DA APLICAÇÃO PRINCIPAL
# =============================================================================


class AppKNN:
    def __init__(self, root, dados_carregados):
        self.root = root
        self.root.title("Previsão de Desempenho no ENEM")
        self.root.geometry("800x600")

        # Desempacota os dados carregados
        self.modelos, self.scaler, self.colunas_treinamento, self.X_treino, self.y_treino = dados_carregados

        # --- Frames para organização ---
        frame_input = tk.Frame(root, padx=10, pady=10)
        frame_input.pack(fill='x')
        frame_output = tk.Frame(root, padx=10, pady=10)
        frame_output.pack(fill='both', expand=True)

        # --- Dicionários para os menus ---
        self.mapa_renda = {'Nenhuma Renda': 'A', 'Até R$ 1.212': 'B',
                           'De R$ 1.212 a R$ 1.818': 'C', 'Mais de R$ 24.240': 'Q'}
        self.mapa_escolaridade_mae = {'Não estudou': 'A', 'Fundamental Incompleto': 'B', 'Fundamental Completo': 'C',
                                      'Médio Incompleto': 'D', 'Médio Completo': 'E', 'Superior Completo': 'F', 'Pós-graduação': 'G', 'Não sabe': 'H'}
        self.mapa_tipo_escola = {'Pública': 2,
                                 'Privada': 3, 'Não Respondeu': 1}
        self.mapa_raca = {'Branca': 1, 'Preta': 2, 'Parda': 3,
                          'Amarela': 4, 'Indígena': 5, 'Não declarado': 0}
        self.mapa_estados = ['CE', 'SP', 'RJ', 'BA', 'MG', 'PE', 'OUTRO']

        # --- Widgets de Input ---
        ttk.Label(frame_input, text="Renda Familiar:").grid(
            row=0, column=0, sticky='w', padx=5, pady=5)
        self.renda_var = tk.StringVar()
        self.renda_cb = ttk.Combobox(frame_input, textvariable=self.renda_var, values=list(
            self.mapa_renda.keys()), state="readonly")
        self.renda_cb.grid(row=0, column=1, sticky='ew', padx=5)
        self.renda_cb.set(list(self.mapa_renda.keys())[0])  # Valor padrão

        ttk.Label(frame_input, text="Escolaridade da Mãe:").grid(
            row=1, column=0, sticky='w', padx=5, pady=5)
        self.escolaridade_var = tk.StringVar()
        self.escolaridade_cb = ttk.Combobox(frame_input, textvariable=self.escolaridade_var, values=list(
            self.mapa_escolaridade_mae.keys()), state="readonly")
        self.escolaridade_cb.grid(row=1, column=1, sticky='ew', padx=5)
        self.escolaridade_cb.set(list(self.mapa_escolaridade_mae.keys())[0])

        ttk.Label(frame_input, text="Tipo de Escola:").grid(
            row=0, column=2, sticky='w', padx=5, pady=5)
        self.escola_var = tk.StringVar()
        self.escola_cb = ttk.Combobox(frame_input, textvariable=self.escola_var, values=list(
            self.mapa_tipo_escola.keys()), state="readonly")
        self.escola_cb.grid(row=0, column=3, sticky='ew', padx=5)
        self.escola_cb.set(list(self.mapa_tipo_escola.keys())[0])

        ttk.Label(frame_input, text="Raça/Cor:").grid(row=1,
                                                      column=2, sticky='w', padx=5, pady=5)
        self.raca_var = tk.StringVar()
        self.raca_cb = ttk.Combobox(frame_input, textvariable=self.raca_var, values=list(
            self.mapa_raca.keys()), state="readonly")
        self.raca_cb.grid(row=1, column=3, sticky='ew', padx=5)
        self.raca_cb.set(list(self.mapa_raca.keys())[0])

        ttk.Label(frame_input, text="Estado:").grid(
            row=2, column=0, sticky='w', padx=5, pady=5)
        self.estado_var = tk.StringVar()
        self.estado_cb = ttk.Combobox(
            frame_input, textvariable=self.estado_var, values=self.mapa_estados, state="readonly")
        self.estado_cb.grid(row=2, column=1, sticky='ew', padx=5)
        self.estado_cb.set(self.mapa_estados[0])

        # --- Botão e Área de Resultado ---
        self.botao_prever = ttk.Button(
            frame_input, text="Gerar Previsão e Análise", command=self.gerar_previsao)
        self.botao_prever.grid(row=3, column=0, columnspan=4, pady=10)

        self.texto_resultado = scrolledtext.ScrolledText(
            frame_output, wrap=tk.WORD, state="disabled")
        self.texto_resultado.pack(fill='both', expand=True)

    def gerar_previsao(self):
        """Função chamada pelo botão para gerar e exibir o relatório."""
        # --- Coleta e transforma os inputs ---
        dados_usuario = {
            'Q006': [self.mapa_renda[self.renda_var.get()]],
            'Q002': [self.mapa_escolaridade_mae[self.escolaridade_var.get()]],
            'TP_ESCOLA': [self.mapa_tipo_escola[self.escola_var.get()]],
            'TP_COR_RACA': [self.mapa_raca[self.raca_var.get()]],
            'SG_UF_RESIDENCIA': [self.estado_var.get()]
        }
        input_df = pd.DataFrame(dados_usuario)
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(
            columns=self.colunas_treinamento, fill_value=0)
        input_scaled = self.scaler.transform(input_aligned)

        # --- Gera o texto do relatório ---
        relatorio = "RELATÓRIO DE PREVISÃO DE DESEMPENHO NO ENEM\n"
        relatorio += "="*60 + "\n"

        for nome_nota, modelo in self.modelos.items():
            nota_prevista = modelo.predict(input_scaled)[0]
            dist, indices = modelo.kneighbors(input_scaled)
            notas_dos_vizinhos = self.y_treino[nome_nota].iloc[indices[0]]
            media_vizinhos = notas_dos_vizinhos.mean()

            relatorio += f"\n>>> ANÁLISE PARA: {nome_nota} <<<\n"
            relatorio += f"    PREVISÃO DE NOTA: {nota_prevista:.2f}\n"
            relatorio += f"    Média dos vizinhos: {media_vizinhos:.2f}\n"

            if nota_prevista > media_vizinhos * 1.05:
                relatorio += "    [AVALIAÇÃO]: Desempenho previsto é SUPERIOR à média do grupo de referência.\n"
            elif nota_prevista < media_vizinhos * 0.95:
                relatorio += "    [AVALIAÇÃO]: Desempenho previsto é INFERIOR à média do grupo de referência.\n"
            else:
                relatorio += "    [AVALIAÇÃO]: Desempenho previsto está NA MÉDIA do grupo de referência.\n"

        # --- Exibe o relatório na tela ---
        self.texto_resultado.config(state="normal")
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_resultado.insert(tk.INSERT, relatorio)
        self.texto_resultado.config(state="disabled")


# =============================================================================
# 4. EXECUÇÃO DA APLICAÇÃO
# =============================================================================
if __name__ == "__main__":
    root_loading = tk.Tk()
    root_loading.title("Carregando")
    root_loading.geometry("300x100")
    ttk.Label(root_loading,
              text="Carregando modelos e dados...\nPor favor, aguarde.").pack(pady=20)
    root_loading.update()

    dados_carregados = carregar_dados_e_modelos()

    root_loading.destroy()

    if isinstance(dados_carregados, Exception):
        root_error = tk.Tk()
        root_error.title("Erro")
        ttk.Label(root_error, text=f"Ocorreu um erro ao carregar os dados:\n{dados_carregados}").pack(
            padx=10, pady=10)
        root_error.mainloop()
    else:
        root_main = tk.Tk()
        app = AppKNN(root_main, dados_carregados)
        root_main.mainloop()
