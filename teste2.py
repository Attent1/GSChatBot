import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

app = Flask(__name__)

# Modelos fictícios
modelo_regressao = LinearRegression()
modelo_clusterizacao = KMeans(n_clusters=3)

@app.route('/')
def mensagem_teste():
    return 'Teste Flask em execução!'

@app.route('/treinar_e_prever', methods=['POST', 'GET'])
def treinar_e_prever():
    try:
        if request.method == 'POST':
            dados_treino_previsao = request.get_json()

            # Obter dados de treinamento para regressão
            X_treino_regressao = dados_treino_previsao['dados_regressao']['X']
            y_treino_regressao = dados_treino_previsao['dados_regressao']['y']

            # Treinar o modelo de regressão
            modelo_regressao.fit(X_treino_regressao, y_treino_regressao)

            # Prever com o modelo de regressão
            X_previsao_regressao = dados_treino_previsao.get('dados_regressao', {}).get('X', [])
            if len(X_previsao_regressao) > 0:
                resultado_regressao = modelo_regressao.predict(np.array(X_previsao_regressao).reshape(-1, 2)).tolist()
            else:
                resultado_regressao = []

            # Retornar os resultados como resposta JSON
            return jsonify({'resultado_regressao': resultado_regressao})
           
    except Exception as e:
        print(f"Erro no servidor: {e}")
        return jsonify({'erro': 'Erro interno no servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True)