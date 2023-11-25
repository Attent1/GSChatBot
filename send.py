import requests
import pandas as pd

# URL do servidor Flask
url_servidor_flask = 'http://localhost:5000/treinar_e_prever'

dados = pd.read_csv('heart_attack_prediction_dataset.csv')
colunas_relevantes = ['Age', 'Sex', 'Smoking', 'Heart Rate']
print("Colunas do DataFrame:", colunas_relevantes)
print(dados.dtypes)

dados.loc[dados['Sex'] == 'Male', 'Sex'] = 1
dados.loc[dados['Sex'] == 'Female', 'Sex'] = 0

dados_json = {
    'dados_regressao': {
        'X': dados[['Age', 'Sex']].values.tolist(),
        'y': dados['Heart Attack Risk'].tolist()  
    }
}

# Enviar solicitação POST para o servidor Flask
resposta = requests.post(url_servidor_flask, json=dados_json)

# Verificar a resposta
if resposta.status_code == 200:
    resultado = resposta.json()
    print(resultado)
else:
    print('Erro na solicitação:', resposta.status_code)