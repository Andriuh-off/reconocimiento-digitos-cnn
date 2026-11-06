import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='static')
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos')

#(nombre, archivo, ¿es CNN?)
MODELOS = [
    ('Red densa 1', 'modelo1.keras', False),
    ('Red densa 2', 'modelo2.keras', False),
    ('CNN', 'modelo3_cnn.keras', True),
]

cargados = []
for nombre, archivo, es_cnn in MODELOS:
    ruta = os.path.join(CARPETA, archivo)
    if not os.path.exists(ruta):
        raise SystemExit(f'Falta {ruta}. Ejecuta la PARTE 5 del notebook para guardar los modelos '
                         'y copia la carpeta "modelos" junto a app.py.')
    cargados.append((nombre, tf.keras.models.load_model(ruta), es_cnn))


@app.route('/')
def inicio():
    base = os.path.dirname(os.path.abspath(__file__))
    #busca index.html en la carpeta static o junto a app.py
    for carpeta in (os.path.join(base, 'static'), base):
        if os.path.exists(os.path.join(carpeta, 'index.html')):
            return send_from_directory(carpeta, 'index.html')
    return 'No se encuentra index.html junto a app.py ni en la carpeta static.', 404


@app.route('/predict', methods=['POST'])
def predict():
    x = np.array(request.get_json().get('pixels', []), dtype='float32')
    if x.size != 784:
        return jsonify(error='Se esperaban 784 píxeles'), 400
    salida = []
    for nombre, modelo, es_cnn in cargados:
        entrada = x.reshape(1, 28, 28, 1) if es_cnn else x.reshape(1, 784)
        p = modelo.predict(entrada, verbose=0)[0]
        salida.append({'nombre': nombre, 'digito': int(p.argmax()),
                       'confianza': float(p.max()), 'probs': [float(v) for v in p]})
    return jsonify(modelos=salida)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
