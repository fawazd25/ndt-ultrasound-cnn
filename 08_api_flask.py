"""
Petite API : on envoie un signal brut, elle répond défaut / saine.
Lancer :   python 08_api_flask.py
Tester :   python 08_api_flask.py --test
"""
import json, sys
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from simulateur import normaliser, generer, LONGUEUR

app = Flask(__name__)
session = ort.InferenceSession("sorties/cnn.onnx", providers=["CPUExecutionProvider"])
SEUIL = json.load(open("sorties/seuil.json"))["seuil"]


@app.post("/diagnostic")
def diagnostic():
    donnees = request.get_json(silent=True) or {}
    signal = donnees.get("signal")
    # Ne JAMAIS faire confiance à l'entrée (réflexe cybersécurité) : on valide tout.
    if not isinstance(signal, list) or len(signal) != LONGUEUR:
        return jsonify(erreur=f"'signal' doit être une liste de {LONGUEUR} nombres"), 400
    try:
        x = np.asarray(signal, dtype=np.float64)[None, :]
    except (TypeError, ValueError):
        return jsonify(erreur="valeurs non numériques"), 400
    if not np.isfinite(x).all() or np.abs(x).max() > 1e4:
        return jsonify(erreur="valeurs hors plage"), 400
    xn = normaliser(x).reshape(1, 1, LONGUEUR)
    logit = float(session.run(None, {"signal": xn})[0][0])
    proba = 1.0 / (1.0 + np.exp(-logit))
    return jsonify(probabilite_defaut=round(float(proba), 4),
                   decision="DEFAUT" if proba >= SEUIL else "saine", seuil=SEUIL)


if __name__ == "__main__":
    if "--test" in sys.argv:
        client = app.test_client()
        X, y = generer(6, seed=102)
        for xi, yi in zip(X, y):
            r = client.post("/diagnostic", json={"signal": xi.tolist()})
            print("réel:", "défaut" if yi else "saine ", "->", r.get_json())
        print("entrée invalide ->", client.post("/diagnostic", json={"signal": [1, 2, 3]}).get_json())
        print("pas du JSON     ->", client.post("/diagnostic", data="bonjour").get_json())
    else:
        app.run(host="127.0.0.1", port=5000)
