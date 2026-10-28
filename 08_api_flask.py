"""
Petite API : on envoie un signal brut, elle répond défaut / saine.
Lancer :   python 08_api_flask.py
Tester :   python 08_api_flask.py --test
"""
import hashlib, json, logging, sys
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from simulateur import normaliser, generer, LONGUEUR

app = Flask(__name__)
# Taille maximale d'une requête (exercice 2) : un signal de 512 nombres en JSON pèse
# environ 10 Ko. Au-delà de 64 Ko, Flask refuse la requête AVANT de la lire en entier.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
session = ort.InferenceSession("sorties/cnn.onnx", providers=["CPUExecutionProvider"])
SEUIL = json.load(open("sorties/seuil.json"))["seuil"]

# Journal des requêtes (exercice 3) : heure, décision, probabilité. Jamais le signal complet,
# seulement une empreinte (hash) qui permet de retrouver un signal sans le stocker.
journal = logging.getLogger("diagnostics")
journal.setLevel(logging.INFO)
_fichier = logging.FileHandler("sorties/requetes.log", encoding="utf-8")
_fichier.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"))
journal.addHandler(_fichier)
journal.propagate = False


def empreinte(signal):
    """16 premiers caractères du SHA-256 du signal : identifie le signal sans le révéler."""
    return hashlib.sha256(json.dumps(signal).encode()).hexdigest()[:16]


@app.post("/diagnostic")
def diagnostic():
    donnees = request.get_json(silent=True) or {}
    signal = donnees.get("signal")
    # Ne JAMAIS faire confiance à l'entrée (réflexe cybersécurité) : on valide tout.
    if not isinstance(signal, list) or len(signal) != LONGUEUR:
        journal.info("rejet=format")
        return jsonify(erreur=f"'signal' doit être une liste de {LONGUEUR} nombres"), 400
    try:
        x = np.asarray(signal, dtype=np.float64)[None, :]
    except (TypeError, ValueError):
        journal.info("rejet=non_numerique")
        return jsonify(erreur="valeurs non numériques"), 400
    if not np.isfinite(x).all() or np.abs(x).max() > 1e4:
        journal.info("rejet=hors_plage")
        return jsonify(erreur="valeurs hors plage"), 400
    xn = normaliser(x).reshape(1, 1, LONGUEUR)
    logit = float(session.run(None, {"signal": xn})[0][0])
    proba = 1.0 / (1.0 + np.exp(-logit))
    decision = "DEFAUT" if proba >= SEUIL else "saine"
    journal.info(f"decision={decision} proba={proba:.4f} seuil={SEUIL} signal={empreinte(signal)}")
    return jsonify(probabilite_defaut=round(float(proba), 4),
                   decision=decision, seuil=SEUIL)


@app.errorhandler(413)
def requete_trop_grosse(_):
    journal.info("rejet=trop_volumineux")
    """Par défaut, Flask répond une page HTML : on garde une réponse JSON, comme le reste de l'API."""
    return jsonify(erreur=f"requête trop volumineuse (maximum {app.config['MAX_CONTENT_LENGTH'] // 1024} Ko)"), 413


@app.get("/sante")
def sante():
    """Contrôle de santé : le service répond ET le modèle sait encore calculer."""
    try:
        entree = session.get_inputs()[0]
        # Une prédiction sur un signal nul : très rapide, mais elle prouve que le modèle fonctionne.
        sortie = session.run(None, {entree.name: np.zeros((1, 1, LONGUEUR), dtype=np.float32)})[0]
        if not np.isfinite(sortie).all():
            raise ValueError("sortie du modèle non finie")
    except Exception as e:
        return jsonify(statut="erreur", detail=str(e)), 503
    return jsonify(statut="ok", modele="sorties/cnn.onnx",
                   entree=[str(d) for d in entree.shape], seuil=SEUIL)


if __name__ == "__main__":
    if "--test" in sys.argv:
        client = app.test_client()
        X, y = generer(6, seed=102)
        for xi, yi in zip(X, y):
            r = client.post("/diagnostic", json={"signal": xi.tolist()})
            print("réel:", "défaut" if yi else "saine ", "->", r.get_json())
        print("entrée invalide ->", client.post("/diagnostic", json={"signal": [1, 2, 3]}).get_json())
        print("pas du JSON     ->", client.post("/diagnostic", data="bonjour").get_json())
        normal = json.dumps({"signal": X[0].tolist()})
        print(f"taille d'une requête normale : {len(normal) / 1024:.1f} Ko")
        enorme = json.dumps({"signal": [0.123456789] * 2_000_000})          # environ 24 Mo
        r = client.post("/diagnostic", data=enorme, content_type="application/json")
        print(f"énorme JSON ({len(enorme) / 1e6:.0f} Mo) ->", r.status_code, r.get_json())
        r = client.get("/sante")
        print("santé           ->", r.status_code, r.get_json())
        print("\nDernières lignes de sorties/requetes.log :")
        with open("sorties/requetes.log", encoding="utf-8") as f:
            print("".join(f.readlines()[-11:]), end="")
    else:
        app.run(host="127.0.0.1", port=5000)
