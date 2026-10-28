"""Tests de l'API de diagnostic (étape 10, exercice 4).

Lancer depuis la racine du projet :   pytest -v
"""
import importlib.util
import json
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(monkeypatch):
    """Charge 08_api_flask.py et renvoie un client de test Flask (aucun vrai serveur n'est lancé)."""
    monkeypatch.chdir(RACINE)          # l'API lit sorties/cnn.onnx et sorties/seuil.json en relatif
    monkeypatch.syspath_prepend(str(RACINE))
    # Le nom du fichier commence par un chiffre : on ne peut pas écrire « import 08_api_flask ».
    spec = importlib.util.spec_from_file_location("api_flask", RACINE / "08_api_flask.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api.app.test_client()


def test_signal_valide(client):
    """Un vrai signal de 512 nombres donne une réponse 200 complète et cohérente."""
    from simulateur import generer
    X, _ = generer(1, seed=7)
    reponse = client.post("/diagnostic", json={"signal": X[0].tolist()})

    assert reponse.status_code == 200
    corps = reponse.get_json()
    assert set(corps) == {"probabilite_defaut", "decision", "seuil"}
    assert 0.0 <= corps["probabilite_defaut"] <= 1.0
    assert corps["seuil"] == json.load(open(RACINE / "sorties" / "seuil.json"))["seuil"]
    attendue = "DEFAUT" if corps["probabilite_defaut"] >= corps["seuil"] else "saine"
    assert corps["decision"] == attendue


def test_signal_trop_court(client):
    """Une liste de 3 nombres est refusée avec un code 400 et un message d'erreur, sans décision."""
    reponse = client.post("/diagnostic", json={"signal": [1, 2, 3]})

    assert reponse.status_code == 400
    corps = reponse.get_json()
    assert "erreur" in corps
    assert "decision" not in corps
