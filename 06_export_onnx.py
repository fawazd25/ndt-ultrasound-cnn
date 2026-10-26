"""Exporte le modèle PyTorch au format ONNX et vérifie que les résultats sont identiques."""
import numpy as np
import torch
import onnxruntime as ort
from simulateur import jeux_standard
from modele import CNN1D

modele = CNN1D()
modele.load_state_dict(torch.load("sorties/cnn_meilleur.pt"))
modele.eval()

exemple = torch.zeros(1, 1, 512)
torch.onnx.export(
    modele, exemple, "sorties/cnn.onnx",
    input_names=["signal"], output_names=["logit"],
    dynamic_axes={"signal": {0: "lot"}, "logit": {0: "lot"}},
    opset_version=17, dynamo=False,
)
print("Écrit : sorties/cnn.onnx")

_, _, (Xte, yte) = jeux_standard()
sess = ort.InferenceSession("sorties/cnn.onnx", providers=["CPUExecutionProvider"])
X = Xte[:200].reshape(200, 1, 512)
sortie_onnx = sess.run(None, {"signal": X})[0]
with torch.no_grad():
    sortie_torch = modele(torch.from_numpy(X)).numpy()
ecart = np.abs(sortie_onnx - sortie_torch).max()
print(f"Écart max PyTorch vs ONNX Runtime : {ecart:.2e}")
assert ecart < 1e-4, "Les deux sorties devraient être quasi identiques"
