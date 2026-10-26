// Inférence du CNN 1D en C++ pur, sans aucune bibliothèque.
// Objectif pédagogique : comprendre ce que fait réellement un réseau de neurones.
//
// Compilation :  g++ -O2 -std=c++17 -o infer infer.cpp
// Exécution   :  ./infer            (lit poids.txt et test_vecteurs.txt)

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Conv {
    int c_out, c_in, k;
    std::vector<float> w;  // forme (c_out, c_in, k), rangée à plat
    std::vector<float> b;  // forme (c_out)
};

struct Modele {
    std::vector<Conv> convs;
    std::vector<float> fc_w;
    float fc_b;
};

Modele charger(const std::string& chemin) {
    std::ifstream f(chemin);
    if (!f) throw std::runtime_error("poids.txt introuvable (lance exporter_poids.py)");
    Modele m;
    int n;
    f >> n;
    for (int i = 0; i < n; ++i) {
        Conv c;
        f >> c.c_out >> c.c_in >> c.k;
        c.w.resize(c.c_out * c.c_in * c.k);
        c.b.resize(c.c_out);
        for (auto& v : c.w) f >> v;
        for (auto& v : c.b) f >> v;
        m.convs.push_back(std::move(c));
    }
    int taille_fc;
    f >> taille_fc;
    m.fc_w.resize(taille_fc);
    for (auto& v : m.fc_w) f >> v;
    f >> m.fc_b;
    return m;
}

// Un signal multi-canaux : canaux[c][t]
using Tenseur = std::vector<std::vector<float>>;

// Convolution 1D avec remplissage de zéros ("same"), comme padding=k//2 en PyTorch.
Tenseur convolution(const Tenseur& x, const Conv& c) {
    const int L = x[0].size(), pad = c.k / 2;
    Tenseur y(c.c_out, std::vector<float>(L, 0.0f));
    for (int o = 0; o < c.c_out; ++o)
        for (int t = 0; t < L; ++t) {
            float somme = c.b[o];
            for (int i = 0; i < c.c_in; ++i)
                for (int j = 0; j < c.k; ++j) {
                    int s = t + j - pad;
                    if (s >= 0 && s < L) somme += c.w[(o * c.c_in + i) * c.k + j] * x[i][s];
                }
            y[o][t] = somme;
        }
    return y;
}

void relu(Tenseur& x) {
    for (auto& canal : x)
        for (auto& v : canal) v = std::max(0.0f, v);
}

// Max-pooling de taille 2 : on garde le plus grand de chaque paire.
Tenseur maxpool2(const Tenseur& x) {
    const int L2 = x[0].size() / 2;
    Tenseur y(x.size(), std::vector<float>(L2));
    for (size_t c = 0; c < x.size(); ++c)
        for (int t = 0; t < L2; ++t) y[c][t] = std::max(x[c][2 * t], x[c][2 * t + 1]);
    return y;
}

float predire(const Modele& m, const std::vector<float>& signal) {
    Tenseur x(1, signal);
    for (const auto& c : m.convs) {
        x = convolution(x, c);
        relu(x);
        x = maxpool2(x);
    }
    // Moyenne globale sur le temps, puis couche linéaire.
    float logit = m.fc_b;
    for (size_t c = 0; c < x.size(); ++c) {
        float moy = 0;
        for (float v : x[c]) moy += v;
        moy /= x[c].size();
        logit += m.fc_w[c] * moy;
    }
    return logit;
}

int main() {
    Modele m = charger("poids.txt");
    std::ifstream f("test_vecteurs.txt");
    if (!f) { std::cerr << "test_vecteurs.txt introuvable\n"; return 1; }
    int n, L;
    f >> n >> L;
    float ecart_max = 0;
    for (int i = 0; i < n; ++i) {
        float attendu;
        f >> attendu;
        std::vector<float> s(L);
        for (auto& v : s) f >> v;
        float obtenu = predire(m, s);
        ecart_max = std::max(ecart_max, std::fabs(obtenu - attendu));
        std::cout << "signal " << i << "  PyTorch=" << attendu << "  C++=" << obtenu
                  << "  -> " << (obtenu > 0 ? "DÉFAUT" : "saine") << "\n";
    }
    std::cout << "\nÉcart maximal PyTorch vs C++ : " << ecart_max << "\n";
    return ecart_max < 1e-3 ? 0 : 2;
}
