// Inférence du CNN 1D en C++ pur, sans aucune bibliothèque.
// Objectif pédagogique : comprendre ce que fait réellement un réseau de neurones.
//
// Compilation :  g++ -O2 -std=c++17 -o infer infer.cpp
// Exécution   :  ./infer            (lit poids.txt, test_vecteurs.txt et ../sorties/seuil.json)
//
// Exercice 3 de l'étape 9 : compiler avec -DBUG_INDEX introduit volontairement
// une erreur d'indexation dans la convolution (g++ -O2 -std=c++17 -DBUG_INDEX ...).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
#ifdef BUG_INDEX
                    if (s >= 0 && s < L) somme += c.w[(o * c.c_in + i) * c.k] * x[i][s];   // bug volontaire : oublie + j
#else
                    if (s >= 0 && s < L) somme += c.w[(o * c.c_in + i) * c.k + j] * x[i][s];
#endif
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

// Lit le seuil d'inspection choisi à l'étape 5 dans ../sorties/seuil.json (0,5 si absent).
float lire_seuil(const std::string& chemin) {
    std::ifstream f(chemin);
    if (!f) return 0.5f;
    std::stringstream ss; ss << f.rdbuf();
    const std::string txt = ss.str();
    const auto pos = txt.find("\"seuil\"");
    if (pos == std::string::npos) return 0.5f;
    return std::stof(txt.substr(txt.find(':', pos) + 1));
}

float sigmoide(float z) { return 1.0f / (1.0f + std::exp(-z)); }

int main() {
    Modele m = charger("poids.txt");
    const float seuil = lire_seuil("../sorties/seuil.json");
    std::cout << "Seuil d'inspection : " << seuil << "\n\n";
    std::ifstream f("test_vecteurs.txt");
    if (!f) { std::cerr << "test_vecteurs.txt introuvable\n"; return 1; }
    int n, L;
    f >> n >> L;
    float ecart_max = 0;
    std::vector<std::vector<float>> signaux;
    for (int i = 0; i < n; ++i) {
        float attendu;
        f >> attendu;
        std::vector<float> s(L);
        for (auto& v : s) f >> v;
        float obtenu = predire(m, s);
        ecart_max = std::max(ecart_max, std::fabs(obtenu - attendu));
        const float p = sigmoide(obtenu);
        std::cout << "signal " << i << "  PyTorch=" << attendu << "  C++=" << obtenu
                  << "  proba=" << p << "  -> " << (p >= seuil ? "DÉFAUT" : "saine") << "\n";
        signaux.push_back(std::move(s));
    }
    std::cout << "\nÉcart maximal PyTorch vs C++ : " << ecart_max << "\n";

    // Chronomètre (exercice 1) : on prédit tous les signaux plusieurs fois et on mesure le temps moyen.
    const int repetitions = 50;
    volatile float puits = 0;                       // empêche le compilateur de supprimer les calculs
    for (const auto& s : signaux) puits = puits + predire(m, s);   // échauffement
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repetitions; ++r)
        for (const auto& s : signaux) puits = puits + predire(m, s);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / (repetitions * signaux.size());
    std::cout << "Temps moyen par signal : " << ms << " ms\n";
    return ecart_max < 1e-3 ? 0 : 2;
}
