#ifndef MLP_H
#define MLP_H

#include <Eigen/Dense>
#include<vector>
#include<list>
#include<iostream>
#include"capa.h"
using namespace std;
using namespace Eigen;

struct MLP {
    int n;
    int hidden_size;
    vector<Capa> capas;
    VectorXd salida;
    MatrixXd X, Y;

    VectorXd producto_recursivo(const int, const int, const int);
    // VectorXd coste(const VectorXd&, const VectorXd&);
    VectorXd derivada_coste(const VectorXd&, const VectorXd&);
    double coste(const VectorXd&, const VectorXd&);
    const VectorXd& vector_activacion(const int indice_capa);
    const VectorXd& vector_derivada_activacion(const int indice_capa);
    const MatrixXd& matriz_pesos(const int indice_capa);

    MLP();
    ~MLP();

    void agregar_capa(Capa);
    VectorXd propagacion_adelante(const VectorXd&, const VectorXd&);
    MatrixXd propagacion_atras();

    void entrenar(const MatrixXd&, const MatrixXd&);
};


#endif //MLP_MLP_H
