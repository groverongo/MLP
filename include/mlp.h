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

    VectorXd coste(const VectorXd&, const VectorXd&);
    VectorXd derivada_coste(const VectorXd&, const VectorXd&);

    MLP(MatrixXd, MatrixXd);
    ~MLP();

    void agregar_capa(Capa);
    VectorXd propagacion_adelante(const int);
    void propagacion_atras(const int, const double);

    void entrenar(const int, const double);
};


#endif //MLP_MLP_H
