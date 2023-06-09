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
    list<Capa> capas;
    VectorXd salida;

    VectorXd coste(const VectorXd&, const VectorXd&);
    VectorXd derivada_coste(const VectorXd&, const VectorXd&);

    MLP();
    ~MLP();

    void agregar_capa(Capa);
    VectorXd propagacion_adelante(const VectorXd&, const VectorXd&);
    MatrixXd propagacion_atras();

    void entrenar(const MatrixXd&, const MatrixXd&);
};


#endif //MLP_MLP_H
