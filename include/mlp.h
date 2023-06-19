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
    void derivadas_oculta(const int, const int, const int, MatrixXd&, VectorXd&);
    void derivadas_salida(const int, MatrixXd&, VectorXd&);
    double coste(const VectorXd&, const VectorXd&);
    const VectorXd& vector_activacion(const int, const int);
    const VectorXd& vector_derivada_activacion(const int);
    MatrixXd& matriz_pesos(const int);
    VectorXd& vector_sesgo(const int);

    MLP(MatrixXd, MatrixXd);
    ~MLP();

    void agregar_capa(Capa);
    VectorXd propagacion_adelante(const int);
    void propagacion_atras(const int, const double);

    void entrenar(const int, const double);
};


#endif //MLP_MLP_H
