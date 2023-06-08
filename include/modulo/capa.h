#ifndef CAPA_H
#define CAPA_H

#include<Eigen/Dense>
#include"modulo/modulo.h"

using namespace Eigen;

struct Capa: public Modulo{
    VectorXd sesgo;
    // matriz de pesos
    MatrixXd pesos;
    // funcion de activacion
    Capa(int, int);

    virtual VectorXd operator()(const VectorXd&) override;
};

#endif