#ifndef ACTIVACION_H
#define ACTIVACION_H

#include<Eigen/Dense>
#include"modulo/modulo.h"

using namespace Eigen;

enum class Activacion_t{
    sigmoidea,
    tanh,
    relu
};

struct Activacion: public Modulo{
    Activacion_t tipo;
    static VectorXd sigmoidea(const VectorXd&);
    static VectorXd tanh(const VectorXd&);
    static VectorXd relu(const VectorXd&);
    Activacion(Activacion_t _tipo);
    virtual VectorXd operator()(const VectorXd&) override;
};

#endif