#include "modulo/activacion.h"

VectorXd derivada_sigmoidea(const VectorXd& vec){
    ArrayXd normal = Activacion::sigmoidea(vec).array();
    return normal*(1-normal);
}

VectorXd derivada_tanh(const VectorXd& vec){
    ArrayXd normal = Activacion::tanh(vec).array();
    return 1 - normal.pow(2);
}

VectorXd derivada_relu(const VectorXd& vec){
    VectorXd normal = Activacion::relu(vec);
    return normal.unaryExpr([](double valor){
        return valor != 0.0 ? 1.0: 0.0;
    });
}

VectorXd Activacion::relu(const VectorXd &vec)
{
    return vec.array().max(0.0);
}

VectorXd Activacion::sigmoidea(const VectorXd &vec)
{
    return (1 + (-vec.array()).exp()).pow(-1);
}

VectorXd Activacion::tanh(const VectorXd &vec)
{
    return vec.array().tanh();
}

Activacion::Activacion(Activacion_t _tipo)
    : tipo(_tipo) {}

VectorXd Activacion::operator()(const VectorXd &vec)
{
    switch (this->tipo)
    {
    case Activacion_t::sigmoidea:
        return Activacion::sigmoidea(vec);
    case Activacion_t::tanh:
        return Activacion::tanh(vec);
    case Activacion_t::relu:
        return Activacion::relu(vec);
    }
    return Activacion::relu(vec);
}