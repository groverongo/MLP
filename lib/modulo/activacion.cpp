#include "modulo/activacion.h"

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