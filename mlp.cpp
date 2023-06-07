#include "mlp.h"

VectorXd Capa::operator()(const VectorXd &vec)
{
    return this->pesos.transpose() * vec + this->sesgo;
}

VectorXd Activacion::relu(const VectorXd &vec)
{
    return vec.array().max(0.0);
}

VectorXd MLP::reenviar(const VectorXd& vec){
    VectorXd result = vec;
    for(Modulo *m: this->modulos){
        result = (*m)(result);
    }
    return result;
}

/**
 * (1 + e**(-x))**-1
 */
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

Capa::Capa(int entrada, int salida)
    : pesos(MatrixXd::Random(entrada, salida)), sesgo(VectorXd::Random(salida)) {}

MLP::MLP() = default;

void MLP::agregar_modulo(Modulo* _modulo)
{
    this->modulos.push_back(_modulo);
}

MLP::~MLP(){
    for(Modulo* m: this->modulos)
        delete m;
}