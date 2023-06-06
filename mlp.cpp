#include "mlp.h"

VectorXd Capa::operator()(const VectorXd &vec)
{
    return pesos * vec;
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
    ArrayXd x = vec.array();
    return (1 + (-x).exp()).pow(-1);
}

VectorXd Activacion::softmax(const VectorXd &vec)
{
    VectorXd num = vec.array().exp();
    return (num / num.sum());
}

Activacion::Activacion(Activacion_t _tipo)
    : tipo(_tipo) {}

VectorXd Activacion::operator()(const VectorXd &vec)
{
    switch (this->tipo)
    {
    case Activacion_t::sigmoidea:
        return Activacion::sigmoidea(vec);
    case Activacion_t::softmax:
        return Activacion::softmax(vec);
    case Activacion_t::relu:
        return Activacion::relu(vec);
    }
    return Activacion::relu(vec);
}

Capa::Capa(int entrada, int salida)
    : pesos(MatrixXd::Random(salida, entrada)) {}

MLP::MLP() = default;

void MLP::agregar_modulo(Modulo* _modulo)
{
    this->modulos.push_back(_modulo);
}

MLP::~MLP(){
    for(Modulo* m: this->modulos)
        delete m;
}