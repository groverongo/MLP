#include "capa.h"

Capa::Capa(int entrada, int salida, Activacion act)
    : pesos(MatrixXd::Random(entrada, salida) / 2), sesgo(VectorXd::Zero(salida)), tipo(act) {}

Capa::Capa(Activacion act)
    :tipo(act){}

VectorXd Capa::activacion(const VectorXd& vec){
    switch (this->tipo)
    {
    case Activacion::sigmoidea:
        return this->sigmoidea(vec);
    case Activacion::tanh:
        return this->tanh(vec);
    case Activacion::relu:
        return this->relu(vec);
    }
    return this->sigmoidea(vec);
}

VectorXd Capa::derivada_activacion(const VectorXd& vec){
    switch (this->tipo)
    {
    case Activacion::sigmoidea:
        return this->derivada_sigmoidea(vec);
    case Activacion::tanh:
        return this->derivada_tanh(vec);
    case Activacion::relu:
        return this->derivada_relu(vec);
    }
    return this->derivada_sigmoidea(vec);
    
}

VectorXd Capa::propagar(const VectorXd &vec){
    this->neto = this->pesos.transpose() * vec + this->sesgo;
    this->activado = this->activacion(this->neto);
    this->derivada_activado = this->derivada_activacion(this->activado);
    return this->activado;
}

VectorXd Capa::relu(const VectorXd& vec)
{
    return vec.array().max(0.0);
}

VectorXd Capa::sigmoidea(const VectorXd& vec)
{
    return (1 + (-vec.array()).exp()).pow(-1);
}

VectorXd Capa::tanh(const VectorXd& vec)
{
    return vec.array().tanh();
}

VectorXd Capa::derivada_sigmoidea(const VectorXd& vec){
    return vec.array()*(1-this->activado.array());
}

VectorXd Capa::derivada_tanh(const VectorXd& vec){
    return 1 - vec.array().pow(2);
}

VectorXd Capa::derivada_relu(const VectorXd& vec){
    return vec.unaryExpr([](double valor){
        return valor != 0.0 ? 1.0: 0.0;
    });
}