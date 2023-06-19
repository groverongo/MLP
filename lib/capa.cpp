#include "capa.h"

Capa::Capa(int entrada, int salida, Activacion act)
    : pesos(MatrixXd::Random(entrada, salida) / 2), sesgo(VectorXd::Zero(salida)), tipo(act) {}

Capa::Capa(Activacion act)
    :tipo(act){}

VectorXd Capa::activacion(){
    switch (this->tipo)
    {
    case Activacion::sigmoidea:
        return this->sigmoidea();
    case Activacion::tanh:
        return this->tanh();
    case Activacion::relu:
        return this->relu();
    }
    return this->sigmoidea();
}

VectorXd Capa::derivada_activacion(){
    switch (this->tipo)
    {
    case Activacion::sigmoidea:
        return this->derivada_sigmoidea();
    case Activacion::tanh:
        return this->derivada_tanh();
    case Activacion::relu:
        return this->derivada_relu();
    }
    return this->derivada_sigmoidea();
    
}

VectorXd Capa::propagar(const VectorXd &vec){
    this->neto = this->pesos.transpose() * vec + this->sesgo;
    this->activado = this->activacion();
    this->derivada_activado = this->derivada_activacion();
    return this->activado;
}

VectorXd Capa::relu()
{
    return this->neto.array().max(0.0);
}

VectorXd Capa::sigmoidea()
{
    return (1 + (-this->neto.array()).exp()).pow(-1);
}

VectorXd Capa::tanh()
{
    return this->neto.array().tanh();
}

VectorXd Capa::derivada_sigmoidea(){
    return this->activado.array()*(1-this->activado.array());
}

VectorXd Capa::derivada_tanh(){
    return 1 - this->activado.array().pow(2);
}

VectorXd Capa::derivada_relu(){
    return this->activado.unaryExpr([](double valor){
        return valor != 0.0 ? 1.0: 0.0;
    });
}