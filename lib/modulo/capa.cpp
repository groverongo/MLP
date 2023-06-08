#include "modulo/capa.h"

Capa::Capa(int entrada, int salida)
    : pesos(MatrixXd::Random(entrada, salida)), sesgo(VectorXd::Zero(salida)) {}

VectorXd Capa::operator()(const VectorXd &vec){
    return this->pesos.transpose() * vec + this->sesgo;
}