#include "mlp.h"

VectorXd MLP::propagacion_adelante(const VectorXd& vec_x, const VectorXd& vec_y){
    VectorXd vec_h = vec_x;
    for(Capa &c: this->capas){
        vec_h = c.propagar(vec_x);
    }
    this->salida = vec_h;
    return this->coste(vec_h, vec_y);
}

void MLP::propagacion_atras(){
    
}

// C = (a^(L) - y)^2
VectorXd MLP::coste(const VectorXd& vec_h, const VectorXd& vec_y){
    return (vec_y - vec_h).array().pow(2);
}

// C = 2*(a^(L) - y)
VectorXd MLP::derivada_coste(const VectorXd& vec_h, const VectorXd& vec_y){
    return 2*(vec_h - vec_y);
}

MLP::MLP() = default;

void MLP::agregar_capa(Capa capa)
{
    this->capas.push_back(capa);
}

MLP::~MLP(){}