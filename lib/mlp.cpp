#include "mlp.h"

VectorXd MLP::reenviar(const VectorXd& vec_x, const VectorXd& vec_y){
    VectorXd vec_h = vec_x;
    for(Modulo *m: this->modulos){
        vec_h = (*m)(vec_h);
    }
    return this->coste(vec_h, vec_y);
}

VectorXd MLP::coste(const VectorXd& vec_h, const VectorXd& vec_y){
    return (vec_y - vec_h).array().pow(2) / 2.0;
}

VectorXd MLP::derivada_coste(const VectorXd& vec_h, const VectorXd& vec_y){
    return vec_h - vec_y;
}

MLP::MLP() = default;

void MLP::agregar_modulo(Modulo* _modulo)
{
    this->modulos.push_back(_modulo);
}

MLP::~MLP(){
    for(Modulo* m: this->modulos)
        delete m;
}