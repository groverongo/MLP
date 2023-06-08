#include "mlp.h"

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


MLP::MLP() = default;

void MLP::agregar_modulo(Modulo* _modulo)
{
    this->modulos.push_back(_modulo);
}

MLP::~MLP(){
    for(Modulo* m: this->modulos)
        delete m;
}