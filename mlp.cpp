//
// Created by Sebastian Knell on 2/06/23.
//

#include "mlp.h"

VectorXd relu(const VectorXd &vec) {
    return vec.array().max(0.0);
}

/**
 * (1 + e**(-x))**-1
*/
VectorXd sigmoidea(const VectorXd &vec) {
    ArrayXd x = vec.array();
    return (1 + (-x).exp()).pow(-1);
}

VectorXd softmax(const VectorXd &vec){
    VectorXd num = vec.array().exp();
    return (num/num.sum());
}

Layer::Layer(int _neuronas, activacion f_activacion) {
    this->pesos = MatrixXd::Random(_neuronas,_neuronas);
}

