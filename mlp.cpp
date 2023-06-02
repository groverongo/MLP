//
// Created by Sebastian Knell on 2/06/23.
//

#include "mlp.h"

VectorXd relu(const VectorXd &vec) {
    return vec.array().max(0.0);
}

VectorXd sigmoidea(const VectorXd &vec) {
    Eigen::ArrayXd result = (1+(vec.array().exp())).array().pow(-1);
    return Eigen::VectorXd::Map(result.data(), result.size());
}

VectorXd softmax(const VectorXd &vec){
    VectorXd num = vec.array().exp();
    return (num/num.sum());
}

Layer::Layer(int _neuronas, activacion f_activacion) {
    this->pesos = MatrixXd::Random(_neuronas,_neuronas);
}

