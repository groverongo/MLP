//
// Created by Sebastian Knell on 2/06/23.
//

#ifndef MLP_MLP_H
#define MLP_MLP_H

#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

enum activacion{
    a_softmax,
    a_sigmoidea,
    a_relu
};

VectorXd sigmoidea(const VectorXd&);
VectorXd sofmax(const VectorXd&);
VectorXd relu(const VectorXd&);

struct Layer {
    // vector neto
    VectorXd salida;
    // matriz de pesos
    MatrixXd pesos;
    // funcion de activacion
    Layer(int _neuronas, activacion f_activacion);
};

class MLP {
    int n;
    int hidden_size;
public:
    int forward();
};


#endif //MLP_MLP_H
