#ifndef MODULO_H
#define MODULO_H

#include<Eigen/Dense>

using namespace Eigen;

struct Modulo{
    virtual VectorXd operator()(const VectorXd&) = 0;
    
};

#endif
