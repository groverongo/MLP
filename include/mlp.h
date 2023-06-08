#ifndef MLP_H
#define MLP_H

#include <Eigen/Dense>
#include<vector>
#include<list>
#include"modulo/modulo.h"
#include"modulo/capa.h"
#include"modulo/activacion.h"
using namespace std;
using namespace Eigen;

class MLP {
    int n;
    int hidden_size;
    list<Modulo*> modulos;


public:
    MLP();
    ~MLP();

    void agregar_modulo(Modulo*);
    VectorXd reenviar(const VectorXd&);
    // int forward();
    // void backward();
};


#endif //MLP_MLP_H
