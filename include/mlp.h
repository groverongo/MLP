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
private:
    int n;
    int hidden_size;
    list<Modulo*> modulos;

protected:
    VectorXd coste(const VectorXd&, const VectorXd&);
    VectorXd derivada_coste(const VectorXd&, const VectorXd&);

public:
    MLP();
    ~MLP();

    void agregar_modulo(Modulo*);
    VectorXd reenviar(const VectorXd&, const VectorXd&);
    // int forward();
    // void backward();
};


#endif //MLP_MLP_H
