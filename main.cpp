#include <cstdio>
#include<iostream>
#include <Eigen/Dense>
#include "mlp.h"
using namespace std;

void prueba_XOR(){
    MatrixXd X(4, 2);
    X << 0,0, 1,0, 0,1, 1,1;
    MatrixXd Y(4, 2); // Falso, Verdadero
    Y << 1,0, 0,1, 0,1, 1,0;

    MLP mlp;
    mlp.agregar_modulo(new Capa{2,2});
    mlp.agregar_modulo(new Activacion{Activacion_t::sigmoidea});
    mlp.agregar_modulo(new Capa{2,2});
    mlp.agregar_modulo(new Activacion{Activacion_t::sigmoidea});

    cout<<mlp.reenviar(X.row(0)).transpose();
}

int main()
{
    prueba_XOR();
    /* MLP mlp;
    Capa* m1 = new Capa{4, 10};
    Activacion* a1 = new Activacion{Activacion_t::sigmoidea};
    mlp.agregar_modulo(m1);
    mlp.agregar_modulo(a1);
    VectorXd vec(4);
    vec<<1,2,3,4;

    cout<<vec.transpose()<<endl;
    cout<<m1->sesgo.rows()<<'\t'<<m1->sesgo.cols()<<"\n\n";

    cout<<m1->pesos<<"\n\n";
    cout<<mlp.reenviar(vec).transpose()<<"\n\n";
    cout<<vec.transpose().array() +3; */
    // mlp.agregar_modulo(new Activacion{Activacion_t::sigmoidea});
    /* VectorXd vec(4);
    vec.Random();
    // cout<<vec.transpose()<<endl;
    mlp.reenviar(vec);
    printf("FIN\n"); */
    return 0;
}
