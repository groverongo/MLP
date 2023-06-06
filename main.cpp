#include <cstdio>
#include<iostream>
#include <Eigen/Dense>
#include "mlp.h"
using namespace std;
int main()
{
    MLP mlp;
    Capa* m1 = new Capa{4, 10};
    Activacion* a1 = new Activacion{Activacion_t::sigmoidea};
    mlp.agregar_modulo(m1);
    mlp.agregar_modulo(a1);
    VectorXd vec(4);
    vec<<1,2,3,4;

    cout<<vec.transpose()<<endl;
    cout<<m1->pesos<<endl;
    cout<<mlp.reenviar(vec).transpose();
    // mlp.agregar_modulo(new Activacion{Activacion_t::sigmoidea});
    /* VectorXd vec(4);
    vec.Random();
    // cout<<vec.transpose()<<endl;
    mlp.reenviar(vec);
    printf("FIN\n"); */
    return 0;
}
