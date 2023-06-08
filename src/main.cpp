#include <cstdio>
#include<iostream>
#include <Eigen/Dense>
#include "mlp.h"
#include"cargar_csv.h"
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

void prueba_CSV(){
    MatrixXd datos = cargar_csv("./../../res/datos.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);
    cout<<Y.row(0);
}

int main()
{
    try{
        prueba_CSV();
    }
    catch(const char* a){
        cout<<a;
    }
    
    return 0;
}
