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

    MLP mlp(X, Y);
    mlp.agregar_capa(Capa{2,2, Activacion::sigmoidea});

    mlp.entrenar(1000, 0.01);
    cout<<mlp.evaluar()<<endl;
}

void prueba_CSV(){
    MatrixXd datos = cargar_csv("./../../res/datos.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);
    
    MLP mlp(X, Y);
    mlp.agregar_capa(Capa{(int) X.cols(), 50, Activacion::sigmoidea});
    mlp.agregar_capa(Capa{50, (int) Y.cols(), Activacion::sigmoidea});

    /* cout<<X.cols()<<' '<<Y.cols()<<endl;
    for(int i =0; i<mlp.capas.size(); i++){
        cout<<mlp.capas[i].pesos.rows()<<' '<<mlp.capas[i].pesos.cols()<<endl;
    } */

    mlp.entrenar(1000, 0.5);
}


int main()
{
    try{
        // prueba_XOR();
        prueba_CSV();
    }
    catch(const char* a){
        cout<<a;
    }
    
    return 0;
}
