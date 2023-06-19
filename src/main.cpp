#include <cstdio>
#include<iostream>
#include <Eigen/Dense>
#include "mlp.h"
#include"cargar_csv.h"
using namespace std;

void ejecutar(){
    MatrixXd datos = cargar_csv("./../../res/training.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);
    
    MLP mlp(X, Y);
    mlp.agregar_capa(Capa{(int) X.cols(), 100, Activacion::sigmoidea});
    mlp.agregar_capa(Capa{100, (int) Y.cols(), Activacion::sigmoidea});

    mlp.entrenar(100, 0.05);
    mlp.exportar();
    // mlp.cargar();
}


int main()
{
    try{
        ejecutar();
    }
    catch(const char* a){
        cout<<a;
    }
    
    return 0;
}
