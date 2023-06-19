#include <cstdio>
#include<iostream>
#include <Eigen/Dense>
#include "mlp.h"
#include"cargar_csv.h"
using namespace std;

void prueba_CSV(){
    MatrixXd datos = cargar_csv("./../../res/datos.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);
    
    MLP mlp(X, Y);
    mlp.agregar_capa(Capa{(int) X.cols(), 50, Activacion::sigmoidea});
    mlp.agregar_capa(Capa{50, (int) Y.cols(), Activacion::sigmoidea});

    // mlp.entrenar(2, 0.5);
    mlp.cargar();
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
