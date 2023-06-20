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
    // cout<<X.rows()<<' '<<X.cols()<<endl;
    // cout<<Y.rows()<<' '<<Y.cols()<<endl;
    mlp.agregar_capa(Capa{(int) X.cols(), 200, Activacion::sigmoidea});
    mlp.agregar_capa(Capa{200, (int) Y.cols(), Activacion::sigmoidea});

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
