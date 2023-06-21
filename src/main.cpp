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
    mlp.cargar("../../data/");
     /*ACA DEFINEN LAS CAPAS HIDDEN DEL SHEETS*/
    // mlp.agregar_capa(Capa{(int) X.cols(), 50, Activacion::tanh});
    // mlp.agregar_capa(Capa{50, 200, Activacion::tanh});

    /*ESTA ES LA CAPA FINAL, LA DEJAN CON SIGMOIDEA, SOLO MODIFICAN EL ENTERO DE INPUTS*/
    // mlp.agregar_capa(Capa{200, (int) Y.cols(), Activacion::sigmoidea});

    // mlp.entrenar(50, 0.05);
    /* CADA VEZ QUE TERMINAN UN EXP, GUARDAN LO QUE ESTÁ EN DATA EN 
    UN FOLDER APARTE PORQUE SINO LO SOBREESCRIBE PARA SU SIGTE EXP */
    // mlp.exportar(); 
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
