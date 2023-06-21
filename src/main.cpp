#include <cstdio>
#include<iostream>
#include <Eigen/Dense>
#include "mlp.h"
#include"cargar_csv.h"
using namespace std;

void ejecutar_entrenamiento(){
    MatrixXd datos = cargar_csv("./../../res/training.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);

    MatrixXd datos_val = cargar_csv("./../../res/testing.csv");
    MatrixXd X_v = datos_val.leftCols(128);
    MatrixXd Y_v = datos_val.rightCols(datos_val.cols() - 128);
    
    MLP mlp(X, Y);
    mlp.agregar_validaciones(X_v, Y_v);
     /*ACA DEFINEN LAS CAPAS HIDDEN DEL SHEETS*/
    mlp.agregar_capa(Capa{(int) X.cols(), 50, Activacion::tanh});

    /*ESTA ES LA CAPA FINAL, LA DEJAN CON SIGMOIDEA, SOLO MODIFICAN EL ENTERO DE INPUTS*/
    mlp.agregar_capa(Capa{50, (int) Y.cols(), Activacion::sigmoidea});

    mlp.entrenar(50, 0.05);
    /* CADA VEZ QUE TERMINAN UN EXP, GUARDAN LO QUE ESTÁ EN DATA EN 
    UN FOLDER APARTE PORQUE SINO LO SOBREESCRIBE PARA SU SIGTE EXP */
    mlp.exportar(); 
}

void ejecutar_evaluacion(){
    MatrixXd datos = cargar_csv("./../../res/training.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);
    
    cout<<Y.rows()<<' '<<Y.cols()<<endl;

    string carpeta = "nuevo";
    MLP mlp(X, Y);
    mlp.cargar("../../data/"+carpeta+"/");
    MatrixXd predicciones = mlp.evaluar();
    exportar_csv(predicciones,"../../data/"+carpeta+"/"+carpeta+".csv");
}


int main()
{
    try{
        ejecutar_entrenamiento();
        // ejecutar_evaluacion();
    }
    catch(const char* a){
        cout<<a;
    }
    
    return 0;
}
