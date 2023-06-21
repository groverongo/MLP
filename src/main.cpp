#include <cstdio>
#include<iostream>
#include <Eigen/Dense>
#include "mlp.h"
#include"cargar_csv.h"
using namespace std;

void ejecutar_entrenamiento(){
    // NO TOCAR
    MatrixXd datos = cargar_csv("./../../res/training.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);
    MatrixXd datos_val = cargar_csv("./../../res/testing.csv");
    MatrixXd X_val = datos.leftCols(128);
    MatrixXd Y_val = datos.rightCols(datos.cols() - 128);
    
    MLP mlp(X, Y);
    mlp.asignar_datos_validacion(X_val, Y_val);

    // FIN DE NO TOCAR

     /*ACA DEFINEN LAS CAPAS HIDDEN DEL SHEETS*/
    mlp.agregar_capa(Capa{(int) X.cols(), 50, Activacion::tanh});
    // mlp.agregar_capa(Capa{50, 200, Activacion::tanh});

    /*ESTA ES LA CAPA FINAL, LA DEJAN CON SIGMOIDEA, SOLO MODIFICAN EL ENTERO DE INPUTS*/
    mlp.agregar_capa(Capa{50,  (int) Y.cols(), Activacion::sigmoidea});

    mlp.entrenar(50, 0.05);
    /* CADA VEZ QUE TERMINAN UN EXP, GUARDAN LO QUE ESTÁ EN DATA EN 
    UN FOLDER APARTE PORQUE SINO LO SOBREESCRIBE PARA SU SIGTE EXP */
    mlp.exportar(); 
}

void ejecutar_evaluacion(string carpeta){
    MatrixXd datos = cargar_csv("./../../res/testing.csv");
    MatrixXd X = datos.leftCols(128);
    MatrixXd Y = datos.rightCols(datos.cols() - 128);
    
    cout<<Y.rows()<<' '<<Y.cols()<<endl;

    MLP mlp(X, Y);
    mlp.cargar("../../data/"+carpeta+"/");
    MatrixXd predicciones = mlp.evaluar();
    exportar_csv(predicciones,"../../data/"+carpeta+"/"+carpeta+".csv");
}


int main()
{
    try{
        // Solo ejecutar el entrenamiento, la evaluación la haré cuando todo esté hecho
        ejecutar_entrenamiento();
        // ejecutar_evaluacion("C1_50_T");
    }
    catch(const char* a){
        cout<<a;
    }
    
    return 0;
}
