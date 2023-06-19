#ifndef CAPA_H
#define CAPA_H

#include<Eigen/Dense>
#include<string>

using namespace Eigen;

enum class Activacion{
    sigmoidea,
    tanh,
    relu
};

struct Capa
{
    // Valores asignados al agregar la capa
    // Tipo de activacion
    Activacion tipo;
    // matriz de pesos
    VectorXd sesgo;
    // funcion de activacion
    MatrixXd pesos;

    // Valores asignados tras propagar
    // Valor Neto
    VectorXd neto;
    // Valor Activacion;
    VectorXd activado;
    // Valor de la derivada de Activacion;
    VectorXd derivada_activado;

    // funcion activacion
    VectorXd activacion();
    // funcion derivada activacion
    VectorXd derivada_activacion();


    // funcion sigmoidea
    VectorXd sigmoidea();
    // funcion tangente hiperbolico
    VectorXd tanh();
    // funcion relu
    VectorXd relu();

    // funcion sigmoidea
    VectorXd derivada_sigmoidea();
    // funcion tangente hiperbolico
    VectorXd derivada_tanh();
    // funcion relu
    VectorXd derivada_relu();

    // Iniciar Capa con entradas, salidas y funcion de activacion
    Capa(int, int, Activacion);
    Capa(Activacion);

    // Evaluar la salida con el calculo del valor neto y activado
    VectorXd propagar(const VectorXd&);
};

#endif