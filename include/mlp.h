#ifndef MLP_H
#define MLP_H

#include <Eigen/Dense>
#include<vector>
#include<list>
#include<iostream>
#include<fstream>
#include"capa.h"
#include<cstdio>
#include<functional>
using namespace std;
using namespace Eigen;

struct MLP {
    int n;
    int hidden_size;
    vector<Capa> capas;
    VectorXd salida;
    MatrixXd X, Y;

    /**
     * @brief Calculo recursivo para las derivadas de los pesos de las capas ocultas
     * @param capa_actual: parametro de recursion, inicia en la ultima capa y disminuye, utilizado para indexar
     * @param capa_peso: en que capa el peso se encuentra, no aplica para pesos de la capa de salida
     * @param neurona_destino: hacia que neurona el peso se dirije
     * @return Un vector del calculo actual
     * @note El caso base ocurre cuando capa_actual = capa_peso
     * */
    VectorXd derivada_recursiva(const int, const int, const int);
    void derivadas_oculta(const int, const int, const int, MatrixXd&, VectorXd&);
    void derivadas_salida(const int, MatrixXd&, VectorXd&);
    double coste(const VectorXd&, const VectorXd&);
    /**
     * @brief Obtener las derivadas de los pesos y una derivada del sesgo de la capa oculta L que se dirigen hacia una neurona de esa capa
     * @param indice_capa: Capa en la cual se quiere hallar la derivada
     * @param neurona_destino: Neurona de la capa L a la cual los pesos y el sesgo se dirigen
     * @param indice_dato: La fila de los datos que se refieren (Xi y Yi)
     * @param derivada_pesos: Referencia para poblar las derivadas de las pesos
     * @param derivada_sesgo: Referencia para poblar las derivadas del sesgo
    */
    void entropia_derivadas_oculta(const int, const int, const int, MatrixXd&, VectorXd&);
    /**
     * @brief Obtener derivadas de los pesos y sesgos de la capa de salida
     * @param fila: Fila de data Y y X
     * @param derivada_pesos: Referencia para poblar los pesos
     * @param derivada_sesgo: Referencia para poblar los sesgos
    */
    void entropia_derivadas_salida(const int, MatrixXd&, VectorXd&);
    /**
     * @brief Obtener la perdida de la entropia cruzada
     * @param vec_h: Vector de predicciones de la capa de salida
     * @param vec_y: Vector de la data Yi
     * @return Valor de la perdida
    */
    double entropia(const VectorXd&, const VectorXd&);
    /**
     * @brief Obtener el vector de valores activados de la capa o la fila Xi
     * @param indice_capa: Identificador de la capa
     * @param fila: En caso la capa sea la -1, se accede a la fila Xi
     * @return Vector copia
    */
    VectorXd vector_activacion(const int, const int=-1);
    const VectorXd& vector_derivada_activacion(const int);
    MatrixXd& matriz_pesos(const int);
    VectorXd& vector_sesgo(const int);

    MLP(MatrixXd, MatrixXd);
    ~MLP();

    void agregar_capa(Capa);
    double propagacion_adelante(const int);
    void propagacion_atras(const int, const double);

    void entrenar(const int, const double);
    MatrixXd evaluar();

    VectorXd softmax(const VectorXd&);

    void exportar();
    void cargar();
};


#endif //MLP_MLP_H
