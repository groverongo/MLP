#include "mlp.h"

/**
 * @brief Calculo recursivo para las derivadas de los pesos de las capas ocultas
 * @param capa_actual: parametro de recursion, inicia en la ultima capa y disminuye, utilizado para indexar
 * @param capa_peso: en que capa el peso se encuentra, no aplica para pesos de la capa de salida
 * @param neurona_destino: hacia que neurona el peso se dirije
 * @return Un vector del calculo actual
 * @note El caso base ocurre cuando capa_actual = capa_peso
 * */
VectorXd MLP::producto_recursivo(const int capa_actual, const int capa_peso, const int neurona_destino){
    const Capa& capa_objetivo = this->capas[capa_actual];
    if(capa_actual == capa_peso){
        return capa_objetivo.derivada_activado.cwiseProduct(capa_objetivo.pesos.transpose().col(neurona_destino));
    }
    return capa_objetivo.derivada_activado.cwiseProduct(capa_objetivo.pesos.transpose() * this->producto_recursivo(capa_actual-1, capa_peso, neurona_destino));
}

VectorXd MLP::propagacion_adelante(const VectorXd& vec_x, const VectorXd& vec_y){
    VectorXd vec_h = vec_x;
    for(Capa &c: this->capas){
        vec_h = c.propagar(vec_x);
    }
    this->salida = vec_h;
    return VectorXd();
    // return this->coste(vec_h, vec_y);
}

// void MLP::propagacion_atras(){

// }

void entrenar(const MatrixXd&, const MatrixXd&){
    
}

const VectorXd& MLP::vector_activacion(const int indice_capa){
    return this->capas.at(indice_capa).activado;
}
const VectorXd& MLP::vector_derivada_activacion(const int indice_capa){
    return this->capas.at(indice_capa).derivada_activado;
}
const MatrixXd& MLP::matriz_pesos(const int indice_capa){
    return this->capas.at(indice_capa).pesos;
}

double MLP::coste(const VectorXd& vec_h, const VectorXd& vec_y){
    return ((vec_h - vec_y).array().pow(2)/2).mean();
}

/* // C = 2^{-1}(a^(L) - y)^2
VectorXd MLP::coste(const VectorXd& vec_h, const VectorXd& vec_y){
    return (vec_y - vec_h).array().pow(2);
}
 */
// C = 2*(a^(L) - y)
VectorXd MLP::derivada_coste(const VectorXd& vec_h, const VectorXd& vec_y){
    return (vec_h - vec_y);
}

MLP::MLP() = default;

void MLP::agregar_capa(Capa capa)
{
    this->capas.push_back(capa);
}

MLP::~MLP(){}