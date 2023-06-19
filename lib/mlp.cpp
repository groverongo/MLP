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

VectorXd MLP::propagacion_adelante(const int fila){
    VectorXd vec_h = X.row(fila);
    for(Capa &c: this->capas){
        vec_h = c.propagar(vec_h);
    }
    this->salida = vec_h;
    return VectorXd();
    // return this->coste(vec_h, vec_y);
}

MatrixXd MLP::derivada_pesos_salida(const int fila){
    int indice_salida = this->capas.size()-1;
    VectorXd diferencia_y = this->vector_activacion(indice_salida) - this->Y.row(fila);
    VectorXd producto_elemento = diferencia_y.cwiseProduct(this->vector_derivada_activacion(indice_salida));
    return (vector_activacion(indice_salida) * producto_elemento.transpose()) / this->Y.cols();
}

void MLP::propagacion_atras(const int fila, const double ratio_aprendizaje) {
    // hidden output
    int n = this->capas.size();
    MatrixXd derivada_pesos = this->derivada_pesos_salida(fila);
    this->matriz_pesos(n-1) -= ratio_aprendizaje * derivada_pesos;

    auto output = this->capas[n-1];
    auto hidden = this->capas[n-2];
    auto ultimo_ro = (this->derivada_coste(this->salida, this->Y.row(fila)).cwiseProduct(output.derivada_activacion()));
    auto derivada = hidden.activado * ultimo_ro.transpose();
    // output.pesos -= ratio_aprendizaje * derivada;
    output.sesgo -= ratio_aprendizaje*ultimo_ro;

    // hidden hidden
    for (int l = n-2; l >= 0; l--) {
        // Luego se debe trasponer
        derivada_pesos = MatrixXd(this->matriz_pesos(l).rows(), this->matriz_pesos(l).cols());

        for(int i = 0; i<derivada_pesos.cols(); i++){
            derivada_pesos.col(i) = this->derivada_coste_pesos(l, i, fila);
        }

        this->matriz_pesos(l) -= ratio_aprendizaje * derivada_pesos; 
    }
}

VectorXd MLP::derivada_coste_pesos(const int capa, const int neurona_destino, const int fila){
    int ultima_capa = this->capas.size()-1;
    VectorXd vector_y = this->Y.row(fila);
    VectorXd vector_h = this->vector_activacion(ultima_capa, fila);
    VectorXd diferencia_y = vector_h - vector_y;
    VectorXd vector_recursion = producto_recursivo(ultima_capa, capa, neurona_destino);
    
    return vector_recursion.cwiseProduct(diferencia_y).mean() * vector_derivada_activacion(capa).coeff(neurona_destino) * vector_activacion(capa-1, fila);
}


void MLP::entrenar(const int epocas, const double ratio_aprendizaje) {
    auto n = X.rows();
    for (int epoca = 0; epoca < epocas; epoca++) {
        for (int i = 0; i < n; ++i) {
            this->propagacion_adelante(i);
            this->propagacion_atras(i, ratio_aprendizaje);
        }
    }
}

const VectorXd& MLP::vector_activacion(const int indice_capa, const int fila=-1){
    if(indice_capa<0)
        return X.row(fila);
    return this->capas.at(indice_capa).activado;
}
const VectorXd& MLP::vector_derivada_activacion(const int indice_capa){
    return this->capas.at(indice_capa).derivada_activado;
}
MatrixXd& MLP::matriz_pesos(const int indice_capa){
    return this->capas.at(indice_capa).pesos;
}
VectorXd& MLP::vector_sesgo(const int indice_capa){
    return this->capas.at(indice_capa).sesgo;
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

MLP::MLP(MatrixXd X, MatrixXd Y): X(X), Y(Y) {};

void MLP::agregar_capa(Capa capa)
{
    this->capas.push_back(capa);
}

MLP::~MLP(){}