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

double MLP::propagacion_adelante(const int fila){
    VectorXd vec_h = X.row(fila).transpose();
    for(Capa &c: this->capas){
        vec_h = c.propagar(vec_h);
    }
    this->salida = vec_h;
    return this->coste(vec_h, Y.row(fila));
}

void MLP::cargar(){
    int l;
    ifstream cantidad("../../data/cantidad.txt");
    cantidad>>l;
    cantidad.close();
    ifstream archivo_act("../../data/activacion.csv");
    archivo_act.close();
    for(int i = 0; i<l; i++){
        
    }
}

void MLP::exportar(){
    int l = 0;

    ofstream archivo_act("../../data/activacion.csv", ios::out);
    for(const Capa& c: this->capas){

        switch(c.tipo){
            case Activacion::sigmoidea:
                archivo_act<<0<<'\n';
                break;
            case Activacion::tanh:
                archivo_act<<1<<'\n';
                break;
            case Activacion::relu:
                archivo_act<<2<<'\n';
                break;
        }

        ofstream archivo("../../data/pesos_capa_"+to_string(l)+".csv", ios::out);
        for(int fila = 0; fila<c.pesos.rows(); fila++){
            for(int columna = 0; columna<c.pesos.cols(); columna++){
                archivo<<c.pesos(fila, columna);
                if(columna != c.pesos.cols() -1)
                    archivo<<',';
            }
            archivo<<'\n';
        }
        archivo.close();
        archivo.open("../../data/sesgo_capa_"+to_string(l)+".csv", ios::out);
        for(int i = 0; i<c.sesgo.size(); i++){
            archivo<<c.sesgo[i]<<'\n';
        }
        archivo.close();
        l++;
    }
    archivo_act.close();

    ofstream cantidad("../../data/cantidad.txt");
    cantidad<<l;
    cantidad.close();
}

void MLP::derivadas_salida(const int fila, MatrixXd& derivada_pesos, VectorXd& derivada_sesgo){
    int indice_salida = this->capas.size()-1;
    VectorXd diferencia_y = this->vector_activacion(indice_salida) - this->Y.row(fila).transpose();
    VectorXd producto_elemento = diferencia_y.cwiseProduct(this->vector_derivada_activacion(indice_salida));
    
    derivada_sesgo = producto_elemento / this->Y.cols();
    derivada_pesos = vector_activacion(indice_salida-1) * derivada_sesgo.transpose();
}

void MLP::propagacion_atras(const int fila, const double ratio_aprendizaje) {
    // hidden output
    int n = this->capas.size();
    MatrixXd derivada_pesos;
    VectorXd derivada_sesgo;
    this->derivadas_salida(fila, derivada_pesos, derivada_sesgo);
    this->matriz_pesos(n-1) -= ratio_aprendizaje * derivada_pesos;
    this->vector_sesgo(n-1) -= ratio_aprendizaje * derivada_sesgo;
    // hidden hidden
    for (int l = n-2; l >= 0; l--) {
        // Luego se debe trasponer
        derivada_pesos = MatrixXd(this->matriz_pesos(l).rows(), this->matriz_pesos(l).cols());
        derivada_sesgo = VectorXd(this->vector_sesgo(l).size());
        // cout<<"NONONO"<<endl;
        for(int i = 0; i<derivada_pesos.cols(); i++){
            this->derivadas_oculta(l, i, fila, derivada_pesos, derivada_sesgo);
            // cout<<"Iteracion "<<derivada_pesos.cols()<<endl;
        }
        // cout<<"NONONO"<<endl;

        this->matriz_pesos(l) -= ratio_aprendizaje * derivada_pesos; 
        this->vector_sesgo(l) -= ratio_aprendizaje * derivada_sesgo;
    }
}

void MLP::derivadas_oculta(const int capa, const int neurona_destino, const int fila, MatrixXd& derivada_pesos, VectorXd& derivada_sesgo){
    int ultima_capa = this->capas.size()-1;
    VectorXd vector_y = this->Y.row(fila);
    VectorXd vector_h = this->vector_activacion(ultima_capa, fila);
    VectorXd diferencia_y = vector_h - vector_y;
    // cout<<capa<<endl;
    // cout<<"Recursivo Inicio"<<endl;
    VectorXd vector_recursion = producto_recursivo(ultima_capa, capa, neurona_destino);
    // cout<<"Recursivo Fin"<<endl;
    
    // cout<<"Actualiza sesgo "<<capa<<" "<<neurona_destino<<endl;
    derivada_sesgo[neurona_destino] = vector_recursion.cwiseProduct(diferencia_y).mean() * vector_derivada_activacion(capa).coeff(neurona_destino);
    // cout<<"Actualiza pesos"<<endl;
    derivada_pesos.col(neurona_destino) = derivada_sesgo[neurona_destino] * vector_activacion(capa-1, fila);
}


void MLP::entrenar(const int epocas, const double ratio_aprendizaje) {
    int n = X.rows();
    for (int epoca = 0; epoca < epocas; epoca++) {
        printf("Epoca N-%d -> ", epoca+1);
        VectorXd perdidas(n);
        for (int i = 0; i < n; ++i) {
            double perdida = this->propagacion_adelante(i);
            perdidas[i] = perdida;
            this->propagacion_atras(i, ratio_aprendizaje);
        }
        printf("Perdida: %lf\n", perdidas.mean());
    }
}

MatrixXd MLP::evaluar(){
    int filas = X.rows();
    MatrixXd resultado(this->Y.rows(), this->Y.cols());
    for(int i = 0; i<filas; i++){
        VectorXd vec_h = X.row(i);
        for(Capa &c: this->capas){
            vec_h = c.propagar(vec_h);
        }
        resultado.row(i) = vec_h.transpose();
    }
    return resultado;
}

VectorXd MLP::vector_activacion(const int indice_capa, const int fila){
    if(indice_capa<0)
        return X.row(fila).transpose();
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

MLP::MLP(MatrixXd X, MatrixXd Y): X(X), Y(Y) {};

void MLP::agregar_capa(Capa capa)
{
    this->capas.push_back(capa);
}

MLP::~MLP(){}