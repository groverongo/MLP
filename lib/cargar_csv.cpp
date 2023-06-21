#include"cargar_csv.h"

MatrixXd cargar_csv(const string& ruta){
    ifstream archivo(ruta);
    if(!archivo.is_open())
        throw("No se dio la ruta correcta del archivo");

    vector<vector<double>> datos;
    string linea;

    while(getline(archivo, linea)){
        stringstream linea_s(linea);
        vector<double> fila;
        string celda;
        while(getline(linea_s, celda, ',')){
            fila.push_back(stod(celda));
        }
        datos.push_back(fila);
    }

    int filas = datos.size(), columnas = datos.front().size();

    MatrixXd matriz(filas, columnas);
    for(int i = 0; i<filas; i++){
        for(int j = 0; j<columnas; j++){
            matriz(i, j) = datos[i][j];
        }
    }

    return matriz;
}

void exportar_csv(const MatrixXd& matriz, const string& ruta){
    ofstream archivo(ruta, ios::out);
    for(int fila = 0; fila<matriz.rows(); fila++){
        for(int columna = 0; columna<matriz.cols(); columna++){
            archivo<<matriz(fila, columna);
            if(columna != matriz.cols() -1)
                archivo<<',';
        }
        archivo<<'\n';
    }
    archivo.close();
}