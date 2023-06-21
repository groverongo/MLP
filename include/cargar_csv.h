#ifndef CARGAR_CSV_H
#define CARGAR_CSV_H

#include<Eigen/Dense>
#include<string>
#include<fstream>
#include<vector>
#include<sstream>
#include<iostream>

using namespace Eigen;
using namespace std;

MatrixXd cargar_csv(const string& ruta);

void exportar_csv(const MatrixXd& matriz, const string& ruta);

#endif