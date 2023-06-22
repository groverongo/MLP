# PERCEPTRON DE MULTIPLES CAPAS

## Descripción básica

Este repositorio contiene el código para:

- La implementación de un perceptrón de múltiples capas
- La visualización de los resultados tras la ejecución del MLP

## Requisitos para la ejecución
### Para la ejecución de la implementación en C++

- Cmake
- Dependencia Eigen 3

### Para ver los gráficos y tablas de python

- matplotlib
- numpy
- pandas
- sklearn

## Ejecución
### Implementación en C++

Lo que hace ```/src/main.cpp``` es obtener las predicciones con *softmax* de todos los modelos de redes neuronales definidos con el nombre de las carpetas dentro de ```/data``` que contienen sus pesos y sesgos. 

Ejemplo:  
Las predicciones *softmax* para la carpeta ```/data/C1_50_R``` se encontrará dentro de la carpeta con el nombre ```C1_50_R.csv```.

Adicionalmente, se obtendrá un archivo ```/data/preds.csv``` el cual contendrá las predicciones del mejor modelo ```/C1_50_T```.

### Visualización de gráficos en Python

Para facilitar el manipulamiento de datos, utilizamos un cuaderno **Jupyter** llamado ```/config/main.ipynb```. Lo que se debe hacer es entrar al cuaderno y ejecutar todas las casillas. Lo que generará será lo siguiente:

1. Una lista con los nombres de las carpetas de los modelos.
2. Un archivo csv con los valores F1, exhaustividad y precisión.
3. Graficas de las perdidas por cada modelo en la carpeta ```/config/graficas```