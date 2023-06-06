#ifndef MLP_H
#define MLP_H

#include <Eigen/Dense>
#include<vector>
#include<list>
using namespace std;
using namespace Eigen;

struct Modulo{
    virtual VectorXd operator()(const VectorXd&) = 0;
};

enum class Activacion_t{
    sigmoidea,
    softmax,
    relu
};

struct Activacion: public Modulo{
    Activacion_t tipo;
    static VectorXd sigmoidea(const VectorXd&);
    static VectorXd softmax(const VectorXd&);
    static VectorXd relu(const VectorXd&);
    Activacion(Activacion_t _tipo);
    virtual VectorXd operator()(const VectorXd&) override;
};

struct Capa: public Modulo{
    // matriz de pesos
    MatrixXd pesos;
    // funcion de activacion
    Capa(int, int);

    virtual VectorXd operator()(const VectorXd&) override;
};

class MLP {
    int n;
    int hidden_size;
    list<Modulo*> modulos;


public:
    MLP();
    ~MLP();

    void agregar_modulo(Modulo*);
    VectorXd reenviar(const VectorXd&);
    // int forward();
    // void backward();
};


#endif //MLP_MLP_H
