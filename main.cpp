#include <iostream>
#include "mlp.h"

int main() {
//    Layer layer(10, a_relu);
//    std::cout << layer.pesos;
    cout << sigmoidea(VectorXd::Random(10));
}