#include <iostream>
#include <Eigen/Dense>
#include"mlp.h"

int main() {
    Eigen::VectorXd a(4);
    a << 2, 2, 4, 9;

    Eigen::VectorXd result = a.array().exp();

    std::cout << "Result: " << result << std::endl;
    std::cout << "Result: " << sigmoidea(a) << std::endl;

    return 0;
}

