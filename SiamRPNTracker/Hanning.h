//
// Created by jjyjb on 22-10-31.
//

#ifndef CPPTEST_HANNING_H
#define CPPTEST_HANNING_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


class Hanning {
public:
    static VectorXf hanning(size_t n, bool flag = true);

//private:
    static VectorXf sym_hanning(size_t n);

    static VectorXf calc_hanning(size_t m, size_t n);
};


#endif //CPPTEST_HANNING_H

//0.0202535
//0.0793732
//0.17257
//0.292292
//0.428843
//0.571157
//0.707708
//0.82743
//0.920627
//0.979746
//1
//0.979746
//0.920627
//0.82743
//0.707708
//0.571157
//0.428843
//0.292292
//0.17257
//0.0793732
//0.0202535