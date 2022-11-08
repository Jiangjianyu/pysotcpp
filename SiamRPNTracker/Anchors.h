//
// Created by jjyjb on 22-11-1.
//

#ifndef CPPTEST_ANCHORS_H
#define CPPTEST_ANCHORS_H

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;

class Anchors {
public:

    Anchors(int &stride, std::vector<float> &ratios, std::vector<float> &scales, int image_center = 0, int size = 0);

    bool generate_all_anchors(int im_c, int size_tt);

public:
    void generate_anchors();

    std::vector<float> ratios;
    std::vector<float> scales;
    int image_center = 0;
    int size = 0;
    int anchor_num;

    MatrixXf anchors;
    int stride;

};


#endif //CPPTEST_ANCHORS_H
