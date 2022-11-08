//
// Created by jjyjb on 22-10-31.
//

#ifndef CPPTEST_UTILS_H
#define CPPTEST_UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;



//hanning 窗计算
VectorXd calc_hanning(size_t m, size_t n)
{
    return 0.5 - 0.5*(2 * M_PI*VectorXd::LinSpaced(m, 1, m).array()/(n+1)).cos();
}
//hanning对称计算
VectorXd sym_hanning(size_t n)
{
    int half;
    VectorXd w1, w2;
    if (n % 2 == 0)//偶数阶
    {
        half = n / 2;
        w1 = calc_hanning(half, n);
        w2 = w1.reverse();
    }
    else
    {
        half = (n + 1) / 2;
        w1 = calc_hanning(half, n);
        w2 = w1.reverse().segment(1, n - half);
    }

    VectorXd w(w1.size() + w2.size());
    w << w1, w2;
    return w;

}



// the N-point symmetric Hanning window in a column vector

VectorXd hanning(size_t n, bool flag=true)
{
    VectorXd w,w1;
    if (flag)
        w = sym_hanning(n);
    else
    {

        w1 = sym_hanning(n - 1);
        w.resize(n);
        w(0) = 0;
        for (int i = 0; i < n - 1; i++)
        {
            w(i + 1) = w1(i);
        }
    }


    return w;
}


#endif //CPPTEST_UTILS_H
