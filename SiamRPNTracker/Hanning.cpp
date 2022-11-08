//
// Created by jjyjb on 22-10-31.
//

#include "Hanning.h"

VectorXf Hanning::hanning(size_t n, bool flag) {
    VectorXf w, w1;
    if (flag)
        w = sym_hanning(n);
    else {

        w1 = sym_hanning(n - 1);
        w.resize(n);
        w(0) = 0;
        for (int i = 0; i < n - 1; i++) {
            w(i + 1) = w1(i);
        }
    }


    return w;
}

VectorXf Hanning::calc_hanning(size_t m, size_t n) {

    return 0.5 - 0.5 * (2 * M_PI * VectorXf::LinSpaced(m, 1, m).array() / (n + 1)).cos();
}

VectorXf Hanning::sym_hanning(size_t n) {

    int half;
    VectorXf w1, w2;
    if (n % 2 == 0)//偶数阶
    {
        half = n / 2;
        w1 = calc_hanning(half, n);
        w2 = w1.reverse();
    } else {
        half = (n + 1) / 2;
        w1 = calc_hanning(half, n);
        w2 = w1.reverse().segment(1, n - half);
    }

    VectorXf w(w1.size() + w2.size());
    w << w1, w2;
    return w;
}