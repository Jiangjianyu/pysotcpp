//
// Created by jjyjb on 22-11-1.
//

#include "Anchors.h"

Anchors::Anchors(int &stride, std::vector<float> &ratios, std::vector<float> &scales, int image_center, int size) :
        stride(stride), ratios(ratios), scales(scales) {
    this->image_center = image_center;
    this->size = size;
    anchor_num = scales.size() * ratios.size();
    this->generate_anchors();

}

void Anchors::generate_anchors() {
//    std::cout << "anchors1:" << anchors.rows() << "; " << anchors.cols() << std::endl;
    this->anchors = MatrixXf(anchor_num, 4);
//    std::cout << this->anchors << std::endl;

//    std::cout << "anchors2:" << anchors.rows() << "; " << anchors.cols() << std::endl;
    int size_temp = this->stride * this->stride;
    int count = 0;
    for (float r : ratios) {
        int ws = int(sqrt(size_temp * 1. / r));
        int hs = int(ws * r);
        for (float s : scales) {
            float w = ws * s;
            float h = hs * s;

            this->anchors(count, 0) = -w * 0.5;
            this->anchors(count, 1) = -h * 0.5;
            this->anchors(count, 2) = w * 0.5;
            this->anchors(count, 3) = h * 0.5;
            count += 1;

        }
    }
//    std::cout << this->anchors << std::endl;


}

bool Anchors::generate_all_anchors(int im_c, int size_tt) {
    if (this->image_center == im_c && this->size == size_tt)
        return false;
    this->image_center = im_c;
    this->size = size_tt;

//    a0x =


}
