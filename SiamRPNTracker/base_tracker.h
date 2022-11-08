//
// Created by jjyjb on 2022/11/2.
//

#ifndef CPPTEST_BASE_TRACKER_H
#define CPPTEST_BASE_TRACKER_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/eigen.hpp>

class BaseTracker {
public:
    BaseTracker() {}

    virtual void init(const cv::Mat &img, const cv::Rect &bbox) {}

};

class SiameseTracker : public BaseTracker {
public:
    SiameseTracker() {}

    void
    get_subwindow(const cv::Mat &im, cv::Mat &im_patch, cv::Point &pos, int &model_sz, int &original_sz,
                  cv::Scalar &channel_average) {
//        std::cout << "######################################################" << std::endl;
//        std::cout << "pos: " << pos << std::endl;
//        std::cout << "model_sz: " << model_sz << std::endl;
//        std::cout << "original_sz: " << original_sz << std::endl;
//        std::cout << "channel_average: " << channel_average << std::endl;

        int sz = original_sz;
        int im_sz_w = im.cols;
        int im_sz_h = im.rows;
        int im_sz_c = im.channels();

        int c = (original_sz + 1) / 2;
        int context_xmin = std::floor(pos.x - c + 0.5);
        int context_xmax = context_xmin + sz - 1;
        int context_ymin = std::floor(pos.y - c + 0.5);
        int context_ymax = context_ymin + sz - 1;
        int left_pad = int(std::max(0, -context_xmin));
        int top_pad = int(std::max(0, -context_ymin));
        int right_pad = int(std::max(0, context_xmax - im_sz_w + 1));
        int bottom_pad = int(std::max(0, context_ymax - im_sz_h + 1));

//        std::cout << "context_xmin1: " << context_xmin << std::endl;
//        std::cout << "contest_xmax1: " << context_xmax << std::endl;
//        std::cout << "context_ymin1: " << context_ymin << std::endl;
//        std::cout << "context_ymax1: " << context_ymax << std::endl;
//        std::cout << "top_pad: " << top_pad << std::endl;
//        std::cout << "bottom_pad: " << bottom_pad << std::endl;
//        std::cout << "left_pad: " << left_pad << std::endl;
//        std::cout << "right_pad: " << right_pad << std::endl;

        context_xmin = context_xmin + left_pad;
        context_xmax = context_xmax + left_pad;
        context_ymin = context_ymin + top_pad;
        context_ymax = context_ymax + top_pad;
//        std::cout << "context_xmin2: " << context_xmin << std::endl;
//        std::cout << "contest_xmax2: " << context_xmax << std::endl;
//        std::cout << "context_ymin2: " << context_ymin << std::endl;
//        std::cout << "context_ymax2: " << context_ymax << std::endl;

//        std::cout << "top_pad:" << top_pad << std::endl;
//        std::cout << "bottom_pad:" << bottom_pad << std::endl;
//        std::cout << "left_pad:" << left_pad << std::endl;
//        std::cout << "right_pad:" << right_pad << std::endl;
//        std::cout << "model_sz:" << model_sz << std::endl;

        if (top_pad != 0 || bottom_pad != 0 || left_pad != 0 || right_pad != 0) {
//            std::cout << "oooooooooooooooooooooooooooooooooo" << std::endl;

//            std::cout << "im_sz_w:" << im_sz_w << std::endl;
//            std::cout << "im_sz_h:" << im_sz_h << std::endl;
            int sz_w_new = im_sz_w + left_pad + right_pad;
            int sz_h_new = im_sz_h + top_pad + bottom_pad;
            cv::Mat te_im(sz_h_new, sz_w_new, im.type(), cv::Scalar(0));
//            std::cout << "te_im0: " << te_im.rows << " " << te_im.cols << std::endl;

            cv::Rect rect_te(left_pad, top_pad, im_sz_w, im_sz_h);
            im.copyTo(te_im(rect_te));
//            std::cout << "pp: " << channel_average << std::endl;
//            std::cout << "sz_w_new: " << sz_w_new << std::endl;
//            std::cout << "sz_h_new: " << sz_h_new << std::endl;
//            std::cout << std::endl;
            if (top_pad != 0) {
                te_im(cv::Rect(left_pad, 0, im_sz_w, top_pad)) = channel_average;
//                std::cout << "top_pad: " << top_pad << std::endl;
//                std::cout << "cv::Rect(left_pad, 0, im_sz_w, top_pad): " << cv::Rect(left_pad, 0, im_sz_w, top_pad)
//                          << std::endl;
            }


            if (bottom_pad != 0) {
                te_im(cv::Rect(left_pad, im_sz_h + top_pad, im_sz_w, bottom_pad)) = channel_average;
//                std::cout << "bottom_pad: " << bottom_pad << std::endl;
//                std::cout << "cv::Rect(left_pad, im_sz_h + top_pad, im_sz_w, bottom_pad): "
//                          << cv::Rect(left_pad, im_sz_h + top_pad, im_sz_w, bottom_pad) << std::endl;
            }

            if (left_pad != 0) {
                te_im(cv::Rect(0, 0, left_pad, sz_h_new)) = channel_average;
//                std::cout << "left_pad: " << left_pad << std::endl;
//                std::cout << "cv::Rect(0, 0, left_pad, sz_h_new): " << cv::Rect(0, 0, left_pad, sz_h_new) << std::endl;
            }

            if (right_pad != 0) {
                te_im(cv::Rect(im_sz_w + left_pad, 0, right_pad, sz_h_new)) = channel_average;
//                std::cout << "right_pad: " << right_pad << std::endl;
//                std::cout << "cv::Rect(im_sz_w + left_pad, 0, right_pad, sz_h_new): "
//                          << cv::Rect(im_sz_w + left_pad, 0, right_pad, sz_h_new) << std::endl;
            }

//            std::cout << std::endl;

//            std::cout << "cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin,\n"
//                         "                           context_ymax - context_ymin): "
//                      << cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin,
//                                  context_ymax - context_ymin) << std::endl;
//            std::cout << "context_xmin: " << context_xmin << std::endl;
//            std::cout << "context_xmax: " << context_xmax << std::endl;
//            std::cout << "context_ymin: " << context_ymin << std::endl;
//            std::cout << "context_ymax: " << context_ymax << std::endl;
//            std::cout << "im_patch: " << im_patch.rows << " " << im_patch.cols << std::endl;
//            std::cout << "te_im: " << te_im.rows << " " << te_im.cols << std::endl;

            te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin,
                           context_ymax - context_ymin)).copyTo(
                    im_patch);


        } else {
            im(cv::Rect(context_xmin, context_ymin, context_xmax + 1 - context_xmin,
                        context_ymax + 1 - context_ymin)).copyTo(
                    im_patch);
        }
//        std::cout << "im1:" << im_patch.cols << " " << im_patch.rows << std::endl;

        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
//        std::cout << "im2:" << im_patch.cols << " " << im_patch.rows << std::endl;
    }

    void soft_max(cv::Mat &src, cv::Mat &dst) {
        src.copyTo(dst);
        float max = 0.0;
        float sum = 0.0;
        int cols = src.cols;
        for (int i = 0; i < src.rows; ++i) {
//            const float *rowDataptr = src.ptr<float>(i);
            auto begin = src.begin<float>() + cols * i;
            auto end = src.begin<float>() + cols * i + cols;
            max = *std::max_element(begin, end);
            cv::exp((src.row(i) - max), dst.row(i));
            sum = cv::sum(dst.row(i))[0];
            dst.row(i) /= sum;
        }
    }

//    virtual void init(const cv::Mat &img, const cv::Rect &bbox) {}

};


#endif //CPPTEST_BASE_TRACKER_H
