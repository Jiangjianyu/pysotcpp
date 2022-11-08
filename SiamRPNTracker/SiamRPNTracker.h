//
// Created by jjyjb on 22-10-28.
//

#ifndef CPPTEST_SIAMRPNTRACKER_H
#define CPPTEST_SIAMRPNTRACKER_H

#include <iostream>
#include <vector>
//#include "utils.cpp"
#include "Hanning.h"
#include "Anchors.h"
#include <memory>
#include <torch/script.h>
#include <torch/torch.h>
#include "base_tracker.h"

using namespace cv;

class SiamRPNTracker : public SiameseTracker {
public:
    SiamRPNTracker(string& template_model_path, string& track_model_path);
    void init(const cv::Mat& img, const cv::Rect& bbox);
    std::pair<cv::Rect, float> track(const cv::Mat& img);


private:
    torch::jit::script::Module template_module_;
    torch::jit::script::Module track_module_;
    torch::DeviceType device_type;

    MatrixXf generate_anchor();

    static void _bbox_clip(float& x, float& y, float& w, float& h, cv::MatSize img_size);
//    static
//    void _convert_score();

    int score_size;
    int anchor_num;

    float PENALTY_K = 0.16;
    float WINDOW_INFLUENCE = 0.40;
    float LR = 0.30;
    int EXEMPLAR_SIZE = 127;
    int INSTANCE_SIZE = 287;
    int BASE_SIZE = 0;
    float CONTEXT_AMOUNT = 0.5;

    int STRIDE = 8;
    std::vector<float> RATIOS;
    std::vector<float> SCALES;

    cv::Mat window;

    std::shared_ptr<Anchors> anchor_;
//    MatrixXf anchors;
    cv::Mat anchors;

    cv::Point center_pos;
    cv::Point2f wh_size;
    cv::Scalar channel_average;

    torch::Tensor zf;
    cv::Mat score_m, pred_bbox_m;

};


#endif //CPPTEST_SIAMRPNTRACKER_H
