//
// Created by jjyjb on 22-10-28.
//

#include "SiamRPNTracker.h"
#include <typeinfo>


SiamRPNTracker::SiamRPNTracker(string &template_model_path, string &track_model_path) {
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    template_module_ = torch::jit::load(template_model_path);
    template_module_.to(device_type);
    template_module_.eval();
    track_module_ = torch::jit::load(track_model_path);
    track_module_.to(device_type);
    track_module_.eval();


    STRIDE = 8;
    RATIOS = {0.33, 0.5, 1, 2, 3};
    SCALES = {8};

    PENALTY_K = 0.16;
    WINDOW_INFLUENCE = 0.40;
    LR = 0.30;
    EXEMPLAR_SIZE = 127;
    INSTANCE_SIZE = 287;
    BASE_SIZE = 0;
    CONTEXT_AMOUNT = 0.5;

    score_size = (INSTANCE_SIZE - EXEMPLAR_SIZE) / STRIDE + 1 + BASE_SIZE;
    anchor_num = RATIOS.size() * SCALES.size();

//    Hanning hanning;
    VectorXf hanning = Hanning::hanning(score_size, false);

    MatrixXf win = hanning * hanning.transpose();
    Map<RowVectorXf> v1(win.data(), win.size() * anchor_num);
    std::vector<float> temp_v(v1.data(), v1.data() + v1.cols() * v1.rows());


    for (int i = 1; i < anchor_num; ++i) {
        for (int j = 0; j < win.size(); ++j) {
            temp_v[i * win.size() + j] = temp_v[j];
        }
    }

    this->window = cv::Mat(1, temp_v.size(), CV_32F, cv::Scalar(0));
    std::memcpy((float *) this->window.data, temp_v.data(), sizeof(float) * temp_v.size());

//    window = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(temp_v.data(), temp_v.size());
//    std::cout << "window: \n" << window << std::endl;

    auto acs = this->generate_anchor();

    cv::eigen2cv(acs, this->anchors);

    int size1[3] = {10, 21, 21};
    int size2[3] = {20, 21, 21};
    this->score_m = cv::Mat(3, size1, CV_32F, cv::Scalar(0));
    this->pred_bbox_m = cv::Mat(3, size2, CV_32F, cv::Scalar(0));


//    std::cout << "this->anchors:\n" << this->anchors << std::endl;
//    std::cout << "anchorss:" << anchorss << std::endl;
//    std::cout << "anchorss:" << anchorss.type() << std::endl;
}

//hanning: [0.         0.02447174 0.0954915  0.20610737 0.3454915  0.5
//0.6545085  0.79389263 0.9045085  0.97552826 1.         0.97552826
//0.9045085  0.79389263 0.6545085  0.5        0.3454915  0.20610737
//0.0954915  0.02447174 0.        ]
void SiamRPNTracker::_bbox_clip(float &x, float &y, float &w, float &h, cv::MatSize img_size) {
    x = std::max(0.f, float(std::min(x, float(img_size[1]))));
    y = std::max(0.f, float(std::min(y, float(img_size[0]))));
    w = std::max(10.f, float(std::min(w, float(img_size[1]))));
    h = std::max(10.f, float(std::min(h, float(img_size[0]))));

}

MatrixXf SiamRPNTracker::generate_anchor() {
    anchor_ = std::make_shared<Anchors>(STRIDE, RATIOS, SCALES);
    MatrixXf anchor = anchor_->anchors;
//    std::cout << "anchor:" << anchor.rows() << "; " << anchor.cols() << std::endl;
//    std::cout << anchor << std::endl;
    MatrixXf x1 = anchor.block(0, 0, anchor.rows(), 1);
    MatrixXf y1 = anchor.block(0, 1, anchor.rows(), 1);
    MatrixXf x2 = anchor.block(0, 2, anchor.rows(), 1);
    MatrixXf y2 = anchor.block(0, 3, anchor.rows(), 1);
    MatrixXf temp0 = anchor;
    temp0.col(0) = (x1 + x2) * 0.5;
    temp0.col(1) = (y1 + y2) * 0.5;
    temp0.col(2) = x2 - x1;
    temp0.col(3) = y2 - y1;
//    std::cout << "temp0: \n" << temp0 << std::endl;
//    std::cout << "anchor_num: \n" << anchor_num << std::endl;
//    std::cout << "score_size: \n" << score_size << std::endl;
//    std::vector<float> temp_v1(score_size * score_size * anchor_num);
//    std::cout << "temp_v1: \n" << temp_v1.size() << std::endl;
    MatrixXf temp_v1(score_size * score_size * anchor_num, 4);
//    std::cout << "temp_v1:" << temp_v1.rows() << "; " << temp_v1.cols() << std::endl;
    int ss = score_size * score_size;

    for (int i = 0; i < anchor_num; ++i) {
        for (int j = 0; j < ss; ++j) {
            int id = i * ss + j;
            temp_v1(id, 0) = temp0(i, 0);
            temp_v1(id, 1) = temp0(i, 1);
            temp_v1(id, 2) = temp0(i, 2);
            temp_v1(id, 3) = temp0(i, 3);
//            std::cout << "id: " << id << std::endl;

        }
    }
    anchor = temp_v1;
//    std::cout << "anchor2:" << anchor.rows() << "; " << anchor.cols() << std::endl;
//    std::cout << "anchor:" << anchor << std::endl;
    int ori = -(score_size / 2) * STRIDE;
//    std::cout << "ori: \n" << ori << std::endl;
//    std::cout << "STRIDE: \n" << STRIDE << std::endl;
    MatrixXf xx(score_size, score_size);
    MatrixXf yy(score_size, score_size);

    for (int k = 0; k < score_size; ++k) {
        for (int i = 0; i < score_size; ++i) {
            xx(k, i) = i * STRIDE + ori;
        }
    }
    for (int k = 0; k < score_size; ++k) {
        for (int i = 0; i < score_size; ++i) {
            yy(k, i) = k * STRIDE + ori;
        }
    }
    Map<RowVectorXf> yy_v1(xx.data(), xx.size());
    Map<RowVectorXf> xx_v1(yy.data(), yy.size());


    for (int l = 0; l < anchor_num; ++l) {

        for (int i = 0; i < ss; ++i) {
            int id = l * ss + i;

            anchor(id, 0) = xx_v1(i);
        }
    }
    for (int l = 0; l < anchor_num; ++l) {

        for (int i = 0; i < ss; ++i) {
            int id = l * ss + i;

            anchor(id, 1) = yy_v1(i);
        }
    }
//    std::cout << "xx: \n" << xx << std::endl;
//    std::cout << "yy: \n" << yy << std::endl;
//    std::cout << "xx_v1: \n" << xx_v1.size() << std::endl;
//    std::cout << "yy_v1: \n" << yy_v1 << std::endl;
//    std::cout << "anchor:" << anchor << std::endl;
    return anchor;

}


void SiamRPNTracker::init(const cv::Mat &img, const cv::Rect &bbox) {
    this->center_pos = cv::Point(bbox.x + (bbox.width - 1) / 2, bbox.y + (bbox.height - 1) / 2);
//    std::cout << "this->center_pos: " << this->center_pos << std::endl;
//    std::cout << "bbox.width: " << bbox.width << std::endl;
//    std::cout << "bbox.height: " << bbox.height << std::endl;
    this->wh_size = cv::Point2f(bbox.width, bbox.height);
//    std::cout << "this->wh_size: " << this->wh_size << std::endl;

    float w_z = this->wh_size.x + this->CONTEXT_AMOUNT * (this->wh_size.x + this->wh_size.y);
    float h_z = this->wh_size.y + this->CONTEXT_AMOUNT * (this->wh_size.x + this->wh_size.y);
    int s_z = std::round(std::sqrt(w_z * h_z));

    this->channel_average = cv::mean(img);
//    std::cout << "this->channel_average:" << this->channel_average << std::endl;

    cv::Mat z_crop;
    this->get_subwindow(img, z_crop, this->center_pos, this->EXEMPLAR_SIZE, s_z, this->channel_average);
//    std::cout << "z_crop:" << z_crop.cols << " " << z_crop.rows << std::endl;
    torch::Tensor tensor_image = torch::from_blob(z_crop.data,
                                                  {z_crop.rows, z_crop.cols, 3},
                                                  torch::kByte).to(device_type);
    tensor_image = tensor_image.permute({2, 0, 1});
    tensor_image = tensor_image.toType(torch::kFloat);
//    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);
    this->zf = template_module_.forward({tensor_image}).toTensor();

//    std::cout << "template zf:" << this->zf.sizes() << std::endl;
//    std::cout << "tempVal:" << tempVal.val[1] << std::endl;
//    std::cout << "tempVal:" << tempVal.val[2] << std::endl;

}

std::pair<cv::Rect, float> SiamRPNTracker::track(const cv::Mat &img) {
    int wh_sum = this->wh_size.x + this->wh_size.y;
    int w_z = this->wh_size.x + this->CONTEXT_AMOUNT * wh_sum;
    int h_z = this->wh_size.y + this->CONTEXT_AMOUNT * wh_sum;
    float s_z = std::sqrt(w_z * h_z);
    float scale_z = this->EXEMPLAR_SIZE / s_z;
    int s_x = std::round(s_z * this->INSTANCE_SIZE / this->EXEMPLAR_SIZE);
    cv::Mat x_crop;

//    std::cout << "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" << std::endl;
//    std::cout << "wh_sum: " << wh_sum << std::endl;
//    std::cout << "w_z: " << w_z << std::endl;
//    std::cout << "h_z: " << h_z << std::endl;
//    std::cout << "s_z: " << s_z << std::endl;
//    std::cout << "s_x: " << s_x << std::endl;
//    std::cout << "s_z * (this->INSTANCE_SIZE / this->EXEMPLAR_SIZE): " << s_z * this->INSTANCE_SIZE / this->EXEMPLAR_SIZE << std::endl;

    this->get_subwindow(img, x_crop, this->center_pos, this->INSTANCE_SIZE, s_x, this->channel_average);

//    std::cout << "x_crop:" << x_crop.cols << " " << x_crop.rows << std::endl;
    torch::Tensor tensor_image = torch::from_blob(x_crop.data,
                                                  {x_crop.rows, x_crop.cols, 3},
                                                  torch::kByte).to(device_type);
    tensor_image = tensor_image.permute({2, 0, 1});
    tensor_image = tensor_image.toType(torch::kFloat);
//    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);
//    std::cout << "tensor_image:" << tensor_image.sizes() << std::endl;
//    std::cout << "template zf:" << this->zf.sizes() << std::endl;

    auto outputs = track_module_.forward({tensor_image, this->zf}).toTuple();
    torch::Tensor cls = outputs->elements()[0].toTensor();
    torch::Tensor loc = outputs->elements()[1].toTensor();
//    std::cout << "track cls:" << cls.sizes() << std::endl;
//    std::cout << "track loc:" << loc.sizes() << std::endl;
    cls = cls.to(torch::kCPU);
    loc = loc.to(torch::kCPU);
//    std::cout << "this score_m:" << this->score_m.size << std::endl;
//    std::cout << "cls.numel():" << cls.numel() << std::endl;
//    std::cout << "sizeof(torch::kFloat):" << sizeof(torch::kFloat) << std::endl;
//    std::cout << "sizeof(float):" << sizeof(float) << std::endl;
//    std::cout << "sizeof(cv):" << sizeof(CV_32F) << std::endl;
//    std::memcpy((float *) this->score_m.data, cls.data_ptr(), sizeof(float) * cls.numel());
//    std::memcpy((float *) this->pred_bbox_m.data, loc.data_ptr(), sizeof(float) * loc.numel());

    cv::Mat score_t(2, 2205, CV_32F, cv::Scalar(0));
    cv::Mat score_fm;
    std::memcpy((float *) score_t.data, cls.data_ptr(), sizeof(float) * cls.numel());
    cv::Mat score_te = score_t.t();
//    std::cout << "score_t: \n" << score_t << std::endl;

    this->soft_max(score_te, score_fm);
    cv::Mat score = score_fm.col(1).t();
//    std::cout << "score_fm: " << score_fm.t() << std::endl;


    cv::Mat pred_bbox(4, 2205, CV_32F, cv::Scalar(0));
    std::memcpy((float *) pred_bbox.data, loc.data_ptr(), sizeof(float) * loc.numel());
    cv::Mat anchors_t = this->anchors.t();
//    std::cout << "anchors: " << this->anchors.size << std::endl;
//    std::cout << "pred_bbox: " << pred_bbox.size << std::endl;
    pred_bbox.row(0) = pred_bbox.row(0).mul(anchors_t.row(2)) + anchors_t.row(0);
    pred_bbox.row(1) = pred_bbox.row(1).mul(anchors_t.row(3)) + anchors_t.row(1);
    cv::exp(pred_bbox.row(2), pred_bbox.row(2));
    pred_bbox.row(2) = pred_bbox.row(2).mul(anchors_t.row(2));
    cv::exp(pred_bbox.row(3), pred_bbox.row(3));
    pred_bbox.row(3) = pred_bbox.row(3).mul(anchors_t.row(3));

//    std::cout << "pred_bbox: " << pred_bbox << std::endl;
    auto pad1 = (pred_bbox.row(2) + pred_bbox.row(3)) * 0.5;
    cv::Mat t1;
    cv::sqrt((pred_bbox.row(2) + pad1).mul(pred_bbox.row(3) + pad1), t1);
//    std::cout << "t11: " << t1 << std::endl;
    float ww = this->wh_size.x * scale_z;
    float hh = this->wh_size.y * scale_z;
    auto pad2 = (ww + hh) * 0.5;
    auto t2 = std::sqrt((ww + pad2) * (hh + pad2));
//    std::cout << "t2: " << t2 << std::endl;

    t1 /= t2;

//    std::cout << "t12: " << t1 << std::endl;
    auto t3 = 1 / t1;
//    std::cout << "t3: " << t3 << std::endl;
    cv::Mat s_c;
//    cv::compare(t1, t3, s_c, cv::CMP_EQ);
    cv::max(t1, t3, s_c);
//    std::cout << "s_c: " << s_c << std::endl;
    cv::Mat t4 = (this->wh_size.x / this->wh_size.y) / (pred_bbox.row(2) / pred_bbox.row(3));
//    std::cout << "(pred_bbox.row(2) / pred_bbox.row(3: " << pred_bbox.row(2) / pred_bbox.row(3) << std::endl;
//    std::cout << "this->wh_size.x / this->wh_size.y: " << this->wh_size.x / this->wh_size.y << std::endl;
//    std::cout << "this->wh_size.x : " << this->wh_size.x << std::endl;
//    std::cout << "this->wh_size.y: " << this->wh_size.y << std::endl;
//    std::cout << "t4: " << t4 << std::endl;
    auto t5 = 1 / t4;
    cv::Mat r_c;
    cv::max(t4, t5, r_c);
//    std::cout << "t5: " << t5 << std::endl;
//    std::cout << "s_c: " << s_c << std::endl;
//    std::cout << "r_c: " << r_c << std::endl;
//    std::cout << "s_c r_c: " << -(s_c.mul(r_c) - 1) * this->PENALTY_K << std::endl;
    cv::Mat penalty;
    cv::exp(-(s_c.mul(r_c) - 1) * this->PENALTY_K, penalty);
    cv::Mat pscore = penalty.mul(score);

//    std::cout << "penalty: " << penalty << std::endl;
//    std::cout << "penalty: " << penalty.size << std::endl;
//    std::cout << "score: " << score << std::endl;
//    std::cout << "score: " << score.size << std::endl;
//    std::cout << "pscore: " << pscore << std::endl;
//    std::cout << "pscore: " << pscore.size << std::endl;
    pscore = pscore * (1 - this->WINDOW_INFLUENCE) + this->window * this->WINDOW_INFLUENCE;
//    std::cout << "pscore: " << pscore << std::endl;
//    max = *std::max_element(begin, end);
    int best_idx = std::distance(pscore.begin<float>(), std::max_element(pscore.begin<float>(), pscore.end<float>()));
//    std::cout << "best_idx: " << best_idx << std::endl;
    cv::Mat bbox = pred_bbox.col(best_idx) / scale_z;
//    std::cout << "bbox: " << bbox << std::endl;


    float best_score =score.at<float>(0, best_idx);


    float lr = penalty.at<float>(0, best_idx) * best_score * this->LR;
//    std::cout << "lr: " << lr << std::endl;
    float cx = bbox.at<float>(0, 0) + this->center_pos.x;
    float cy = bbox.at<float>(1, 0) + this->center_pos.y;
//    std::cout << "cx: " << cx << std::endl;
//    std::cout << "cy: " << cy << std::endl;

    float width = this->wh_size.x * (1 - lr) + bbox.at<float>(2, 0) * lr;
    float height = this->wh_size.y * (1 - lr) + bbox.at<float>(3, 0) * lr;

//    std::cout << "width: " << width << std::endl;
//    std::cout << "height: " << height << std::endl;
//    std::cout << "image: " << img.size[0] << std::endl;
//    std::cout << "image: " << img.size[1] << std::endl;
    _bbox_clip(cx, cy, width, height, img.size);
//    std::cout << "cx: " << cx << std::endl;
//    std::cout << "cy: " << cy << std::endl;
//    std::cout << "width: " << width << std::endl;
//    std::cout << "height: " << height << std::endl;
    this->center_pos = cv::Point(std::floor(cx), std::floor(cy));
    this->wh_size = cv::Point2f(width, height);
    cv::Rect bbox_f(std::floor(cx - width / 2), std::floor(cy - height / 2), std::floor(width), std::floor(height));
    return std::make_pair(bbox_f, best_score);
//    cv::Rect, float
}




