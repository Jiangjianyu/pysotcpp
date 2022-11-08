//
// Created by jjy on 2022/6/24.
//
#include <iostream>


//#include "dataType.h"
#include "SiamRPNTracker.h"


using namespace std;
using namespace cv;

cv::Rect roi_point;
cv::Point g_topLeft(0, 0);
cv::Point g_botRight(0, 0);
cv::Point g_botRight_tmp(0, 0);
bool plot = false;
bool g_trackerInitialized = false;

void S_on_Mouse(int event, int x, int y, int flags, void *param)//画矩形框并截图
{
    cv::Mat img = ((cv::Mat *) param)->clone();
    if (event == cv::EVENT_LBUTTONDOWN && !g_trackerInitialized) {
        std::cout << "DOWN " << std::endl;
        g_topLeft = Point(x, y);
        plot = true;

    } else if (event == cv::EVENT_LBUTTONUP && !g_trackerInitialized) {
        std::cout << "UP " << std::endl;
        g_botRight = Point(x, y);
        plot = false;
        roi_point.x = g_topLeft.x;
        roi_point.y = g_topLeft.y;
        roi_point.height = g_botRight.y - g_topLeft.y;
        roi_point.width = g_botRight.x - g_topLeft.x;

        g_trackerInitialized = true;

    } else if (event == cv::EVENT_MOUSEMOVE && !g_trackerInitialized) {
        g_botRight_tmp = Point(x, y);
        if (plot) {
            rectangle(img, g_topLeft, g_botRight_tmp, Scalar(0, 255, 0), 2, 8);
            imshow("video", img);
        }

    }
}

int main() {


//    torch::jit::script::Module cnn_module_;
    std::string template_model_path = "/home/jjyjb/03_test/cpptest/model/template.pt";
    std::string track_model_path = "/home/jjyjb/03_test/cpptest/model/track.pt";

    SiamRPNTracker siamrpn(template_model_path, track_model_path);
//    cv::Mat aa(3, 3, CV_8UC3, cv::Scalar(0));
//    cv::Scalar ss(2, 3, 4);
//    std::cout << "aa:" << aa << std::endl;
//    std::cout << "ss:" << ss << std::endl;
//    aa(cv::Rect(1, 1, 2, 2)) = ss;
//    std::cout << "aa1:" << aa << std::endl;

//    siamrpn.init()

    cv::namedWindow("video", 1);

//    size_t n = 21;



    for (int i = 0; i < 461; ++i) {
//        std::string image_dir =
//                "/home/jjyjb/04_data/01_460/data/" + std::string(10 - std::to_string(i).length(), '0') +
//                std::to_string(i) + ".png";
        std::string image_dir =
                "/home/jjyjb/04_data/1862fourfishcamera/3/" +
                std::to_string(i) + ".png";

        cv::Mat frame = cv::imread(image_dir);

        if (i == 0) {
            imshow("video", frame);

            while (1) {

                char key = waitKey(10);

                setMouseCallback("video", S_on_Mouse, &frame);

                if (key == 'q')
                    break;

            }
//            std::cout << "roi_poin1t: " << roi_point << std::endl;
//            roi_point = cv::Rect(275, 201, 732, 390);

            siamrpn.init(frame, roi_point);
//            std::cout << "roi_point: " << roi_point << std::endl;

        } else {
            auto outputs = siamrpn.track(frame);
            cv::Rect rect_out = outputs.first;
            cv::rectangle(frame, rect_out, cv::Scalar(255, 255, 0), 2);
        }

        cv::imshow("video", frame);
        cv::waitKey(1);
    }

//    std::string imgdir1 = "/home/jjyjb/03_test/cpptest/image/imag/1.jpg";
//    std::string imgdir2 = "/home/jjyjb/03_test/cpptest/image/imag/6.jpg";
//    std::string imgdir3 = "/home/jjyjb/03_test/cpptest/image/imag/7.jpg";
//    int feature_dim = 512;
//
//    cv::Mat img1 = cv::imread(imgdir1);
//    cv::Mat img2 = cv::imread(imgdir2);
//    cv::Mat img3 = cv::imread(imgdir3);
//
//    if (torch::cuda::is_available()) {
//        device_type = torch::kCUDA;
//    } else {
//        device_type = torch::kCPU;
//    }
//
//    cnn_module_ = torch::jit::load(model_path);
//    cnn_module_.to(device_type);
//    cnn_module_.eval();
//
//    static const auto MEAN = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, -1, 1, 1}).cuda();
//    static const auto STD = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, -1, 1, 1}).cuda();
//
//    std::vector<torch::Tensor> resized;
//
//    cv::resize(img1, img1, cv::Size(64, 128));
//    cv::resize(img2, img2, cv::Size(64, 128));
//    cv::resize(img3, img3, cv::Size(64, 128));
//
//    torch::Tensor tensor_image = torch::from_blob(img1.data,
//                                                  {img1.rows, img1.cols, 3},
//                                                  torch::kByte).to(device_type).to(device_type);
//    tensor_image = tensor_image.permute({2, 0, 1}).toType(torch::kFloat).div(255);
//    resized.push_back(tensor_image);
//
//
//    tensor_image = torch::from_blob(img2.data,
//                                    {img2.rows, img2.cols, 3},
//                                    torch::kByte).to(device_type).to(device_type);
//    tensor_image = tensor_image.permute({2, 0, 1}).toType(torch::kFloat).div(255);
//    resized.push_back(tensor_image);
//
//
//    tensor_image = torch::from_blob(img3.data,
//                                    {img3.rows, img3.cols, 3},
//                                    torch::kByte).to(device_type).to(device_type);
//    tensor_image = tensor_image.permute({2, 0, 1}).toType(torch::kFloat).div(255);
//    resized.push_back(tensor_image);
//
//
//    torch::Tensor tensor_all = torch::stack(resized).sub_(MEAN).div_(STD).to(device_type);
////    torch::Tensor tensor_all = torch::stack(resized).to(device_type);
//
//    torch::NoGradGuard no_grad;
//
//    torch::Tensor features = cnn_module_.forward({tensor_all}).toTensor().squeeze();
//
//    torch::Tensor tensor_buffer = features.contiguous().view({1, -1});
////    std::cout << tensor_buffer.sizes() << std::endl;
////    feature_dim=512;
//    FEATURESS x(1, 512);
//    FEATURESS y(1, 512);
//    FEATURESS a(1, 512);
//
//    postProcess(tensor_buffer, x, y, a);
//
//    NearNeighborDisMetric metric(NearNeighborDisMetric::METRIC_TYPE::cosine, 0.3, 100);
//
//    Eigen::VectorXf aaa = metric._nncosine_distance(x, y);
//    std::cout << "aaa:" << aaa << std::endl;
//
//    Eigen::VectorXf bbb = metric._nncosine_distance(x, a);
//    std::cout << "bbb:" << bbb << std::endl;
//
//    Eigen::VectorXf ccc = metric._nncosine_distance(y, a);
//    std::cout << "ccc:" << ccc << std::endl;

    return 0;
}
