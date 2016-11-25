//
// Created by vbochkov on 02.06.16.
//

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "object_extracting.h"

object_extracting::object_extracting(const json11::Json& oe_node, cv_caps *capabs_ptr, time_t timestamp):
    cv_module(capabs_ptr, timestamp) {
    settings.net_cols = oe_node["net_cols"].int_value();
    settings.net_rows = oe_node["net_rows"].int_value();
    range_net.resize(settings.net_rows);
    for (auto& row : range_net)
        row.resize(settings.net_cols);
}

void object_extracting::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {

    auto filter_conts_by_square = [&](std::vector<std::vector<cv::Point>> contours) -> std::vector<std::vector<cv::Point>> {
        std::vector<std::vector<cv::Point> > target_contours;
        for (auto& cont: contours) {
            cv::Moments m = cv::moments(cont);
            if (m.m00 > 1000)
                target_contours.push_back(cont);
        }
        return target_contours;
    };
    cv::Mat hsv;
    cv::Mat channels[3];
    cv::Mat grads[3];
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, channels);
    for (int i = 0; i < 3; ++i) {
        cv::Mat gray = channels[i];
        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
        cv::convertScaleAbs(grad_x, grad_x);
        cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
        cv::convertScaleAbs(grad_y, grad_y);
        addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grads[i]);
        std::string grad_file = std::string("grad") + std::to_string(i) + std::string(".jpeg");
        cv::threshold(grads[i], grads[i], 0., 255, cv::THRESH_OTSU + cv::THRESH_BINARY);
        cv::imwrite(grad_file, grads[i]);
    }
    cv::Scalar sumH = cv::sum(grads[0]);
    cv::Scalar sumS = cv::sum(grads[1]);
    cv::Scalar sumV = cv::sum(grads[2]);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    int target_id;
    if (sumH.val[0] <= sumS.val[0] && sumH.val[0] <= sumV.val[0]) {
        target_id = 0;
        std::cout << "minimal is H" << std::endl;
    } else if (sumS.val[0] <= sumV.val[0] && sumS.val[0] <= sumH.val[0]) {
        target_id = 1;
        std::cout << "minimal is S" << std::endl;
    } else if (sumV.val[0] <= sumS.val[0] && sumV.val[0] <= sumH.val[0]) {
        target_id = 2;
        std::cout << "minimal is V" << std::endl;
    }
    cv::findContours(grads[target_id], contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point> > target_conts = filter_conts_by_square(contours);
    std::cout << "target_conts size = " << target_conts.size() << std::endl;
    cv::drawContours(frame, target_conts, -1, cv::Scalar(0xff, 0, 0), 2);
    cv::imwrite("frame.jpeg", frame);
    cv_module::compute(frame, metadata, overlay);
}

void object_extracting::draw_overlay(cv::Mat overlay) {
    for (int i = 0; i < settings.net_rows; ++i)
        for (int j = 0; j < settings.net_cols; ++j)
            cv::rectangle(
                    overlay,
                    cv::Rect(
                            (int) (j * (float)overlay.cols / settings.net_cols),
                            (int) (i * (float)overlay.rows / settings.net_rows),
                            (int) ((float)overlay.cols / settings.net_cols),
                            (int) ((float)overlay.rows / settings.net_rows)
                    ),
                    cv::Scalar(0, 0xff, 0), 2
            );
}