//
// Created by vbochkov on 23.05.16.
//

#include <opencv2/imgproc.hpp>
#include "fgbg.h"

fgbg::fgbg(
        const json11::Json &fgbg_node, cv_caps *capabs_ptr, time_t timestamp
) : cv_module(capabs_ptr, timestamp) {
    settings.history             = fgbg_node["history"].int_value();
    settings.learning_rate       = fgbg_node["learning_rate"].number_value();
    settings.min_distance        = fgbg_node["min_distance"].int_value();
    settings.fTau                = fgbg_node["fTau"].number_value();
    settings.init_learning_time  = fgbg_node["init_learning_time"].number_value();
    gauss = cv::createBackgroundSubtractorMOG2(
            settings.history, settings.min_distance, true
    );
    gauss->setShadowThreshold(settings.fTau);
}

void fgbg::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    cv::Mat scale;
    int width   = frame.cols < IMPROC_WIDTH     ? frame.cols : IMPROC_WIDTH;
    int height  = frame.rows < IMPROC_HEIGHT    ? frame.rows : IMPROC_HEIGHT;
    cv::resize(frame, scale, cv::Size(width, height));
    gauss->apply(
            scale, metadata.fg_mask,
            metadata.timestamp - init_time < settings.init_learning_time ?
                1. : settings.learning_rate
    );
    cv_module::compute(frame, metadata, overlay);
}