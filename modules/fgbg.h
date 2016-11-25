//
// Created by vbochkov on 23.05.16.
//

#ifndef PIPELINE_FGBG_H
#define PIPELINE_FGBG_H

#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include "../json11.hpp"
#include "../cvpipeline.h"

struct fgbg_settings {
    int history;
    int min_distance;
    double fTau;
    double learning_rate;
    int init_learning_time;
};

class fgbg: public cv_module {
public:
    fgbg(const json11::Json& fgbg_node, cv_caps *capabs_ptr, time_t timestamp);
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);

private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> gauss;
    fgbg_settings settings;
};


#endif //PIPELINE_FGBG_H
