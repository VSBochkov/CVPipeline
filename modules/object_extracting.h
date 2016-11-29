//
// Created by vbochkov on 02.06.16.
//

#ifndef CV_PIPELINE_OBJECT_EXTRACTING_H
#define CV_PIPELINE_OBJECT_EXTRACTING_H


#include "../cvpipeline.h"
#include "../json11.hpp"

struct object_extr_settings {
    int net_rows;
    int net_cols;
};

class object_extracting: public cv_module {
public:
    object_extracting(const json11::Json& oe_node, cv_caps *capabs_ptr, cv_time timestamp);
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);
    void draw_overlay(cv::Mat overlay);

private:
    struct hsv_range {
        struct range { uchar start, end; };
        range h_range, s_range, v_range;
    };
    object_extr_settings settings;
    std::vector<std::vector<hsv_range>> range_net;
};


#endif //CV_PIPELINE_OBJECT_EXTRACTING_H
