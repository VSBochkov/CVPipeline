//
// Created by vbochkov on 27.05.16.
//

#ifndef CV_PIPELINE_INTRUSION_AREA_H
#define CV_PIPELINE_INTRUSION_AREA_H

#include <set>
#include "../cvpipeline.h"
#include "../json11.hpp"

struct ia_settings
{
    std::vector<rect> areas;
};

class intrusion_area: public cv_module {
public:
    intrusion_area(const json11::Json& ia_node, cv_caps *capabs_ptr, cv_time timestamp);
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);
    void draw_overlay(cv::Mat overlay);

private:
    ia_settings settings;
    std::list<rect> blink_areas;
    std::map<int, std::list<rect*>> object_in_areas;
};


#endif //CV_PIPELINE_INTRUSION_AREA_H
