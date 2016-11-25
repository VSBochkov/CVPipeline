//
// Created by vbochkov on 01.06.16.
//

#ifndef CV_PIPELINE_EXTRACTION_PART_H
#define CV_PIPELINE_EXTRACTION_PART_H


#include "../cvpipeline.h"
#include "../json11.hpp"

struct image_part_settings {
    rect image_region;
};

class image_part: public cv_module {
public:
    image_part(const json11::Json& extraction_node, cv_caps *capabs_ptr, time_t timestamp);
    virtual ~image_part() {}
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);
    void draw_overlay(cv::Mat overlay);

private:
    image_part_settings settings;
};


#endif //CV_PIPELINE_IMAGE_PART_H
