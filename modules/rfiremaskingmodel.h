#ifndef RFIREMASKINGMODEL_H
#define RFIREMASKINGMODEL_H

#include "../cvpipeline.h"
#include "../json11.hpp"

struct r_fire_mm_settings {
    cv::Mat mask;
    ulong pixel_cnt;
};

class r_fire_masking_model : public cv_module {
public:
    r_fire_masking_model(const json11::Json& ia_node, cv_caps *capabs_ptr, time_t timestamp, bool draw = false, bool ip = false);
    virtual ~r_fire_masking_model() {}
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);
    void draw_overlay(cv::Mat overlay);

private:
    r_fire_mm_settings settings;
};

#endif // RFIREMASKINGMODEL_H
