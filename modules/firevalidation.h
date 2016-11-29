#ifndef FIREVALIDATION_H
#define FIREVALIDATION_H

#include "../cvpipeline.h"
#include "../json11.hpp"


struct fire_validation_settings
{
    float alpha1;
    float alpha2;
    float dma_thresh;
};

class fire_validation : public cv_module {
public:
    fire_validation(const json11::Json& fire_valid_node, cv_caps *capabs_ptr, cv_time timestamp, bool ip_del = false, bool over_draw = false);
    virtual void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);

private:
    template<class Type1, class Type2> double dist(cv::Point3_<Type1> p1, cv::Point3_<Type2> p2);

private:
    cv::Mat ema;
    cv::Mat dma;
    fire_validation_settings settings;
    bool first_time;
};

#endif // FIREVALIDATION_H
