//
// Created by vbochkov on 23.05.16.
//

#ifndef PIPELINE_FGSEGM_H
#define PIPELINE_FGSEGM_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include "../cvpipeline.h"
#include "../json11.hpp"

struct cv_object;
struct cv_metadata;
struct pipeline_settings;
struct rect;

struct fgsegm_settings
{
    int dilation_size;
    int opening_size;
    double thresh_gravity;
    double dist_mx;
    double dist_my;
    int min_contour_perim;
};

class fgsegm: public cv_module {
public:
    fgsegm(const json11::Json& fgbg_node, cv_caps *capabs_ptr, time_t timestamp);
    virtual ~fgsegm() {}
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);

private:
    void mask_prepare(cv::Mat mask_scaled);
    void gravitation(cv::Mat mask_scaled);
    std::list<rect> get_bboxes(cv::Mat mask_scaled);
    fgsegm_settings settings;
};


#endif //PIPELINE_FGSEGM_H
