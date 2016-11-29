#ifndef RFIREMASKINGMODEL_H
#define RFIREMASKINGMODEL_H

#include "../cvpipeline.h"
#include "../json11.hpp"


class r_fire_masking_model : public cv_module {
public:
    r_fire_masking_model(const json11::Json& ia_node, cv_caps *capabs_ptr, cv_time timestamp, bool draw = false, bool ip = false);
    virtual ~r_fire_masking_model() {}
    virtual void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);
};

#endif // RFIREMASKINGMODEL_H
