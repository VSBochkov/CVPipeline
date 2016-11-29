#ifndef FLAMESRCBBOX_H
#define FLAMESRCBBOX_H

#include "firebbox.h"

class flame_src_bbox: public fire_bbox
{
public:
    flame_src_bbox(const json11::Json& fire_valid_node, cv_caps *capabs_ptr, cv_time timestamp, bool ip_del = false, bool over_draw = false);
    virtual ~flame_src_bbox() {}
    virtual void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);
};

#endif // FLAMESRCBBOX_H
