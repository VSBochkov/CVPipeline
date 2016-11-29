#ifndef FIREMASKLOGICAND_H
#define FIREMASKLOGICAND_H

#include "firebbox.h"

struct fire_weight_distrib_settings
{
    float period;
    float weight_thr;
    float time_thr;
};

class fire_weight_distrib : public cv_module
{
public:
    fire_weight_distrib(const json11::Json& fire_valid_node, cv_caps *capabs_ptr, cv_time timestamp, bool ip_del = false, bool over_draw = false);
    virtual void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);


private:
    bool ptInRect(int i, int j, cv::Rect& bbox) {
        return (i >= bbox.y && j >= bbox.x &&
                i <= bbox.y + bbox.height &&
                j <= bbox.x + bbox.width);
    }

    cv_object* getBbox(int i, int j, std::vector<cv_object> bboxes) {
        static size_t id = 0;
        for (size_t cnt = 0; cnt < bboxes.size(); ++cnt)
            if (ptInRect(i, j, bboxes[(id + cnt) % bboxes.size()].bbox)) return &bboxes[(id + cnt) % bboxes.size()];
        return NULL;
    }

private:
    cv::Mat base;
    cv::Mat timings;
    int counter;
    fire_weight_distrib_settings settings;
};

#endif // FIREMASKLOGICAND_H
