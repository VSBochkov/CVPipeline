#ifndef FIRE_BBOX_H
#define FIRE_BBOX_H

#include <QSharedPointer>
#include <QDataStream>
#include "cvgraphnode.h"
#include "../cvpipeline.h"
#include "../json11.hpp"
#include <vector>

struct fire_bbox_settings
{
    double grav_thresh;
    double min_area_percent;
    double intersect_thresh;
    double dtime_thresh;
};

class fire_bbox : public cv_module {
public:
    fire_bbox(const json11::Json& fgbg_node, cv_caps *capabs_ptr, time_t timestamp, bool ip_del = false, bool over_draw = false);
    fire_bbox(const json11::Json& fgbg_node, cv_caps *capabs_ptr, time_t timestamp);
    virtual ~fgsegm() {}
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);

protected:
    std::vector<cv_object> calc_bboxes(cv::Mat proc_mask, cv::Mat overlay, ulong pixel_cnt, cv::Scalar bbox_color, CVKernel::CVProcessData &process_data);

    cv::Rect intersection(cv::Rect rect1, cv::Rect rect2) {
        int x1 = std::max(rect1.x, rect2.x);
        int y1 = std::max(rect1.y, rect2.y);
        int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = std::min(rect1.y + rect2.height, rect2.y + rect2.height);
        if (x1 >= x2 || y1 >= y2)
            return cv::Rect();
        else
            return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }
    cv::Rect rect_union(cv::Rect rect1, cv::Rect rect2) {
        int x1 = std::min(rect1.x, rect2.x);
        int y1 = std::min(rect1.y, rect2.y);
        int x2 = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = std::max(rect1.y + rect2.height, rect2.y + rect2.height);
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }


protected:
    std::vector<cv_object> base_bboxes;
    fire_bbox_settings settings;
};

#endif // FIRE_BBOX_H
