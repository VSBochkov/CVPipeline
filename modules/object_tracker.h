//
// Created by vbochkov on 24.05.16.
//

#ifndef PIPELINE_OBJECT_TRACKER_H
#define PIPELINE_OBJECT_TRACKER_H

#include "../json11.hpp"
#include "../cvpipeline.h"
#include <opencv2/core/core.hpp>
#include <list>

struct object_tracker_settings {
    double intersection_percent;
    int seconds_wait;
};

class object_tracker: public cv_module {
public:
    object_tracker(const json11::Json& objtr_node, cv_caps *capabs_ptr, time_t timestamp);
    void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay);

private:
    double get_intersection_percent(rect& obj_bbox, rect& bbox);
    void track_objects(cv_metadata& info);
    void draw_overlay(cv::Mat overlay);

private:
    std::list<cv_object> objects;
    object_tracker_settings settings;
    int global_id_counter;
};


#endif //PIPELINE_OBJECT_TRACKER_H
