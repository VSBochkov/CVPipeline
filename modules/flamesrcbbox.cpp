#include "flamesrcbbox.h"
#include "fireweight.h"

#include <vector>

#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>


flame_src_bbox::flame_src_bbox(
        const json11::Json& flame_bbox_node, cv_caps *capabs_ptr, cv_time timestamp,
        bool draw_overlay, bool ip_deliver
) : cv_module(capabs_ptr, timestamp, draw_overlay, ip_deliver)
{
    settings.grav_thresh = flame_bbox_node["grav_thresh"].number_value();//10.;
    settings.min_area_percent = flame_bbox_node["min_area_percent"].number_value();//10;
    settings.intersect_thresh = flame_bbox_node["intersect_thresh"].number_value();//0.4;
    settings.dtime_thresh = flame_bbox_node["dtime_thresh"].number_value();//0.5;
}

void flame_src_bbox::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay)
{
    metadata.flame_bboxes = calc_bboxes(
            metadata.fire_weight.flame_mask.clone(), overlay, metadata.fire_weight.pixel_cnt,
            cv::Scalar(0xff, 0, 0), metadata
    );
}
