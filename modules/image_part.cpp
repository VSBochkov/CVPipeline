//
// Created by vbochkov on 01.06.16.
//

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "image_part.h"

image_part::image_part(
        const json11::Json& extraction_node, cv_caps *capabs_ptr, time_t timestamp, bool draw, bool ip
) : cv_module(capabs_ptr, timestamp, draw, ip) {
    double dx = capabs_ptr->frame_width / 100;
    double dy = capabs_ptr->frame_height / 100;
    settings.image_region = rect(
            point((int)(dx * extraction_node["x1"].int_value()), (int)(dy * extraction_node["y1"].int_value())),
            point((int)(dx * extraction_node["x2"].int_value()), (int)(dy * extraction_node["y2"].int_value()))
    );
}

void image_part::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    frame = frame
            .rowRange(settings.image_region.p1.y, settings.image_region.p2.y)
            .colRange(settings.image_region.p1.x, settings.image_region.p2.x);
    overlay = overlay
            .rowRange(settings.image_region.p1.y, settings.image_region.p2.y)
            .colRange(settings.image_region.p1.x, settings.image_region.p2.x);
    cv_module::compute(frame, metadata, overlay);
}

void image_part::draw_overlay(cv::Mat overlay) {
    cv::rectangle(overlay, cv::Rect(0, 0, overlay.cols, overlay.rows), cv::Scalar(0xff, 0, 0), 3);
}