//
// Created by vbochkov on 27.05.16.
//

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "intrusion_area.h"

intrusion_area::intrusion_area(
        const json11::Json& ia_node, cv_caps *capabs_ptr, time_t timestamp
) : cv_module(capabs_ptr, timestamp) {
    double dx = capabs_ptr->frame_width / 100;
    double dy = capabs_ptr->frame_height / 100;
    std::vector<json11::Json> areas = ia_node["areas"].array_items();
    for (auto &area: areas)
        settings.areas.push_back(
                rect(
                        point((int)(dx * area["x1"].int_value()), (int)(dy * area["y1"].int_value())),
                        point((int)(dx * area["x2"].int_value()), (int)(dy * area["y2"].int_value()))
                )
        );
}

void intrusion_area::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    blink_areas.clear();
    for (auto& object: metadata.objects) {
        for (int i = 0; i < settings.areas.size(); ++i) {
            rect intersection;
            bool inters_flag = settings.areas[i].intersection(object.bbox, intersection);
            bool found_obj_in_area = std::find(
                    object_in_areas[object.id].begin(), object_in_areas[object.id].end(), &settings.areas[i]
            ) != object_in_areas[object.id].end();

            if (inters_flag) {
                blink_areas.push_back(intersection);
                if (found_obj_in_area) continue;

                std::string event = "[" + std::string(ctime(&metadata.timestamp));
                event += "] Intrusion area: " + object.value + std::to_string(object.id);
                event += " entered in #" + std::to_string(i) + " area";
                metadata.events.push_back(event);
                object_in_areas[object.id].push_back(&settings.areas[i]);
            } else if (found_obj_in_area) {
                std::string event = "[" + std::string(ctime(&metadata.timestamp));
                event += "] Intrusion area: " + object.value + std::to_string(object.id);
                event += " left from #" + std::to_string(i) + " area";
                metadata.events.push_back(event);
                object_in_areas[object.id].remove(&settings.areas[i]);
            }
        }
    }
    cv_module::compute(frame, metadata, overlay);
}

void intrusion_area::draw_overlay(cv::Mat overlay) {
    for (auto& area: settings.areas)
        cv::rectangle(overlay, (cv::Rect) area, cv::Scalar(0, 0, 0xff), 3);

    for (auto& area: blink_areas)
        cv::rectangle(overlay, (cv::Rect) area, cv::Scalar(0, 0xff, 0xff), -1);
}