//
// Created by vbochkov on 24.05.16.
//

#include <opencv2/imgproc.hpp>
#include <iostream>
#include "object_tracker.h"

object_tracker::object_tracker(
        const json11::Json &objtr_node, cv_caps *capabs_ptr, time_t timestamp, bool draw, bool ip
) : cv_module(capabs_ptr, timestamp, draw, ip) {
    global_id_counter = 0;
    settings.intersection_percent = objtr_node["intersection_percent"].number_value();
    settings.seconds_wait         = objtr_node["seconds_wait"].int_value();
}

double object_tracker::get_intersection_percent(rect& obj_bbox, rect& bbox) {
    rect intersect;
    double sq = 0.;
    if (obj_bbox.intersection(bbox, intersect)) {
        int bbox_sq = bbox.square();
        int obj_sq = obj_bbox.square();
        sq = bbox_sq < obj_sq ? (double) bbox_sq / obj_sq : (double) obj_sq / bbox_sq;
    }
    return sq > settings.intersection_percent / 100. ? sq : 0.;
}

void object_tracker::track_objects(cv_metadata& info) {
    std::list<cv_object> objs = objects;
    for (auto& bbox: info.bboxes) {
        cv_object* object = NULL;
        double int_percent = 0.;
        for (auto& obj: objs) {
            double square_rel = get_intersection_percent(obj.bbox, bbox);
            if (square_rel > int_percent) {
                object = &obj;
                int_percent = square_rel;
            }
        }
        if (object) {
            object->bbox = bbox;
            object->lifetime += info.timestamp - object->timestamp;
            object->timestamp = info.timestamp;
            continue;
        }

        std::string event_str = "[" + std::string(ctime(&info.timestamp));
        event_str += "]: ObjectTracker: new object global id " + std::to_string(global_id_counter);
        info.events.push_back(event_str);
        objs.push_back(cv_object(global_id_counter, info.timestamp, bbox, "fgbg_object", "Object"));
        global_id_counter++;
    }
    objects.clear();
    for (auto& obj: objs)
        if (info.timestamp - obj.timestamp < settings.seconds_wait)
            objects.push_back(obj);
}

void object_tracker::draw_overlay(cv::Mat overlay) {
    for (auto& obj: objects) {
        cv::rectangle(overlay, (cv::Rect) obj.bbox, cv::Scalar(0xff, 0, 0), 2);
        rect text_rect = {obj.bbox.p1, point(obj.bbox.p1.x + 150, obj.bbox.p1.y + 20)};
        cv::rectangle(overlay, (cv::Rect) text_rect, cv::Scalar(0xff, 0, 0), -1);
        cv::Point origin = (cv::Point) obj.bbox.p1;
        origin.y += 12; origin.x += 5;
        char time_buff[15];
        strftime(time_buff, 15, " %X", localtime(&obj.lifetime));
        cv::putText(
                overlay, obj.value + std::to_string(obj.id) + time_buff,
                origin, cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0xff, 0xff, 0xff), 1
        );
    }
}

void object_tracker::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    if (!metadata.bboxes.empty()) {
        metadata.objects = objects;
        track_objects(metadata);
    }
    metadata.objects = objects;
    cv_module::compute(frame, metadata, overlay);
}