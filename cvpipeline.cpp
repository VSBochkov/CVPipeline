//
// Created by vbochkov on 23.05.16.
//

#include "cvpipeline.h"
#include "json11.hpp"
#include "modules/fgbg.h"
#include "modules/fgsegm.h"
#include "modules/object_tracker.h"
#include "modules/intrusion_area.h"
#include "modules/image_part.h"
#include "modules/object_extracting.h"
#include "modules/rfiremaskingmodel.h"
#include "modules/firebbox.h"
#include "modules/firevalidation.h"

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


cvpipeline::cvpipeline(std::string filename) {
    display_frame = display_overlay = false;

    std::ifstream file(filename.c_str());
    std::stringstream buffer;

    buffer << file.rdbuf();
    std::string str = buffer.str();

    std::string err;
    auto json = json11::Json::parse(str, err);
    if (json["file_input"].is_string()) {
        std::string file_input = json["file_input"].string_value();
        video_input = cv::VideoCapture(file_input);
        get_caps();
    }
    if (json["device_input"].is_number()) {
        int device_input = json["device_input"].int_value();
        video_input = cv::VideoCapture(device_input);
        get_caps();
    }
    if (json["image_input"].is_string()) {
        input_image = cv::imread(json["image_input"].string_value());
        get_caps(input_image);
    }
    if (json["image_output"].is_string())
        image_output = json["image_output"].string_value();
    if (json["file_output"].is_string()) {
        std::string file_output = json["file_output"].string_value();
        video_output = cv::VideoWriter(
                file_output, 1482049860, caps.fps,
                cv::Size(caps.frame_width, caps.frame_height)
        );
    }
    cv_time timestamp = cv_time(true);
    /*if (json["image_part"].is_object())
        modules.push_back(new image_part(
            json["image_part"], &caps, timestamp,
            json["image_part"]["draw_overlay"].bool_value(),
            json["image_part"]["ip_deliver"].bool_value()
        ));
    if (json["object_extracting"].is_object())
        modules.push_back(new object_extracting(
            json["object_extracting"], &caps, timestamp,
            json["object_extracting"]["draw_overlay"].bool_value(),
            json["object_extracting"]["ip_deliver"].bool_value()
        ));
    if (json["fgbg"].is_object())
        modules.push_back(new fgbg(
            json["fgbg"], &caps, timestamp,
            json["fgbg"]["draw_overlay"].bool_value(),
            json["fgbg"]["ip_deliver"].bool_value()
        ));
    if (json["fgsegm"].is_object())
        modules.push_back(new fgsegm(
            json["fgsegm"], &caps, timestamp,
            json["fgsegm"]["draw_overlay"].bool_value(),
            json["fgsegm"]["ip_deliver"].bool_value()
        ));
    if (json["object_tracker"].is_object())
        modules.push_back(new object_tracker(
            json["object_tracker"], &caps, timestamp,
            json["object_tracker"]["draw_overlay"].bool_value(),
            json["object_tracker"]["ip_deliver"].bool_value()
        ));
    if (json["intrusion_area"].is_object())
        modules.push_back(new intrusion_area(
            json["intrusion_area"], &caps, timestamp,
            json["intrusion_area"]["draw_overlay"].bool_value(),
            json["intrusion_area"]["ip_deliver"].bool_value()
        ));
    */if (json["r_fire_mm"].is_object())
        modules.push_back(new r_fire_masking_model(
            json["r_fire_mm"], &caps, timestamp,
            json["r_fire_mm"]["draw_overlay"].bool_value(),
            json["r_fire_mm"]["ip_deliver"].bool_value()
        ));
    if (json["fire_valid"].is_object())
        modules.push_back(new fire_validation(
            json["fire_valid"], &caps, timestamp,
            json["fire_valid"]["draw_overlay"].bool_value(),
            json["fire_valid"]["ip_deliver"].bool_value()
        ));
    if (json["fire_bbox"].is_object())
        modules.push_back(new fire_bbox(
            json["fire_bbox"], &caps, timestamp,
            json["fire_bbox"]["draw_overlay"].bool_value(),
            json["fire_bbox"]["ip_deliver"].bool_value()
        ));
    if (json["fire_weight"].is_object())
        modules.push_back(new fire_weight(
            json["fire_weight"], &caps, timestamp,
            json["fire_weight"]["draw_overlay"].bool_value(),
            json["fire_weight"]["ip_deliver"].bool_value()
        ));
    if (json["flame_src_bbox"].is_object())
        modules.push_back(new flame_src_bbox(
            json["flame_src_bbox"], &caps, timestamp,
            json["flame_src_bbox"]["draw_overlay"].bool_value(),
            json["flame_src_bbox"]["ip_deliver"].bool_value()
        ));
}

cvpipeline::~cvpipeline() {
    for(cv_module* module: modules)
        delete module;
    modules.clear();

    if (video_input.isOpened())
        video_input.release();
}

void cvpipeline::process() {
    bool process = true;
    bool draw_overlay = std::find_if(
            modules.begin(), modules.end(),
            [](cv_module* mod) { return mod->draw_over; }
    ) != modules.end();
    bool ip_deliver = std::find_if(
            modules.begin(), modules.end(),
            [](cv_module* mod) { return mod->ip_deliver; }
    ) != modules.end();

    cv::Mat frame, overlay;
    while (process) {
        clock_t start = clock();
        if (video_input.isOpened()) {
            video_input >> frame;
            if (frame.empty()) break;
            get_caps();
        } else if (!input_image.empty()) {
            frame = input_image;
            process = false;
        } else break;
        if (draw_overlay)
            frame.copyTo(overlay);
        cv_metadata metadata;
        metadata.timestamp = time(NULL);
        cv::Mat _overlay = overlay;
        for (auto &module: modules)
            module->compute(frame, metadata, _overlay);
        for (auto &event: metadata.events)
            std::cout << event << std::endl;
        if (video_output.isOpened())
            video_output << overlay;
        else if (!image_output.empty())
            cv::imwrite(image_output, overlay);
        if (display_frame)
            cv::imshow("frame", frame);
        if (display_overlay)
            cv::imshow("overlay", overlay);
        if (display_debug)
            cv::imshow("fgbg", metadata.fg_mask);
        int delay_ms = (int) (1000. / caps.fps - (int)((float)(clock() - start) / CLOCKS_PER_SEC * 1000));
        cv::waitKey(0);//delay_ms > 0 ? delay_ms : 1);
    }
}