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
    if (json["display_frame"].is_bool())
        display_frame = json["display_frame"].bool_value();
    if (json["display_overlay"].is_bool())
        display_overlay = json["display_overlay"].bool_value();
    time_t timestamp = time(NULL);
    if (json["image_part"].is_object())
        modules.push_back(new image_part(json["image_part"], &caps, timestamp));
    if (json["object_extracting"].is_object())
        modules.push_back(new object_extracting(json["object_extracting"], &caps, timestamp));
    if (json["fgbg"].is_object())
        modules.push_back(new fgbg(json["fgbg"], &caps, timestamp));
    if (json["fgsegm"].is_object())
        modules.push_back(new fgsegm(json["fgsegm"], &caps, timestamp));
    if (json["object_tracker"].is_object())
        modules.push_back(new object_tracker(json["object_tracker"], &caps, timestamp));
    if (json["intrusion_area"].is_object())
        modules.push_back(new intrusion_area(json["intrusion_area"], &caps, timestamp));
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