//
// Created by vbochkov on 23.05.16.
//

#ifndef PIPELINE_CVPIPELINE_H
#define PIPELINE_CVPIPELINE_H

#include <list>
#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <time.h>
#include <iostream>


#define CIF_WIDTH   352
#define CIF_HEIGHT  288

#define IMPROC_WIDTH   CIF_WIDTH
#define IMPROC_HEIGHT  CIF_HEIGHT

struct point
{
    int x, y;
    point(int a1 = 0, int a2 = 0): x(a1), y(a2) {}
    operator cv::Point() {
        return cv::Point(x, y);
    }
};

struct rect {
    point p1;
    point p2;
    rect(point a1 = point(), point a2 = point()) : p1(a1), p2(a2) {}
    rect(cv::Rect r, double hx = 1.0, double hy = 1.0) {
        p1.x = (int)(r.x * hx);
        p1.y = (int)(r.y * hy);
        p2.x = (int)((r.x + r.width) * hx);
        p2.y = (int)((r.y + r.height) * hy);
    }
    operator cv::Rect() {
        cv::Rect r;
        r.x = p1.x;
        r.y = p1.y;
        r.height = p2.y - p1.y;
        r.width = p2.x - p1.x;
        return r;
    }
    int square() {
        return abs(p2.x - p1.x) * abs(p2.y - p1.y);
    }
    int perimeter() {
        return 2 * (abs(p2.x - p1.x) + abs(p2.y - p1.y));
    }
    bool intersection(rect arg, rect& intersect) {
        intersect.p1 = point(p1.x > arg.p1.x ? p1.x : arg.p1.x, p1.y > arg.p1.y ? p1.y : arg.p1.y);
        intersect.p2 = point(p2.x < arg.p2.x ? p2.x : arg.p2.x, p2.y < arg.p2.y ? p2.y : arg.p2.y);
        return intersect.p1.x <= intersect.p2.x && intersect.p1.y <= intersect.p2.y;
    }
};


struct cv_object {
    int id;
    time_t timestamp;
    time_t lifetime;
    rect bbox;
    std::string type;
    std::string value;
    cv_object *linked_obj;
    cv_object(
            int o_id, time_t o_timestamp, rect o_bbox = rect(), std::string o_type = "",
            std::string o_value = "", cv_object *o_linked_obj = NULL
    ) {
        id = o_id;
        bbox = o_bbox;
        timestamp = o_timestamp;
        struct tm null_time;
        null_time.tm_year = 0; null_time.tm_mon = 0; null_time.tm_yday = 0;
        null_time.tm_hour = 0; null_time.tm_min = 0; null_time.tm_sec = 0;
        lifetime = mktime(&null_time);
        type = o_type;
        value = o_value;
        linked_obj = o_linked_obj;
    }
};

struct cv_metadata {
    time_t timestamp;
    double fps;
    cv::Mat fg_mask;
    std::list<rect> bboxes;
    std::list<cv_object> objects;
    std::list<std::string> events;
};

struct cv_caps {
    double frame_width;
    double frame_height;
    double fps;
};

class cv_module {
public:
    cv_module(cv_caps* capabs_ptr, time_t current_time) {
        caps = capabs_ptr;
        init_time = current_time;
    }
    virtual ~cv_module() {};
    virtual void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
        this->draw_overlay(overlay);
    };
    virtual void draw_overlay(cv::Mat overlay) {}

private:
    cv_module(const cv_module&);
    cv_module& operator=(const cv_module&);

protected:
    cv_caps* caps;
    time_t init_time;
};

class cvpipeline {
public:
    cvpipeline(std::string filename);
    ~cvpipeline();

    void process();

private:
    void get_caps() {
        caps.fps = video_input.get(CV_CAP_PROP_FPS) > 0 ? video_input.get(CV_CAP_PROP_FPS) : 25.;
        caps.frame_width = video_input.get(CV_CAP_PROP_FRAME_WIDTH);
        caps.frame_height = video_input.get(CV_CAP_PROP_FRAME_HEIGHT);
    }

    void get_caps(cv::Mat frame) {
        caps.frame_width = frame.cols;
        caps.frame_height = frame.rows;
    }

private:
    bool display_overlay;
    bool display_frame;
    bool display_debug;
    cv::Mat input_image;
    std::string image_output;
    cv::VideoCapture video_input;
    cv::VideoWriter  video_output;
    std::list<cv_module*> modules;
    cv_caps caps;
};


#endif //PIPELINE_CVPIPELINE_H
