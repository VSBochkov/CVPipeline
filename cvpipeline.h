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


struct cv_time {
    time_t   time;
    clock_t  clocks;

    cv_time(bool current = true)
    {
        if (current)
            get_time();
        else
            null_time();
    }

    cv_time& operator-(cv_time& from)
    {
        difftime(time, from.time);
        if (from.clocks > clocks)
        {
            time -= 1;
            clocks = CLOCKS_PER_SEC + clocks - from.clocks;
        }
    }

    cv_time& operator+(cv_time& arg)
    {
        clocks += arg.clocks;
        time += arg.time + (time_t) (clocks / CLOCKS_PER_SEC);
        clocks %= CLOCKS_PER_SEC;
    }

    void get_time()
    {
        time = time(NULL);
        clocks = clock() % CLOCKS_PER_SEC;
    }

    void null_time()
    {
        struct tm null_time;
        null_time.tm_year = 0; null_time.tm_mon = 0; null_time.tm_yday = 0;
        null_time.tm_hour = 0; null_time.tm_min = 0; null_time.tm_sec = 0;
        time = mktime(&null_time);
        clocks = 0;
    }

    unsigned long long millis()
    {
        unsigned long long res = time * 1000;
        return res + clocks * 1000 / CLOCKS_PER_SEC;
    }
};

struct cv_object {
    int id;
    cv_time timestamp;
    cv_time lifetime;
    cv::Rect bbox;
    std::string value;
    cv_object(
            int o_id, cv_time o_timestamp = cv_time(true), cv::Rect o_bbox = cv::Rect(),
            std::string o_value = ""//, cv_object *o_linked_obj = NULL
    ) {
        id = o_id;
        bbox = o_bbox;
        timestamp = o_timestamp;
        lifetime = cv_time(false);
        value = o_value;
    }

    std::ostream& operator<<(std::ostream &out)
    {
        out << bbox.x << bbox.y << bbox.height << bbox.width;
    }

    std::istream& operator>>(std::istream &in)
    {
        in >> bbox.x >> bbox.y >> bbox.height >> bbox.width;
    }
};

struct cv_metadata {
    double fps;
    cv_time timestamp;
    cv::Mat fg_mask;
    std::list<cv::Rect> bboxes;
    std::list<cv_object> objects;
    std::list<std::string> events;
    struct fire_mm_data {
        cv::Mat r_fire_mask;
        int pixel_cnt;
    } fire_mm;
    cv::Mat dynamic_mask;
    std::vector<cv_object> fire_valid_bboxes;
    cv::Mat static_mask;
    cv::Mat flame_mask;
    std::vector<cv_object> flame_bboxes;
};

struct cv_caps {
    double frame_width;
    double frame_height;
    double fps;
};

class cv_module {
public:
    cv_module(cv_caps* capabs_ptr, time_t current_time, bool draw = false, bool ip = false) {
        caps = capabs_ptr;
        init_time = current_time;
        draw_over = draw;
        ip_deliver = ip;
    }
    virtual ~cv_module() {};
    virtual void compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
        if (draw_over)
            this->draw_overlay(overlay);
    };
    virtual void draw_overlay(cv::Mat overlay) {}

private:
    cv_module(const cv_module&);
    cv_module& operator=(const cv_module&);

protected:
    cv_caps* caps;
    time_t init_time;
public:
    bool draw_over;
    bool ip_deliver;
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
