#include "fireweight.h"
#include "firebbox.h"
#include "rfiremaskingmodel.h"
#include "firevalidation.h"
#include <omp.h>
#include <opencv2/imgproc.hpp>


fire_weight_distrib::fire_weight_distrib(const json11::Json& fire_weight_distrib_node, cv_caps *capabs_ptr, cv_time timestamp, bool ip_del = false, bool over_draw = false):
    cv_module(capabs_ptr, timestamp, ip_del, over_draw) {
    counter = 0;
    settings.period = (float)fire_weight_distrib_node["period"].number_value();//0.5;
    settings.weight_thr = (float)fire_weight_distrib_node["weight_thresh"].number_value();//0.3;
    settings.time_thr = (float)fire_weight_distrib_node["time_thresh"].number_value();//3;
}

void fire_weight_distrib::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    std::vector<cv_object> fire_bboxes = metadata.fire_valid_bboxes;
    cv::Mat  fire_mask = metadata.fire_mm.r_fire_mask;
    uchar* fire_mask_matr = metadata.fire_mm.r_fire_mask.data;
    uchar* flame_matr = metadata.fire_weight.flame_mask.data;

    if (counter == 0) {
        base = cv::Mat::zeros(fire_mask.rows, fire_mask.cols, CV_32FC1);
        timings = cv::Mat::zeros(fire_mask.rows, fire_mask.cols, CV_8U);
    }
    uchar* timings_matr = timings.data;
    float* base_matr = (float*)base.data;

    if (!draw_overlay) {
    #pragma omp parallel for
        for (int i = 0; i < fire_mask.rows; ++i) {
            for (int j = 0; j < fire_mask.cols; ++j) {
                int id = i * fire_mask.cols + j;
                cv_object* object = getBbox(i, j, fire_bboxes);
                float window = (float)counter / (counter + (float)1.);
                if (object)
                    base_matr[id] = window * base_matr[id] + fire_mask_matr[id] / (counter + (float)1.);
                else
                    base_matr[id] = window * base_matr[id];

                if (base_matr[id] > settings.weight_thr) {
                    timings_matr[id] = (uchar) std::min(timings_matr[id] + 1, 0xff);
                    if (timings_matr[id] > settings.period * settings.time_thr) {
                        flame_matr[id] = 1;
                        metadata.fire_weight.pixel_cnt++;
                    }
                } else
                    timings_matr[id] = 0;
            }
        }
    } else {
        uchar* overlay_matr = overlay.data;
    #pragma omp parallel for
        for (int i = 0; i < fire_mask.rows; ++i) {
            for (int j = 0; j < fire_mask.cols; ++j) {
                int id = i * fire_mask.cols + j;
                cv_object* object = getBbox(i, j, fire_bboxes);
                float window = (float)counter / (counter + (float)1.);
                if (object)
                    base_matr[id] = window * base_matr[id] + fire_mask_matr[id] / (counter + (float)1.);
                else
                    base_matr[id] = window * base_matr[id];

                if (base_matr[id] > settings.weight_thr) {
                    timings_matr[id] = (uchar) std::min(timings_matr[id] + 1, 0xff);
                    if (timings_matr[id] > settings.period * settings.time_thr) {
                        overlay_matr[id * 3]     = 0xff;
                        overlay_matr[id * 3 + 1] = 0;
                        flame_matr[id] = 1;
                        metadata.fire_weight.pixel_cnt++;
                    } else {
                        overlay_matr[id * 3]     = 0;
                        overlay_matr[id * 3 + 1] = 0xff;
                    }
                    overlay_matr[id * 3 + 2] = 0;
                } else
                    timings_matr[id] = 0;
            }
        }
    }

    counter = std::min(counter + 1, (int)metadata.fps);
}
