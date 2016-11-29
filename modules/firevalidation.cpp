#include "firevalidation.h"
#include "rfiremaskingmodel.h"

#include <iostream>
#include <omp.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

fire_validation::fire_validation(const json11::Json &fire_valid_node, cv_caps *capabs_ptr, cv_time timestamp,
                                 bool ip_del, bool over_draw) :
        cv_module(capabs_ptr, timestamp, ip_del, over_draw) {
    settings.alpha1 = (float)fire_valid_node["alpha1"].number_value();//0.25;
    settings.alpha2 = (float)fire_valid_node["alpha2"].number_value();//0.75;
    settings.dma_thresh = (float)fire_valid_node["dma_thresh"].number_value();//12.;
    first_time = true;
}

template<class Type1, class Type2> double fire_validation::dist(cv::Point3_<Type1> p1, cv::Point3_<Type2> p2) {
    return sqrt(pow((float)p1.x - (float)p2.x, 2.) + pow((float)p1.y - (float)p2.y, 2.) + pow((float)p1.z - (float)p2.z, 2.));
}

void fire_validation::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    cv::Mat rgbSignal = metadata.fire_mm.r_fire_mask;

    if (first_time) {
        ema = frame.clone();
        ema.convertTo(ema, CV_32FC3);
        dma = cv::Mat(frame.rows, frame.cols, CV_32FC1);
        first_time = false;
        return;
    }

    uchar* frame_matr       = frame.data;
    uchar* rgbSignal_matr   = rgbSignal.data;
    float* ema_matr         = (float*)ema.data;
    float* dma_matr         = (float*)dma.data;
    uchar* res_mask         = metadata.dynamic_mask.data;

    if (!draw_over) {
    #pragma omp parallel for
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                int id = i * frame.cols + j;
                if (rgbSignal_matr[id]) {
                    for (int k = 0; k < 3; ++k)
                       ema_matr[id * 3 + k] = ema_matr[id * 3 + k] * ((float)1. - settings.alpha1) + (float) frame_matr[id * 3 + k] * settings.alpha1;
                    dma_matr[id] = ((float)1. - settings.alpha2) * dma_matr[id] + (float) sqrt(
                        pow((float) frame_matr[id * 3]      - ema_matr[id * 3],     2.) +
                        pow((float) frame_matr[id * 3 + 1]  - ema_matr[id * 3 + 1], 2.) +
                        pow((float) frame_matr[id * 3 + 2]  - ema_matr[id * 3 + 2], 2.)
                    ) * settings.alpha2;
                } else
                    dma_matr[id] = ((float)1. - settings.alpha2) * dma_matr[id] + settings.alpha2 * 10;
                res_mask[id] = (uchar)(dma_matr[id] >= settings.dma_thresh ? 1 : 0);
            }
        }
    } else {
        uchar* overlay_matr     = overlay.data;
    #pragma omp parallel for
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                int id = i * frame.cols + j;
                if (rgbSignal_matr[id]) {
                    for (int k = 0; k < 3; ++k)
                       ema_matr[id * 3 + k] = ema_matr[id * 3 + k] * ((float)1. - settings.alpha1) + (float) frame_matr[id * 3 + k] * settings.alpha1;
                    dma_matr[id] = ((float)1. - settings.alpha2) * dma_matr[id] + (float) sqrt(
                        pow((float) frame_matr[id * 3]      - ema_matr[id * 3],     2.) +
                        pow((float) frame_matr[id * 3 + 1]  - ema_matr[id * 3 + 1], 2.) +
                        pow((float) frame_matr[id * 3 + 2]  - ema_matr[id * 3 + 2], 2.)
                    ) * settings.alpha2;
                } else
                    dma_matr[id] = ((float)1. - settings.alpha2) * dma_matr[id] + settings.alpha2 * 10;

                if (dma_matr[id] >= settings.dma_thresh) {
                    overlay_matr[id * 3] = 0xff;
                    overlay_matr[id * 3 + 1] = frame_matr[id * 3 + 1];
                    overlay_matr[id * 3 + 2] = frame_matr[id * 3 + 2];
                    res_mask[id] = 1;
                } else {
                    res_mask[id] = 0;
                }
            }
        }
    }
}
