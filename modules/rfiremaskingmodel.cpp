#include "rfiremaskingmodel.h"
#include <opencv2/imgproc.hpp>


r_fire_masking_model::r_fire_masking_model(const json11::Json& ia_node, cv_caps *capabs_ptr, cv_time timestamp, bool draw_overlay, bool ip_deliver) :
    cv_module(capabs_ptr, timestamp, draw_overlay, ip_deliver) {}

void r_fire_masking_model::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    uchar* frame_matr = frame.data;
    uchar* res_matr   = metadata.fire_mm.r_fire_mask.data;

    if (!draw_over) {
    #pragma omp parallel for
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                uchar b = frame_matr[(i * frame.cols + j) * 3],
                      g = frame_matr[(i * frame.cols + j) * 3 + 1],
                      r = frame_matr[(i * frame.cols + j) * 3 + 2];
                res_matr[i * frame.cols + j] = (uchar)(r > g && g > b && r > 190 && g > 100 && b < 140) ? 1 : 0;
                metadata.fire_mm.pixel_cnt += res_matr[i * frame.cols + j];
            }
        }
    } else {
        uchar* over_matr  = overlay.data;

    #pragma omp parallel for
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                uchar b = frame_matr[(i * frame.cols + j) * 3],
                      g = frame_matr[(i * frame.cols + j) * 3 + 1],
                      r = frame_matr[(i * frame.cols + j) * 3 + 2];
                res_matr[i * frame.cols + j] = (r > g && g > b && r > 190 && g > 100 && b < 140) ? 1 : 0;
                if (res_matr[i * frame.cols + j]) {
                   over_matr[(i * frame.cols + j) * 3 + 1] = 0xff;
                   over_matr[(i * frame.cols + j) * 3 + 2] = 0xff;
                   metadata.fire_mm.pixel_cnt++;
                }
            }
        }
    }
}
