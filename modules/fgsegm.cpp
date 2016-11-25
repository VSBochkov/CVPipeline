//
// Created by vbochkov on 23.05.16.
//

#include "fgsegm.h"

fgsegm::fgsegm(
        const json11::Json& fgsegm_node, cv_caps *capabs_ptr, time_t timestamp
): cv_module(capabs_ptr, timestamp) {
    settings.min_contour_perim   = fgsegm_node["min_contour_perim"].int_value();
    settings.thresh_gravity      = fgsegm_node["thresh_gravity"].number_value();
    settings.dilation_size       = fgsegm_node["dilation_size"].int_value();
    settings.opening_size        = fgsegm_node["opening_size"].int_value();
    settings.dist_mx             = fgsegm_node["dist_mx"].number_value();
    settings.dist_my             = fgsegm_node["dist_my"].number_value();
}

void fgsegm::mask_prepare(cv::Mat mask_scaled) {
    double hx = caps->frame_width / IMPROC_WIDTH;
    double hy = caps->frame_height / IMPROC_HEIGHT;

    cv::Mat opening_kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            hy < hx ? cv::Size((int) (hy * settings.opening_size / hx), settings.opening_size) :
            cv::Size(settings.opening_size, (int) (hy * settings.opening_size / hx))
    );
    cv::Mat dilation_kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            hy < hx ? cv::Size((int) (hy * settings.dilation_size / hx), settings.dilation_size) :
            cv::Size(settings.dilation_size, (int) (hy * settings.dilation_size / hx))
    );
    cv::morphologyEx(mask_scaled, mask_scaled, cv::MORPH_OPEN, opening_kernel);
    cv::dilate(mask_scaled, mask_scaled, dilation_kernel);
}

void fgsegm::gravitation(cv::Mat mask_scaled) {
    cv::Mat fg_mask;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    mask_scaled.copyTo(fg_mask);
    cv::findContours(fg_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        cv::Moments mu1 = cv::moments(contours[i]);
        for (int j = 0; j < contours.size(); ++j) {
            if (i == j) continue;
            cv::Moments mu2 = cv::moments(contours[j]);
            if (mu2.m00 > 0.) {
                cv::Point2f mc1(mu1.m10 / mu1.m00, mu1.m01 / mu1.m00);
                cv::Point2f mc2(mu2.m10 / mu2.m00, mu2.m01 / mu2.m00);
                double grav = mu1.m00 * mu2.m00 / (settings.dist_mx * pow(mc1.x - mc2.x, 2.) + settings.dist_my * pow(mc1.y - mc2.y, 2.));
                if (grav > settings.thresh_gravity)
                    cv::line(
                            mask_scaled,
                            cv::Point((int) mc1.x, (int) mc1.y),
                            cv::Point((int) mc2.x, (int) mc2.y),
                            cv::Scalar(0xff, 0xff, 0xff),
                            15
                    );
            }
        }
    }
}

std::list<rect> fgsegm::get_bboxes(cv::Mat mask_scaled) {
    double hx = caps->frame_width / IMPROC_WIDTH;
    double hy = caps->frame_height / IMPROC_HEIGHT;
    cv::Mat fg_mask;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    mask_scaled.copyTo(fg_mask);
    cv::findContours(fg_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::list<rect> bboxes;
    for (auto& cont: contours) {
        cv::Rect r = cv::boundingRect(cont);
        rect bbox(r, hx, hy);
        if (bbox.perimeter() > settings.min_contour_perim)
            bboxes.push_back(bbox);
    }
    return bboxes;
}

void fgsegm::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    if (!metadata.fg_mask.empty()) {
        mask_prepare(metadata.fg_mask);
        gravitation(metadata.fg_mask);
        metadata.bboxes = get_bboxes(metadata.fg_mask);
    }
    cv_module::compute(frame, metadata, overlay);
}