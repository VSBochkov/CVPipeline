#include "firebbox.h"
#include "firevalidation.h"
#include "rfiremaskingmodel.h"
#include "../cvpipeline.h"

#include <omp.h>
#include <opencv2/imgproc.hpp>


fire_bbox::fire_bbox(const json11::Json& fbbox_node, cv_caps *capabs_ptr, time_t timestamp, bool draw_overlay, bool ip_deliver) :
        cv_module(capabs_ptr, timestamp, draw_overlay, ip_deliver)
{
    settings.grav_thresh = fbbox_node["grav_thresh"].number_value();//5.;
    settings.min_area_percent = fbbox_node["min_area_percent"].number_value();//10;
    settings.intersect_thresh = fbbox_node["intersect_thresh"].number_value();//0.4;
    settings.dtime_thresh = fbbox_node["dtime_thresh"].number_value();//0.1;
    settings.aver_bbox_square = fbbox_node["aver_bbox_square"].number_value();//0.;
}


std::vector<cv_object> fire_bbox::calc_bboxes(cv::Mat proc_mask, cv::Mat overlay, ulong pixel_cnt, cv::Scalar bbox_color, cv_metadata &metadata) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(proc_mask.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    std::vector<cv::Moments> mu(contours.size());
    for (uint i = 0; i < contours.size(); i++)
        mu[i] = moments(contours[i], false);

    auto dist2 = [&](double x1, double y1, double x2, double y2) {
        return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    };

    for (uint i = 0; i < contours.size(); ++i) {
        for (uint j = 0; j < contours.size(); ++j) {
            if (i == j) continue;
            double x1 = (double) mu[i].m10 / mu[i].m00;
            double y1 = (double) mu[i].m01 / mu[i].m00;
            double x2 = (double) mu[j].m10 / mu[j].m00;
            double y2 = (double) mu[j].m01 / mu[j].m00;
            double grav = (double) (cv::contourArea(contours[i]) * cv::contourArea(contours[j])) / dist2(x1, y1, x2, y2);
            if (grav > settings.grav_thresh) {
                cv::line(proc_mask, cv::Point((int) x1, (int) y1), cv::Point((int) x2, (int) y2), 0xff, 2);
            }
        }
    }

    contours.clear();
    cv::findContours(proc_mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    double area_thresh = settings.min_area_percent / 100. * pixel_cnt;
    std::vector<cv::Rect> bboxes;
    for (uint i = 0; i < contours.size(); ++i) {
        auto bbox = cv::boundingRect(contours[i]);
        if (bbox.area() > area_thresh)
            bboxes.push_back(bbox);
    }

    std::vector<cv::Rect> merged_bboxes;
    for (uint i = 0; i < bboxes.size(); ++i) {
        cv::Rect merge_rect;
        for (uint j = 0; j < bboxes.size(); ++j) {
            if (i == j) continue;
            cv::Rect intersect = intersection(bboxes[i], bboxes[j]);
            if (std::max((double) intersect.area() / bboxes[i].area(), (double) intersect.area() / bboxes[j].area()) > settings.intersect_thresh)
                merge_rect = rect_union(bboxes[i], bboxes[j]);
        }
        merged_bboxes.push_back(merge_rect.area() > 0 ? merge_rect : bboxes[i]);
    }
    bboxes = merged_bboxes;
    if (!base_bboxes.empty()) {
        for (auto& bbox : bboxes) {
            double best_rel_koeff = 0.;
            cv_object* target_obj;
            for (auto& base_bbox : base_bboxes) {
                if (intersection(bbox, base_bbox.bbox).area() > settings.intersect_thresh) {
                    double relation_koeff = bbox.area() > base_bbox.bbox.area() ?
                                            ((double)base_bbox.bbox.area() / (double)bbox.area()) :
                                            ((double)bbox.area() / (double)base_bbox.bbox.area());
                    if (relation_koeff > best_rel_koeff) {
                        best_rel_koeff = relation_koeff;
                        target_obj = &base_bbox;
                    }
                }
            }
            if (best_rel_koeff > 0.) {
                target_obj->timestamp = metadata.timestamp;
                target_obj->bbox = bbox;
            } else
                base_bboxes.push_back(cv_object(0, metadata.timestamp, bbox));
        }
    } else {
        base_bboxes.reserve(bboxes.size());
        for (cv::Rect& bbox : bboxes)
            base_bboxes.push_back(cv_object(0, metadata.timestamp, bbox));
    }

    std::vector<cv_object> result_bboxes;

    int deleted = 0;
    for (auto& base_bbox : base_bboxes) {
        if ((metadata.timestamp - base_bbox.timestamp + base_bbox.lifetime).millis() <= (unsigned long long) (settings.dtime_thresh * 1000)) {
            if (draw_overlay)
                cv::rectangle(overlay, base_bbox.bbox, bbox_color, 1);

            result_bboxes.push_back(base_bbox);
        } else
            deleted++;
    }

    if (result_bboxes.empty())
        return result_bboxes;

    double aver_bbox_square = 0.;
    for (auto& base_bbox : result_bboxes)
        aver_bbox_square += base_bbox.bbox.area();

    aver_bbox_square /= result_bboxes.size();

    if (aver_bbox_square < settings.min_area_percent / 10000. * (proc_mask.rows * proc_mask.cols) && settings.grav_thresh > 1)
        settings.grav_thresh -= 0.5;
    else if (aver_bbox_square > (proc_mask.rows * proc_mask.cols) / 50.)
        settings.grav_thresh += 0.5;

    /*if (ip_deliever) {
        ip_mutex->lock();
        QDataStream out(&process_data.data_serialized, QIODevice::WriteOnly);
        if (!result_bboxes.empty())
            out << qint16((short)result_bboxes.size());
        for (auto& obj : result_bboxes)
            out << obj;
        ip_mutex->unlock();
    }*/

    return result_bboxes;
}

void fire_bbox::compute(cv::Mat& frame, cv_metadata& metadata, cv::Mat& overlay) {
    metadata.fire_valid_bboxes = calc_bboxes(
            metadata.dynamic_mask.clone(), overlay, metadata.fire_mm.pixel_cnt,
            cv::Scalar(0, 0xff, 0xff), metadata
    );
}
