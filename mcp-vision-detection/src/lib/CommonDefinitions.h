#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <deque>
#include <string>
#include <cfloat>
#include <cmath>
#include <chrono>

#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <nlohmann/json.hpp>

#define _USE_MATH_DEFINES

using json = nlohmann::json;

struct item {
    int code;
    int intersects;
    int frame;
    float d_l;
    float d_r;
    bool step_l;
    bool step_r;
};

struct Section {
    bool error;
    int takeoff_frame;
    int arrival_frame;
    int arrival_code;
    std::vector<item> items;

    Section() : error(false), takeoff_frame(0), arrival_frame(0), arrival_code(0) {}
};

struct Contour {
    int x;
    int y;
    int z;
    int indiceContorno;
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2i> ipoints;
};

struct MarkAndTime {
    int mark_correct;
    int frame;
};

struct Position {
    cv::Point2i heel;
    cv::Point2i foot_index;
    cv::Point2i ankle;
    cv::Point2i center;
};

struct FrameInfo {
    int frame_index;
    bool stepDetection;
    std::string stepSide;
    Position left_position;
    Position right_position;
    cv::Mat segmentation_mask;
};

std::vector<cv::Point2i> intersectConvexPolygons(std::vector<cv::Point2i> &p1, std::vector<cv::Point2i> &p2);
std::vector<cv::Point2i> searchSegmentIntersections(cv::Point2i &pp1, cv::Point2i &pp2, 
                                                    std::vector<cv::Point2i> &p, std::vector<uint> &p_inter_id);
bool segmentIntersection(cv::Point2i &o1, cv::Point2i &p1, cv::Point2i &o2, cv::Point2i &p2, cv::Point2i &r);
bool isPolygonIntersection(std::vector<cv::Point2i> &p1, std::vector<cv::Point2i> &p2);


float dotProduct(const cv::Point2f &u, const cv::Point2f &v);
float magnitude(const cv::Point2f &v);
cv::Point2f projectVector(cv::Point2f v1, const cv::Point2f v2);


#endif // COMMON_DEFINITIONS_H
