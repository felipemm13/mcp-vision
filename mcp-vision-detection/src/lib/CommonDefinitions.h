#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <opencv2/core/types.hpp>
#include <vector>

struct Contour {
    int x;
    int y;
    int z;
    int indiceContorno;
    std::vector<cv::Point2f> points;
};

struct MarkAndTime {
    int mark_correct;
    int frame;
};

struct item {
    int code;
    int intersects;
    int frame;
    float d_l;
    float d_r;
};

#endif // COMMON_DEFINITIONS_H
