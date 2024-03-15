#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <vector>

struct Point {
    int x;
    int y;
};

struct Contour {
    int x;
    int y;
    int z;
    int indiceContorno;
    std::vector<Point> points;
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
