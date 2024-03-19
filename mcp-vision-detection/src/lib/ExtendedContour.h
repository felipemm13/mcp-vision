#ifndef EXTENDEDCONTOUR_H
#define EXTENDEDCONTOUR_H

#include "CommonDefinitions.h"

class extendedContour
{
public:

    //USED
    extendedContour();
    extendedContour(std::vector<cv::Point> c);

    cv::Mat extendContour(cv::Mat &fg, cv::Mat &labels, cv::Mat &rr, cv::Rect &result_rect);
    void findExtremeIndexes();
    void setHIntervals(cv::Mat &labels, int label, int x, int y, int w, std::vector<cv::Point2i> &segs);
    void setVIntervals(cv::Mat &labels, int label, int x, int y, int h, std::vector<cv::Point2i> &segs);
    float getMinimalDistanceToContourSimple(extendedContour &c, cv::Point &own, cv::Point &other);
    void getBoundingBox(std::vector<std::vector<cv::Point> > contours, cv::Rect &result);
    void getFinalContour(cv::Mat &mask);

    std::vector<cv::Point> contour;
    std::vector<cv::Point> nps_big;
    std::vector<cv::Point> nps_near;
    std::vector<std::vector<cv::Point> > cintegrated;
    cv::Mat final_mask;
    std::vector<cv::Point> cfinal;

};

class extendedTrackedContours
{
public:
    extendedTrackedContours();

    std::map<int, extendedContour> contours;

    void addContour(int frame, extendedContour);
};


#endif // EXTENDEDCONTOUR_H