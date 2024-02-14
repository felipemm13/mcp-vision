#ifndef Calibration_3_H
#define Calibration_3_H

#include <iostream>
#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <map>
#include <deque>
#include <string.h>


#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <chrono>

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ConvertImage.h"


//#define SHOW_INTERMEDIATE_RESULTS
#define SHOW_MAIN_RESULTS

using namespace std;

class Calibration_3
{
public:
    Calibration_3();
    
    //If only cut boxes remaining
    bool cutBBoxes(std::map<int, cv::Rect> &bboxes);

    //Suppress unviable bboxes based on normal shape features from markers
    void suppressUnviable(std::map<int, cv::Rect> &bboxes, int calibImgW, int calibImgH, cv::Mat &curImgMask);

    //Establish the correction factor for new algorithm iteration, based on lowest bound of remaining bboxes
    int getLowestFromBBoxes(std::map<int, cv::Rect> &bboxes);

    //Establish the correction factor for new algorithm iteration, based on lowest bound of probably cut bboxes
    int getCorrectionToCutBoxes(std::map<int, cv::Rect> &bboxes);

    //Get potential marker position 5, using previously detected markers 1, 2, and 3.
    int getViableCentralBBox(std::map<int, cv::Rect> &bboxes, int y_start,
                            cv::Point2i &ml, cv::Point2i &mc, cv::Point2i &mr, cv::Mat &curImgMask);

    //Get potential marker position 3, using previously detected marker 2 (it also works for detecting position 6, from marker 5).
    int getViableBBoxNearestToMarkerLeft(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask);

    int getViableBBoxHorizontalmostToMarkerRight(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask);

    int getViableBBoxHorizontalmostToMarkerLeft(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask);


    //Get potential marker position 1, using previously detected marker 2 (it also works for detecting position 4, from marker 5).
    int getViableBBoxNearestToMarkerRight(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask);

    //Gets centroid from fg mask inside a rect bbox
    cv::Point2i getMarker(cv::Rect &r, cv::Mat &curImgMask);

    //Gets lowest bbox position and bbox index
    int lowestMarker(std::map<int, cv::Rect> &bboxes, int &yl_index);

    //Calculates and prints stats for bboxes to determine viability parameters
    void candidateStats(std::map<int, cv::Rect> &bboxes, int calibImgW, int calibImgH, cv::Mat &curImgMask);

    //Draw bboxes and id (for debugging)
    void paintRectangles(cv::Mat &img, std::map<int, cv::Rect> &bboxes, cv::Scalar &color);

    //Gets bboxes from connected components image
    void getBlobs(cv::Mat &connected_components, std::map<int, cv::Rect> &bboxes);

    //Obtains foreground mask from initial fg mean and bg mean, according to proximity of each pixel.
    //It also calculates the new mean fg and bg from actual fg and bg pixels, and returns pixel counts.
    cv::Mat FGMaskFromImage(const cv::Mat &image,
                            cv::Scalar &fg, cv::Scalar &bg,
                            cv::Scalar &bg_adj, cv::Scalar &fg_adj, int &count, int &count2);

    //Masked
    //Obtains foreground mask from initial fg mean and bg mean, according to proximity of each pixel.
    //It also calculates the new mean fg and bg from actual fg and bg pixels, and returns pixel counts.
    cv::Mat FGMaskFromImage_masked(const cv::Mat &image,
                            cv::Scalar &fg, cv::Scalar &bg,
                            cv::Scalar &bg_adj, cv::Scalar &fg_adj, int &count, int &count2, cv::Mat &gmask);



    cv::Mat completeNearlyConnected(cv::Mat &img, cv::Mat initial_mask, int T);

    cv::Mat setMaskFromColor(cv::Mat &img, cv::Scalar mask_color, double T);

    //Finds histogram peak and adjusts it interpolating on neighborhood (2*nsize)x(2*nsize)x(2*nsize)
    double histoFindAdjustedMax(const cv::Mat &histo, int bins, int nsize, cv::Scalar &best_max);

    void histogram3D_masked(const cv::Mat &im, int bins, cv::Mat *histo, cv::Mat &mask);

    //Calculates the 3D histogram on 3-channel color space
    void histogram3D(const cv::Mat &im, int bins, cv::Mat *histo);

    //Set scene points as star configuration (5 at center, and 8 markers at 200cm distances, 45° distance each
    void setScenePoints(std::vector<cv::Point2f> &scenePoints);


    //Transforms scene coordinate to image
    cv::Point2i transform(cv::Point2f p, cv::Mat &H);

    //Draws a 15 cm square on image
    void drawRectangle(cv::Mat &img, cv::Point2f &p, cv::Mat &H);

    //Get image point from scene coordinate
    cv::Point2i getPoint(cv::Point2f p, cv::Mat &H);

    //Draw circular marker and writes id on image
    void addMarker(cv::Mat &img, cv::Point2i &p, int index);

    //Adds marker with circle, 15 cm box and id, to image
    void addPoint(cv::Mat &img, cv::Point2f &p, cv::Mat &H, int index);


    static size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata);

    //function to retrieve the image as cv::Mat data type
    cv::Mat curlImg(const char *img_url, int timeout);

    std::pair<int,std::string> getMarksAutomatic(std::string email,std::string screenshot);

    std::pair<int,std::string> getMarksSemiAutomatic(std::string email,std::string screenshot,
                                                    std::string mark1_x,std::string mark1_y,
                                                    std::string mark2_x,std::string mark2_y,
                                                    std::string mark3_x,std::string mark3_y,
                                                    std::string mark4_x,std::string mark4_y,
                                                    std::string mark5_x,std::string mark5_y,
                                                    std::string mark6_x,std::string mark6_y);

};


#endif
