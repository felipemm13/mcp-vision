#ifndef Calibration_fixed_H
#define Calibration_fixed_H


#include <iostream>
#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <map>
#include <deque>
#include <string>

#define _USE_MATH_DEFINES
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d_c.h>

#include <chrono>

#include "ConvertImage.h"

//#define SHOW_INTERMEDIATE_RESULTS
//#define SHOW_MAIN_RESULTS
//#define SHOW_TEST

using namespace std;

struct PointWithContour {
    cv::Point3f punto;
    size_t indiceContorno;
    std::vector<cv::Point> contorno;
};

class Calibration_fixed
{
public:
    Calibration_fixed();
    
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

    cv::Mat BGMaskFromYUV(const cv::Mat &image, int Yl, int Yh, cv::Scalar &yuv_bin, int bins);

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

    //Get current interpolated max:
    float interpolatedMax(const std::vector<float> &v, std::vector<bool> &used, int &freq, int &i_low, int &i_up);

    //Evaluates two top most peaks and returns interpolated Y of max frequency:
    float getPeakYInterval(const std::vector<float> &Y, int &freq, int &iY_low, int &iY_high);


    //Finds histogram UV peak and adjusts it interpolating on Y neighborhood (2*nsize)x(2*nsize)x(2*nsize)
    double histoFindYUVMax(const cv::Mat &histo, int bins, cv::Scalar &best_max, int &Y_low, int &Y_high);

    //Finds histogram peak and adjusts it interpolating on neighborhood (2*nsize)x(2*nsize)x(2*nsize)
    double histoFindAdjustedMax(const cv::Mat &histo, int bins, int nsize, cv::Scalar &best_max);

    void histogram3D_masked(const cv::Mat &im, int bins, cv::Mat *histo, cv::Mat &mask);

    //Calculates the 3D histogram on 3-channel color space
    void histogram3D(const cv::Mat &im, int bins, cv::Mat *histo, bool normalize=true);

    //Set scene points as star configuration (5 at center, and 8 markers at 200cm distances, 45° distance each
    void setScenePoints(std::vector<cv::Point2f> &scenePoints);


    //Transforms scene coordinate to image
    cv::Point2i transform(cv::Point2f p, cv::Mat &H);

    //Draws a 15 cm square on image
    std::vector<cv::Point> drawRectangle(cv::Mat &img, cv::Point2f &p, cv::Mat &H);
    //Get image point from scene coordinate
    cv::Point2i getPoint(cv::Point2f p, cv::Mat &H);

    //Draw circular marker and writes id on image
    void addMarker(cv::Mat &img, const cv::Point2i &p, int index);

    //Adds marker with circle, 15 cm box and id, to image
    void addPoint(cv::Mat &img, cv::Point2f &p, cv::Mat &H, int index, std::vector<cv::Point3f> &drawnPoints);
    
    //Adds marker with circle, 15 cm box and id, to image
    std::vector<cv::Point> addPoint_2(cv::Mat &img, cv::Point2f &p, cv::Mat &H, int id);

    static size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata);

    //function to retrieve the image as cv::Mat data type
    cv::Mat curlImg(const char *img_url, int timeout=10);

    // Function that finds the closest black pixel to a given point (mask)
    void findClosestBlackPoints(const cv::Mat& mask, const std::vector<cv::Point3f>& points, std::vector<cv::Point>& closestBlackPoints, cv::Mat& visualizedMask, std::map<int, bool>& marker_availability);


    // Functions to compare the color between two points
    bool areColorsSimilar(cv::Vec3b color1, cv::Vec3b color2, int Tolerance);

    bool compareColorsAt(cv::Mat &image, int x1, int y1, cv::Vec3b color_ref, int luminosityTolerance);

    cv::Vec3b findPredominantColor(const cv::Mat& imagen, int x, int y, int radio);

    // Here we calculate de distances between the 6 points calculated with the algotim vs the 6 first point after H
    void addFixed_marks(std::vector<cv::Point3f> imagePoints_aux, std::vector<cv::Point3f> drawnPoints, cv::Mat& fout_fixed, cv::Mat visualizedMask);

    std::tuple<int,std::string, std::vector<PointWithContour> > getMarksAutomatic(std::string email,std::string screenshot);

    std::pair<int,std::string> getMarksSemiAutomatic(std::string email,std::string screenshot);
};



#endif
