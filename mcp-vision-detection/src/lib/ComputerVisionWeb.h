#ifndef MAIN_H
#define MAIN_H

//#define SHOW_INTERMEDIATE_RESULTS
//#define SHOW_FINAL_RESULTS

//#define MEMORY_DEBUG

#include "CommonDefinitions.h"
#include "FeetTracker.h"
#include "ExtendedContour.h"

using namespace std;

//Per frame: Two feet. By foot: (x y w h code xp yp d)
// (x,y,w,h): foot rect                (left_step, right_step)
// code:                               (in_objective1, in_objective2)
//     0: No step
//   1-9: Step to nearest objective
// (xp,yp): Feet contact point         (left_foot, right_foot)
// d: distance to nearest center       (odist1, odist2)
class ComputerVisionWeb
{
public:
    ComputerVisionWeb();

    cv::Point2f imageToScene(cv::Point2i p);
    std::string mainFunction(std::string contourjson, std::string videoUrl, std::string imageUrl, std::string jsonString, std::string frameRate);
    std::string buildOutput(std::vector<MarkAndTime> sequence, int maxFrame);

    bool callApi(const std::string& videoUrl);

    // Global variables
    std::vector<FrameInfo> frames_info;
    std::vector<Contour> contornos;
    std::vector<Contour> contoursScene;
    std::vector<cv::Point2f> contourCenters;
    std::vector<cv::Point2f> contourCentersScene;


    //Final info to store
    std::vector<cv::Rect> left_rects_s;
    std::vector<cv::Rect> right_rects_s;
    std::vector<cv::Point2f> left_foot; //Pixel position
    std::vector<cv::Point2f> right_foot; //Pixel position
    std::vector<int> in_objective1;
    std::vector<int> in_objective2;
    std::vector<float> odist1;
    std::vector<float> odist2;
    std::vector<int> left_intersects;
    std::vector<int> right_intersects;

    //Step: 0 is not; 1 is beginning; 2 is still stepping
    std::vector<bool> left_step;
    std::vector<bool> right_step;

    cv::Mat H; 

    int real_w, real_h;
    int calib_w;
    int calib_h;
};


#endif // ComputerVisionWeb_H
