#ifndef MAIN_H
#define MAIN_H

//#define SHOW_INTERMEDIATE_RESULTS
//#define SHOW_FINAL_RESULTS

//#define MEMORY_DEBUG

#include "CommonDefinitions.h"
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

    // Functions declaration
    
    string mainFunction(string contourjson, string videoUrl, string imageUrl, string jsonString, string frameRate);
    string buildOutput(vector<MarkAndTime> sequence, int maxFrame);
    cv::Mat recalibrateHomography();
    cv::Point2f transformInv(cv::Point2f p);
    cv::Point2f getStepPosition(int frame, cv::Rect &feet);
    cv::Point2f imageToScene(cv::Point2i p);

    void processStepsWithCoverageArea(int index, int frame, int cur_objective);
    void processAvailableStepsWithCoverageArea(int index, int cur_objective);
    bool feetIntersectsObjective(cv::Rect rect, vector<cv::Point2i> &contour);
    bool callApi(const string& videoUrl);
    int intersectsObjective(cv::Mat &img, int index, int frame, cv::Rect &leftStep, bool leftStepOccurred, cv::Rect &rightStep, bool rightStepOccurred);
    float calculatePointToLineDistance(const cv::Point2f &pointA, const cv::Point2f &pointB, const cv::Point2f &point);
    float distance(cv::Point2f &p1, cv::Point2f &p2);


    // Global variables
    vector<FrameInfo> frames_info;
    vector<Contour> contornos;
    vector<Contour> contours;
    vector<Contour> contoursScene;
    vector<cv::Point2f> contourCenters;
    vector<cv::Point2f> contourCentersScene;
    const int frames_to_store = 7;
    const float cm_to_feet_contact = 10;
    const float max_cm_to_center = 15;

    //Final info to store
    vector<cv::Rect> left_rects_s;
    vector<cv::Rect> right_rects_s;
    vector<cv::Point2f> left_foot; //Pixel position
    vector<cv::Point2f> right_foot; //Pixel position
    vector<int> in_objective1;
    vector<int> in_objective2;
    vector<float> odist1;
    vector<float> odist2;
    vector<int> left_intersects;
    vector<int> right_intersects;

    //Step: 0 is not; 1 is beginning; 2 is still stepping
    vector<bool> left_step;
    vector<bool> right_step;

    cv::Mat H; 

    int real_w, real_h;
    int calib_w;
    int calib_h;
};


#endif // ComputerVisionWeb_H
