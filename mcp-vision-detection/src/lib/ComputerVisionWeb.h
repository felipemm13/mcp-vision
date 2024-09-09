#ifndef MAIN_H
#define MAIN_H

//#define SHOW_INTERMEDIATE_RESULTS
//#define SHOW_FINAL_RESULTS

//#define MEMORY_DEBUG

#include "CommonDefinitions.h"
#include "FeetTracker.h"
#include "ExtendedContour.h"

using namespace std;

class ComputerVisionWeb
{
public:
    ComputerVisionWeb();
    static size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata);
    //function to retrieve the image as cv::Mat data type
    cv::Mat curlImg(const char *img_url, int timeout);
    void initTracker(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog);
    void trainMog(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog, cv::Mat &img, cv::Mat &fg, cv::Mat &bg, double learningRate);
    void trainMog(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog, cv::Mat &img, cv::Mat &fg, double learningRate);

    cv::Mat maskBiggest(cv::Mat &fg, cv::Mat &labels, cv::Mat &stats, cv::Mat &big_mask);
    //// Segmentation and Foot Boxes ////
    cv::Mat presegmentation(cv::Ptr<cv::BackgroundSubtractorMOG2> mog, cv::Mat &current, cv::Mat &labels, cv::Mat &r);
    void setScenePoints(std::vector<cv::Point2f> &scenePoints);
    cv::Point2i transform(cv::Point2f p);
    cv::Point2i getPoint(cv::Point2f p);

    int getCalibrationData(const std::string query, cv::Mat &H, int &w, int &h);

    //Per frame: Two feet. By foot: (x y w h code xp yp d)
    // (x,y,w,h): foot rect                (left_step, right_step)
    // code:                               (in_objective1, in_objective2)
    //     0: No step
    //   1-9: Step to nearest objective
    // (xp,yp): Feet contact point         (left_foot, right_foot)
    // d: distance to nearest center       (odist1, odist2)
    
    std::string buildJsonData(FeetTracker &ft);
    std::string mainFunction(std::string contourjson, std::string videoUrl, std::string imageUrl, std::string jsonString, std::string frameRate);
    std::string buildFinalOutput(std::string jsonData, std::vector<MarkAndTime> sequence);
    std::string buildFinalOutputImproved(FeetTracker &ft, std::vector<MarkAndTime> sequence, int maxFrame);
    std::string buildFinalOutputFinal(FeetTracker &ft, std::vector<MarkAndTime> sequence, int maxFrame);

    bool callApi(const std::string& videoUrl);
};


#endif // ComputerVisionWeb_H
