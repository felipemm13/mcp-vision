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

    std::string mainFunction(std::string contourjson, std::string videoUrl, std::string imageUrl, std::string jsonString, std::string frameRate);
    std::string buildFinalOutputFinal(std::vector<MarkAndTime> sequence, int maxFrame);

    bool callApi(const std::string& videoUrl);

    // Global variables
    std::vector<FrameInfo> frames_info;
    std::vector<Contour> contornos;
    std::vector<cv::Point2f> contourCenters;


};


#endif // ComputerVisionWeb_H
