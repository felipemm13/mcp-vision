#ifndef FEETTRACKER_H
#define FEETTRACKER_H

#include "CommonDefinitions.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <deque>
#include <vector>
#include <cfloat>

//#define SHOW_INTERMEDIATE_RESULTS
//#define SHOW_FINAL_RESULTS

class FeetTracker
{
public:
    FeetTracker(unsigned int num_frames);

    static const int frames_to_store;
    static const float phantom_current_weight;
    static const float max_score_rate;
    static const bool saveTracking;
    static const bool showTracking;
    static const std::string saveTrackingDir;
    static const bool saveSteps;
    static const bool showSteps;
    static const bool sendSteps;
    static const std::string saveStepsDir;
    static int min_displacement;
    static const int weights[3];
    static const float max_cm_to_center;
    static const float cm_to_feet_contact;
    static const float objective_coverage_rate;

    int frame_count; //Counts how many frames has been processed
    int real_w, real_h, calib_w, calib_h;

    std::vector<cv::Rect> player_roi; //Player ROI

    //Vectors of feet positions (no order)
    std::map<int, std::vector<cv::Rect> > candidates_per_frame;
    std::vector<cv::Rect> left_rects;
    std::vector<cv::Rect> right_rects;

    //Final info to store
    std::vector<cv::Rect> left_rects_s;
    std::vector<cv::Rect> right_rects_s;
    std::vector<cv::Point2f> left_foot; //Pixel position
    std::vector<cv::Point2f> right_foot; //Pixel position
    std::vector<int> in_objective1;
    std::vector<int> in_objective2;
    std::vector<float> odist1;
    std::vector<float> odist2;
    std::vector<Contour> contours;
    std::vector<int> left_intersects;
    std::vector<int> right_intersects;

    //Step: 0 is not; 1 is beginning; 2 is still stepping
    std::vector<int> left_step;
    std::vector<int> right_step;

    std::vector<int> in_objective;

    std::deque<cv::Mat> stored_oframes; //Original image frames
    std::deque<cv::Mat> stored_mframes; //Player mask frames
    std::deque<int> stored_times; //Frame times

    std::map<int, cv::Mat>  sframes;
    std::map<int, cv::Mat>  smasks;
    std::map<int, cv::Rect> spos;

    std::vector<int> Dx_left; //Position variation from previous frame in x for left foot
    std::vector<int> Dy_left; //Position variation from previous frame in y for left foot
    std::vector<int> Dx_right; //Position variation from previous frame in x for right foot
    std::vector<int> Dy_right; //Position variation from previous frame in y for right foot
    std::vector<int> Dx_left_s; //Smooth position variation from previous frame in x for left foot
    std::vector<int> Dy_left_s; //Smooth position variation from previous frame in y for left foot
    std::vector<int> Dx_right_s; //Smooth position variation from previous frame in x for right foot
    std::vector<int> Dy_right_s; //Smooth position variation from previous frame in y for right foot
    std::vector<bool> from_one_candidate; //Current left/right feet position infered from just one box (poor size estimation)

    std::vector<int> num_candidates; //Candidates in current frame (1 means occlusion)

    int total_frames;

    //Calibration
    cv::Mat Hinv;
    std::vector<cv::Point2f> scenePoints;
    std::map<int, std::vector<cv::Point2i> > objImPos;
    //METHODS
    inline cv::Point2i getRectCenter(cv::Rect &r);
    inline bool firstNearest(cv::Point2i &p, cv::Point2i &p1, cv::Point2i &p2);

    void processSteps(int index, int frame, cv::Mat &current, std::map<int, std::vector<cv::Point2i> > &objectiveImPos);
    void processSteps(int index, int frame, std::map<int, std::vector<cv::Point2i> > &objectiveImPos);
    //void processAvailableStepsWithDistanceToCenter(int index);
    // void processStepsWithDistanceToCenter(int index, int frame, cv::Mat &current);
    void processAvailableStepsWithCoverageArea(int index);
    void processStepsWithCoverageArea(int index, int frame, cv::Mat &current);

    void setFeetPositionsByBBox(int frame, cv::Rect &pos, cv::Mat &pmask);
    void trackPositions(int frame, cv::Rect &pos, cv::Mat &pmask, cv::Mat &current, int msecs, int frame_index);
    void completeTracking(int frame_to_start);
    void smoothDisplacement(int index);
    void smoothBBoxes(int index);
    bool leftStepCriteria(int index);
    bool rightStepCriteria(int index);
    int intersectsObjective(cv::Mat &img, int index, int frame, cv::Rect &left, bool lstep, cv::Rect &right, bool rstep);

    void insideObjective(int index, int frame, cv::Rect &left, bool lstep, cv::Rect &right, bool rstep);

    int insideObjective(cv::Rect &left, bool lstep, cv::Rect &right, bool rstep, std::map<int,
                        std::vector<cv::Point2i> > &objectiveImPos);
    void drawObjectives(cv::Mat &img, int obj, std::map<int, std::vector<cv::Point2i> > &objectiveImPos);
    void drawObjectives(cv::Mat &img, int id);

    void associateLeftAndRight(std::vector<cv::Rect> &fbboxes, cv::Rect &left, cv::Rect &right);
    void associateLeftAndRight(std::vector<cv::Rect> &near_bboxes, int near_frame,
                               std::vector<cv::Rect> &far_bboxes, int far_frame,
                               cv::Rect &current, int cframe, cv::Rect &cleft, cv::Rect &cright, cv::Mat &mask);
    void associateLeftAndRight(std::vector<cv::Rect> &cbboxes, cv::Rect &cleft, cv::Rect &cright, cv::Rect &next_bbox);
    void associateLeftAndRight(std::vector<cv::Rect> &cbboxes, std::vector<cv::Rect> &next_bboxes,
                               cv::Rect &sole_bbox, cv::Rect &cleft, cv::Rect &cright);
    void associateLeftAndRight(std::vector<cv::Rect> &cbboxes, std::vector<cv::Rect> &next_bboxes,
                               cv::Rect &sole_bbox, int diff, cv::Rect &cleft, cv::Rect &cright);
    void associateLeftAndRight(std::vector<cv::Rect> &cbboxes, std::vector<cv::Rect> &n1_bboxes, int d1,
                               std::vector<cv::Rect> &n2_bboxes, int d2, cv::Rect &cleft, cv::Rect &cright);
    void fitRectToMask(cv::Rect &prev, float dx, float dy, cv::Rect &best_fit, cv::Mat &mask, cv::Rect &current);
    void fitRectToMask(int w, int h, cv::Rect &current, cv::Mat &mask, cv::Rect &best_fit);
    void fitRectToMask(int w, int h, cv::Point2i pprev, cv::Rect &best_fit, cv::Mat &mask, cv::Rect &current);
    void fitRectToMask(cv::Rect &prev, int w, int h, float dx, float dy, cv::Mat &mask,
                       cv::Rect &current, cv::Rect &best_fit);
    void fitRectToMask(int x, int y, int w, int h, float dx, float dy, cv::Rect &best_fit, cv::Mat &mask,
                       cv::Rect &current);

    cv::Mat getPhantomTrackingImage(int cindex, int num);
    void saveResult(cv::Mat &image, int frame);
    void saveStepsResult(cv::Mat &image, int frame);
    cv::Point2f transformInv(cv::Point2f p);
    cv::Point2f getStepPosition(int frame, cv::Rect &feet);

    void paintInitialFeetPositionBBoxes(std::vector<cv::Rect> &bboxes, cv::Mat &img, uchar B, uchar G, uchar R);

    bool segmentIntersection(cv::Point2i &o1, cv::Point2i &p1, cv::Point2i &o2, cv::Point2i &p2, cv::Point2i &r);
    std::vector<cv::Point2i> searchSegmentIntersections(cv::Point2i &pp1, cv::Point2i &pp2,
                                                        std::vector<cv::Point2i> &p,
                                                        std::vector<uint> &p_inter_id);
    std::vector<cv::Point2i> intersectConvexPolygons(std::vector<cv::Point2i> &p_wall, std::vector<cv::Point2i> &p);


//STATIC
    static void getBBoxes(cv::Mat &labels, cv::Mat &stats, std::vector<cv::Rect> &bboxes, int i_x, int i_y, std::vector<cv::Point2i> &pointInRegion);
    static void getSamplesAndBBoxesExceptLabel(cv::Mat &labels, cv::Mat &stats,
                                               std::vector<cv::Point2i> &sample_points,
                                               std::vector<cv::Rect> &bboxes, int i_x, int i_y, int label);
    static void getBBoxesExceptLabel(cv::Mat &stats, std::vector<cv::Rect> &bboxes, int i_x, int i_y, int label);
    static float distance(cv::Point2f &p1, cv::Point2f &p2);
};

#endif // FEETTRACKER_H