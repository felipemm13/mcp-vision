#include "FeetTracker.h"

const int FeetTracker::frames_to_store = 7; // Must be odd number (min 5)

const bool FeetTracker::saveTracking = false;
const bool FeetTracker::showTracking = false;
const std::string FeetTracker::saveTrackingDir = "TRACKING_RESULTS";
const bool FeetTracker::saveSteps = false;
const bool FeetTracker::showSteps = false;
const bool FeetTracker::sendSteps = false;
const std::string FeetTracker::saveStepsDir = "STEPS_RESULTS";
const float FeetTracker::phantom_current_weight = 0.6;

const int FeetTracker::weights[3] = {4, 2, 1};
int FeetTracker::min_displacement = 3;
const float FeetTracker::max_score_rate = 0.7;

const float FeetTracker::max_cm_to_center = 15;
const float FeetTracker::cm_to_feet_contact = 10;
const float FeetTracker::objective_coverage_rate = 0.02;

FeetTracker::FeetTracker(uint num_frames) : frame_count(0), left_foot(num_frames), right_foot(num_frames)
{
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool FeetTracker::segmentIntersection(cv::Point2i &o1, cv::Point2i &p1, cv::Point2i &o2, cv::Point2i &p2, cv::Point2i &r)
{
    float s02_x, s02_y, s10_x, s10_y, s32_x, s32_y, s_numer, t_numer, denom, t;
    s10_x = p1.x - o1.x;
    s10_y = p1.y - o1.y;
    s32_x = p2.x - o2.x;
    s32_y = p2.y - o2.y;

    denom = s10_x * s32_y - s32_x * s10_y;
    if (denom == 0)
        return false; // Collinear
    bool denomPositive = denom > 0;

    s02_x = o1.x - o2.x;
    s02_y = o1.y - o2.y;
    s_numer = s10_x * s02_y - s10_y * s02_x;
    if ((s_numer < 0) == denomPositive)
        return false; // No collision

    t_numer = s32_x * s02_y - s32_y * s02_x;
    if ((t_numer < 0) == denomPositive)
        return false; // No collision

    if (((s_numer > denom) == denomPositive) || ((t_numer > denom) == denomPositive))
        return false; // No collision
    // Collision detected
    t = t_numer / denom;
    r.x = o1.x + (t * s10_x);
    r.y = o1.y + (t * s10_y);

    return true;
}

std::vector<cv::Point2i> FeetTracker::searchSegmentIntersections(cv::Point2i &pp1, cv::Point2i &pp2,
                                                                 std::vector<cv::Point2i> &p,
                                                                 std::vector<uint> &p_inter_id)
{
    cv::Point2i r;
    std::vector<cv::Point2i> rr;
    uint psize = p.size();
    for (uint i = 0; i < psize; ++i)
        if (segmentIntersection(pp1, pp2, p[i], p[(i + 1) % psize], r))
        { // There is intersectoin
            p_inter_id.push_back(i);
            rr.push_back(r);
        }
    return rr;
}

// For Convex polygon, max 2 intersections per segment...
std::vector<cv::Point2i> FeetTracker::intersectConvexPolygons(std::vector<cv::Point2i> &p_wall, std::vector<cv::Point2i> &p)
{
    uint psize = p.size(), num_in = 0;
    int first_in = -1;
    std::vector<int> in(psize);
    for (uint i = 0; i < psize; ++i)
        if ((in[i] = cv::pointPolygonTest(p_wall, p[i], true)) >= 0)
        {
            if (first_in == -1)
                first_in = i;
            ++num_in;
        }

    // Non inside: return None
    if (num_in == 0)
    {
        std::vector<int> in2(psize);
        uint pwsize = p_wall.size(), num_in2 = 0;
        for (uint i = 0; i < pwsize; ++i)
            if ((in2[i] = cv::pointPolygonTest(p, p_wall[i], true)) > 0)
                num_in2++;

        // Return wall if all wall points inside
        if (num_in2 == pwsize)
            return p_wall;

        // If some point of wall inside, and non of the polygon p, case is covered inverting order.
        if (num_in2 > 0)
            return intersectConvexPolygons(p, p_wall);

        // Else, check for intersection points, and add them (no point of one polygon into the other)
        std::vector<cv::Point2i> r, r2;
        bool added = false;
        for (uint j = 0; j < psize; ++j)
        {
            std::vector<uint> wall_inter_id;
            uint next = (j + 1) % psize;
            std::vector<cv::Point2i> intersections = searchSegmentIntersections(p[j], p[next], p_wall, wall_inter_id);
            uint num_int = intersections.size();
            for (uint k = 0; k < num_int; ++k)
            {
                r.push_back(intersections[k]);
                added = true;
            }
        }

        if (added)
        {
            cv::convexHull(r, r2);
            return r2;
        }
        // If no input, return empty polygon
        return std::vector<cv::Point2i>();
    }
    else if (num_in == psize)
        return p;

    std::vector<int> in2(psize);
    uint pwsize = p_wall.size(), num_in2 = 0;
    for (uint i = 0; i < pwsize; ++i)
        if ((in2[i] = cv::pointPolygonTest(p, p_wall[i], true)) > 0)
            num_in2++;

    uint i = first_in, j = 0;
    int other_p_out = -1;
    std::vector<cv::Point2i> r;

    while (j <= psize)
    {
        if (in[i] >= 0)
        { // If current point in, add it
            if (other_p_out != -1)
            { // It comes from the outside
                std::vector<uint> wall_inter_id;
                uint prev = i == 0 ? psize - 1 : i - 1;
                std::vector<cv::Point2i> intersections = searchSegmentIntersections(p[prev], p[i], p_wall, wall_inter_id);
                uint num_int = intersections.size();
                if (num_int != 1)
                { // If there is no just one, coming from inside, the other polygon is not convex?
                    std::cerr << "Something is wrong!! Should have intersected!" << std::endl;
                    return r;
                }

                if (other_p_out != wall_inter_id[0])
                { // It means we need to add wall points...
                    // Check begginning, end and sense:
                    uint beggining;
                    if (in2[other_p_out] < 0) // Starting point of wall segment is out, start from next;
                        beggining = (other_p_out + 1) % pwsize;
                    else
                        beggining = other_p_out;
                    if (in2[(beggining + 1) % pwsize] > 0)
                        for (uint k = beggining; in2[k % pwsize] > 0; ++k)
                            r.push_back(p_wall[k % pwsize]);
                    else
                        for (uint k = beggining; in2[k % pwsize] > 0; --k)
                            r.push_back(p_wall[k % pwsize]);
                }
                // Reaching here, we add intersection point
                r.push_back(intersections[0]);
            }
            // Finally we add current inner point
            if (j < psize)
                r.push_back(p[i]);
            other_p_out = -1;
        }
        else
        { // If not... add intersections and inner points of the other polygons
            std::vector<uint> wall_inter_id;
            cv::Point2i last = r.back();
            uint prev = i == 0 ? psize - 1 : i - 1;
            std::vector<cv::Point2i> intersections = searchSegmentIntersections(p[prev], p[i], p_wall, wall_inter_id);
            uint num_int = intersections.size();
            if (other_p_out == -1)
            { // Comes from inside
                if (num_int != 1)
                { // If there is no just one, coming from inside, the other polygon is not convex?
                    std::cerr << "Something is wrong!! Should have intersected!" << std::endl;
                    return r;
                }
                // Reaching here, we add intersection point
                r.push_back(intersections[0]);
                other_p_out = wall_inter_id[0];
            }
            else
            { // Comes from outside: Could be double intersection or none (if none, ignore)
                if (num_int == 2)
                { // Add both intersections, nearest first
                    cv::Point2i pp1 = intersections[0], pp2 = intersections[1];
                    float d1 = sqrt((last.x - pp1.x) * (last.x - pp1.x) + (last.y - pp1.y) * (last.y - pp1.y)),
                          d2 = sqrt((last.x - pp2.x) * (last.x - pp2.x) + (last.y - pp2.y) * (last.y - pp2.y));
                    if (d1 < d2)
                    {
                        r.push_back(pp1);
                        r.push_back(pp2);
                        other_p_out = wall_inter_id[1];
                    }
                    else if (d1 > d2)
                    {
                        r.push_back(pp2);
                        r.push_back(pp1);
                        other_p_out = wall_inter_id[0];
                    }
                    else
                    { // Same point. A corner? Add one...
                        r.push_back(pp1);
                        other_p_out = wall_inter_id[0] > wall_inter_id[1] ? wall_inter_id[0] : wall_inter_id[1];
                    }
                }
                else if (num_int != 0)
                { // If is not 0 or 2, the other polygon is not convex?
                    std::cerr << "Something is wrong!! Not possible in convex polygons!" << std::endl;
                    return r;
                }
            }
        }
        i = (i + 1) % psize;
        ++j;
    }

    return r;
}

inline cv::Point2i FeetTracker::getRectCenter(cv::Rect &r)
{
    return cv::Point2i(round(r.x + r.width / 2), round(r.y + r.height / 2));
}

inline bool FeetTracker::firstNearest(cv::Point2i &p, cv::Point2i &p1, cv::Point2i &p2)
{
    if (abs(p.x - p1.x) + abs(p.y - p1.y) < abs(p.x - p2.x) + abs(p.y - p2.y))
        return true;
    return false;
}

void FeetTracker::processSteps(int index, int frame, std::map<int, std::vector<cv::Point2i>> &objectiveImPos)
{
    smoothDisplacement(index);
    smoothBBoxes(index);

    cv::Rect &left = left_rects_s[index], &right = right_rects_s[index];

    if (leftStepCriteria(index))
    {
        left_step[index] = 1;
    }

    if (rightStepCriteria(index))
    {
        right_step[index] = 1;
    }

    //int obj = insideObjective(left, left_step[index], right, right_step[index], objectiveImPos);
    //in_objective.push_back(obj);
#ifdef SHOW_INTERMEDIATE_RESULTS
    if (obj > 0)
    {
        std::cout << "Inside objective " << obj << std::endl;
    }
    std::cout << "Processed step frame: " << frame << std::endl;
#endif
}

/*void FeetTracker::processAvailableStepsWithDistanceToCenter(int index)
{
    int pos_correction = frames_to_store / 2 + 3;
    if (index >= pos_correction)
    {
        int sframe = index - pos_correction + 1;
        processStepsWithDistanceToCenter(index - pos_correction, sframe, sframes[sframe]);
        sframes.erase(sframe);
        smasks.erase(sframe);
        spos.erase(sframe);
    }
}*/

void FeetTracker::processAvailableStepsWithCoverageArea(int index)
{
    int pos_correction = frames_to_store / 2 + 3;
    if (index >= pos_correction)
    {
        int sframe = index - pos_correction + 1;
        processStepsWithCoverageArea(index - pos_correction, sframe, sframes[sframe]);
        sframes.erase(sframe);
        smasks.erase(sframe);
        spos.erase(sframe);
    }
}

// void FeetTracker::processStepsWithDistanceToCenter(int index, int frame, cv::Mat &current)
// {
//     smoothDisplacement(index);
//     smoothBBoxes(index);

//     cv::Rect &left = left_rects_s[index], &right = right_rects_s[index];

//     if (leftStepCriteria(index))
//     {
//         left_step[index] = 1;
//     }

//     if (rightStepCriteria(index))
//     {
//         right_step[index] = 1;
//     }

//     //insideObjective(index, frame, left, left_step[index], right, right_step[index]);

// #ifdef SHOW_FINAL_RESULTS
//     std::cout << "Processed step index: " << index << std::endl;
//     std::cout << "Processed step frame: " << frame << std::endl;

//     if (left_step[index])
//     {
//         std::cout << "Step on left foot:" << this->odist1[index] << " to " << in_objective1[index] << " objective." << std::endl;
//         cv::Point2f p = transformInv(left_foot[index]);
//         std::cout << "Left foot position:" << p.x << ", " << p.y << std::endl;
//     }

//     if (right_step[index])
//     {
//         std::cout << "Step on right foot:" << this->odist2[index] << " to " << in_objective2[index] << " objective." << std::endl;
//         cv::Point2f p = transformInv(right_foot[index]);
//         std::cout << "Right foot position:" << p.x << ", " << p.y << std::endl;
//     }

//     cv::Mat cur_copy2;

//     int index_contour = intersectsObjective(cur_copy2, index, frame, left, left_step[index], right, right_step[index]);

//     sframes[frame].copyTo(cur_copy2);
//     if (left_step[index])
//         cv::rectangle(cur_copy2, left, cv::Scalar(0, 255, 0));
//     else
//         cv::rectangle(cur_copy2, left, cv::Scalar(0, 0, 255)); // Left
//     if (right_step[index])
//         cv::rectangle(cur_copy2, right, cv::Scalar(0, 255, 0));
//     else
//         cv::rectangle(cur_copy2, right, cv::Scalar(0, 255, 255)); // Right
        
//     drawObjectives(cur_copy2 , index_contour+1);

//     if (left_step[index])
//         cv::circle(cur_copy2, left_foot[index], 1, cv::Scalar(0, 0, 255));
//     if (right_step[index])
//         cv::circle(cur_copy2, right_foot[index], 1, cv::Scalar(0, 0, 255));
//     // cv::resize(cur_copy2, cur_copy2, cv::Size(4*cur_copy2.cols, 4*cur_copy2.rows));
//     cv::imshow("Everything", cur_copy2);
//     //    saveResult(cur_copy2, frame);
//     cv::waitKey(1);
// #endif
// }

void FeetTracker::processStepsWithCoverageArea(int index, int frame, cv::Mat &current)
{
    smoothDisplacement(index);
    smoothBBoxes(index);

    cv::Rect &left = left_rects_s[index], &right = right_rects_s[index];
    if (leftStepCriteria(index))
    {
        left_step[index] = 1;
    }

    if (rightStepCriteria(index))
    {
        right_step[index] = 1;
    }

#ifdef SHOW_FINAL_RESULTS
    std::cout << "Processed step index: " << index << std::endl;
    std::cout << "Processed step frame: " << frame << std::endl;

    if (left_step[index])
    {
        std::cout << "Step on left foot:" << this->odist1[index] << " to " << in_objective1[index] << " objective." << std::endl;
        cv::Point2f p = transformInv(left_foot[index]);
        std::cout << "Left foot position:" << p.x << ", " << p.y << std::endl;
    }

    if (right_step[index])
    {
        std::cout << "Step on right foot:" << this->odist2[index] << " to " << in_objective2[index] << " objective." << std::endl;
        cv::Point2f p = transformInv(right_foot[index]);
        std::cout << "Right foot position:" << p.x << ", " << p.y << std::endl;
    }

    cv::Mat cur_copy2;

    int index_contour = intersectsObjective(cur_copy2, index, frame, left, left_step[index], right, right_step[index]);

    sframes[frame].copyTo(cur_copy2);
    if (left_step[index])
        cv::rectangle(cur_copy2, left, cv::Scalar(0, 255, 0));
    //else
        //cv::rectangle(cur_copy2, left, cv::Scalar(0, 0, 255)); // Left
    if (right_step[index])
        cv::rectangle(cur_copy2, right, cv::Scalar(0, 255, 0));
    //else
    //    cv::rectangle(cur_copy2, right, cv::Scalar(0, 255, 255)); // Right

    drawObjectives(cur_copy2 , index_contour+1);
    
    if (left_step[index])
        cv::circle(cur_copy2, left_foot[index], 1, cv::Scalar(0, 0, 255));
    if (right_step[index])
        cv::circle(cur_copy2, right_foot[index], 1, cv::Scalar(0, 0, 255));
    // cv::resize(cur_copy2, cur_copy2, cv::Size(4*cur_copy2.cols, 4*cur_copy2.rows));

    cv::namedWindow("Everything", cv::WINDOW_NORMAL);
    cv::resizeWindow("Everything", 1920, 1000);
    cv::imshow("Everything", cur_copy2);
    saveResult(cur_copy2, frame);
    cv::waitKey(1);
#endif
}

void FeetTracker::processSteps(int index, int frame, cv::Mat &current, std::map<int, std::vector<cv::Point2i>> &objectiveImPos)
{
    cv::Rect proi = player_roi[index];
    cv::Mat cur_copy2;
    if (showSteps || saveSteps)
    {
        current.copyTo(cur_copy2);
    }

    smoothDisplacement(index);
    smoothBBoxes(index);

    cv::Rect &left = left_rects_s[index], &right = right_rects_s[index];

    if (leftStepCriteria(index))
    {
        left_step[index] = 1;
        if (showSteps || saveSteps)
            cv::rectangle(cur_copy2, left, cv::Scalar(0, 255, 0));
    }
    else
    {
        if (showSteps || saveSteps)
            cv::rectangle(cur_copy2, left, cv::Scalar(0, 0, 255)); // Left
    }

    if (rightStepCriteria(index))
    {
        right_step[index] = 1;
        //        cv::rectangle(cur_copy, right_rects[index], cv::Scalar(0,255,0));
        if (showSteps || saveSteps)
            cv::rectangle(cur_copy2, right, cv::Scalar(0, 255, 0));
    }
    else
    {
        if (showSteps || saveSteps)
            cv::rectangle(cur_copy2, right, cv::Scalar(0, 255, 255)); // Right
    }

    //int obj = insideObjective(left, left_step[index], right, right_step[index], objectiveImPos);
    //in_objective.push_back(obj);
#ifdef SHOW_INTERMEDIATE_RESULTS
    if (obj > 0)
    {
        std::cout << "Inside objective " << obj << std::endl;
    }
#endif
    //if (showSteps || saveSteps)
    //    drawObjectives(cur_copy2, obj, objectiveImPos);

#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "Processed step frame: " << frame << std::endl;
#endif
    if (saveSteps) // Save results
        saveStepsResult(cur_copy2, frame);

    if (showSteps)
    {
        cv::resize(cur_copy2, cur_copy2, cv::Size(current.cols * 3, current.rows * 3));
        cv::imshow("Steps Smooth", cur_copy2);
        // cv::waitKey(0);
    }
}

void FeetTracker::setFeetPositionsByBBox(int frame, cv::Rect &pos, cv::Mat &pmask)
{
    float feet_proportion = 0.125, max_search = 0.3; // According to typical human proportions
    uint h, href;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Rect> nbboxes;
    std::vector<cv::Rect> candidates;

    // Get initial bounding boxes
    h = pos.height * feet_proportion;
    if (h == 0)
        h = 1;
    // 1. Get initial bounding box and possible initial boxes
    cv::Rect initR(pos.x, pos.y + pos.height - h, pos.width, h);
    cv::Mat initSeg(pmask, initR), ilabels, istats, icentroids;
    cv::connectedComponentsWithStats(initSeg, ilabels, istats, icentroids);
    std::vector<cv::Point2i> pointInRegion;
    getBBoxes(ilabels, istats, bboxes, initR.x, initR.y, pointInRegion);
    uint n = bboxes.size(), ref_index;

    // 2. Set ref bounding box
    cv::Rect b;
    if (n == 0)
    {
        std::cout << "Error: No player feet detected." << std::endl;
        return;
    }
    else if (n == 1)
    {
        b = bboxes[0];
        ref_index = 0;
    }
    else
    {
        b = bboxes[0];
        uint y, max_y = b.y + b.height - 1;
        ref_index = 0;
        for (uint i = 1; i < n; ++i)
        {
            cv::Rect &r = bboxes[i];
            y = r.y + r.height - 1;
            if (y > max_y)
            {
                ref_index = i;
                max_y = y;
            }
        }
        if (ref_index != 0)
            b = bboxes[ref_index];
    }

    // Add FIRST FOOT:
    candidates.push_back(b);

    // 3. SECOND FOOT:
    std::vector<cv::Rect> second_candidates;
    std::vector<int> connected_at;

    uint x, y, ym, ylim, xi, xf, ref_x, ref_y;
    href = max_search * pos.height;
    ylim = pos.height - href + pos.y;
    ym = b.y;
    uint bstep = 2;
    h = b.height;
    cv::Rect cur_r;

    // 3.1 Check if in some point we found an initial second feet
    cv::Mat labels, stats, centroids;
    for (y = ym; y >= ylim; y -= bstep, h += bstep)
    {
        cur_r = cv::Rect(pos.x, y, pos.width, h);
        cv::Mat initSeg(pmask, cur_r);
        cv::connectedComponentsWithStats(initSeg, labels, stats, centroids);
        if (stats.rows > 2)
        { // If there are more than one box
            n = stats.rows - 1;
            break;
        }
    }

    if (n > 1)
    { // 3.2 If found more than one in some point: complete it (or them) to ref box height
        // 3.2.1 Find label of original box
        uint olabel;
        ref_x = pointInRegion[ref_index].x - pos.x;
        ref_y = b.y - cur_r.y;
        olabel = labels.at<int>(ref_y, ref_x);
        // std::cout << "olabel: " << olabel << std::endl;

        // 3.2.2 Set initial extra bboxes
        std::vector<cv::Point2i> sample_points;
        getSamplesAndBBoxesExceptLabel(labels, stats, sample_points, nbboxes, pos.x, y, olabel);
        std::vector<bool> incomplete(nbboxes.size(), true);
        std::vector<bool> merged(nbboxes.size(), false);
        int j;

        // 3.2.3 Iterate to build incomplete to ref box height
        uint i, w, ssize, label, hh;

        if (frame == 7)
            std::cout << "Stop." << std::endl;

        for (y = y - 1, h += 1; y >= ylim; y -= 1, h += 1)
        {
            // 3.2.3.1 Adjust connected components
            cv::Rect cur_rr(pos.x, y, pos.width, h);
            cv::Mat initSeg(pmask, cur_rr);
            cv::connectedComponentsWithStats(initSeg, labels, stats, centroids);

            // 3.2.3.2 Update current box positions
            ++ref_y;
            ssize = sample_points.size();
            for (i = 0; i < ssize; ++i)
                ++sample_points[i].y;

            if (stats.rows > 2)
            { // 3.2.3.2.1 If there are more than one box (still some disconnected)
                std::vector<bool> referenced(stats.rows - 1, false);
                std::vector<uint> blabels(ssize);

                olabel = labels.at<int>(ref_y, ref_x); // Reference box always on first line

                // 3.2.3.2.1.1 Get labels for other valid boxes and set if component referenced:
                for (i = 0; i < ssize; ++i)
                    if (!merged[i])
                    {
                        label = labels.at<int>(sample_points[i]);
                        blabels[i] = label;
                        referenced[label - 1] = true;
                    }

                // 3.2.3.2.1.2 Set merged flag for those newly connected
                if (ssize > 1) // No sense if only ref box and another one
                    for (i = 0; i < ssize - 1; ++i)
                        if (!merged[i])
                        {
                            label = blabels[i];
                            if (label != olabel)
                            { // Not ref box
                                for (uint j = 1; j < ssize; ++j)
                                {
                                    if (i != j && !merged[j] && label == blabels[j]) // Merged!!
                                        merged[j] = true;
                                }
                            }
                        }

                // 3.2.3.2.1.3 Update not merged bboxes and create new ones

                // Update
                bool ready = false;
                for (i = 0; i < ssize; ++i)
                    if (!merged[i])
                    {
                        label = blabels[i];
                        hh = stats.at<int>(label, 3);
                        if (label == olabel)
                        { // Merging with ref bounding box region: We are ready: Put as second feet candidate
                            // second_candidates
                            if (incomplete[i])
                                second_candidates.push_back(cv::Rect(nbboxes[i].x,
                                                                     stats.at<int>(label, 1) + y,
                                                                     nbboxes[i].width, nbboxes[i].height + 1));
                            else
                                second_candidates.push_back(cv::Rect(nbboxes[i]));
                            connected_at.push_back(y);
                            merged[i] = true;
                            incomplete[i] = false;
                        }
                        else
                        { // Not merging with ref: Update bounding box
                            if (incomplete[i])
                            {
                                nbboxes[i] = cv::Rect(stats.at<int>(label, 0) + pos.x, stats.at<int>(label, 1) + y,
                                                      stats.at<int>(label, 2), hh);
                                if (hh >= b.height)
                                { // Completed size of bbox... Ready to be the candidate, as it is connected
                                    incomplete[i] = false;
                                    second_candidates.push_back(cv::Rect(nbboxes[i]));
                                    connected_at.push_back(y);
                                    ready = true;
                                    break;
                                }
                            }
                        }
                    }
                if (ready)
                    break;
                // Check new boxes
                for (j = 1; j < stats.rows; ++j)
                    if (!referenced[j - 1] && j != olabel)
                    { // New box
                        xi = stats.at<int>(j, 0);
                        w = stats.at<int>(j, 2);
                        xf = xi + w - 1;
                        nbboxes.push_back(cv::Rect(xi + pos.x, y, w, 1));
                        incomplete.push_back(true);
                        merged.push_back(false);
                        for (x = xi; x <= xf; ++x)
                        {
                            if (labels.at<int>(0, x) == j)
                            {
                                sample_points.push_back(cv::Point2i(x, 0));
                                break;
                            }
                        }
                    }
            }
            else
            { // Now just found one connected region: All active are then complete
                ssize = nbboxes.size();
                for (i = 0; i < ssize; ++i)
                {
                    if (!merged[i])
                    {
                        if (!incomplete[i])
                            second_candidates.push_back(cv::Rect(nbboxes[i].x,
                                                                 stats.at<int>(1, 1) + y,
                                                                 nbboxes[i].width, nbboxes[i].height + 1));
                        else
                        {
                            second_candidates.push_back(cv::Rect(nbboxes[i]));
                            connected_at.push_back(y);
                            merged[i] = true;
                            incomplete[i] = false;
                        }
                    }
                }
            }
        } // End of second feet completion cycle

        ssize = second_candidates.size();

        /* WHY COMMENTED? --> If not connected, cannot be a candidate.
                if(ssize == 0) { //No complete candidates found
                    if(nbboxes.size() > 0) {
                        if(nbboxes.size() == 1) //Only one
                            candidates.push_back(nbboxes[0]);
                        else { //Select the largest one
                            int imax, max_h = 0;
                            for(i=0; i<nbboxes.size(); ++i) {
                                cv::Rect &nbb = nbboxes[i];
                                if(nbb.height > max_h) {
                                    max_h = nbb.height;
                                    imax = i;
                                }
                            }
                            candidates.push_back(nbboxes[imax]);
                        }
                    }
                } else { //Select the best candidate (complete feet size (ref) connected to the ref regions - prevent to considered first those merged too soon):
        */

        if (ssize > 0)
        {                                                                  // Select the best candidate (complete feet size (ref) connected to the ref regions - prevent to considered first those merged too soon):
            if (ssize == 1 && second_candidates[0].height >= b.height / 2) // Only one?
                candidates.push_back(second_candidates[0]);
            else
            { // Several candidates:
                // First check among complete, the lowest connection
                bool incomplete = true; // Unchecks if found
                int conn, max_conn = 0, conn_candidate;

                for (i = 0; i < ssize; ++i)
                    if (second_candidates[i].height >= b.height)
                    {
                        incomplete = false; // One is enough
                        conn = connected_at[i];
                        if (conn > max_conn)
                        {
                            max_conn = conn;
                            conn_candidate = i;
                        }
                    }
                if (incomplete)
                { // If no other complete candidates, take the highest one
                    int imin = -1, diff, min_diff = INT_MAX, bh = b.height, hhalf = bh / 2;
                    for (i = 0; i < ssize; ++i)
                    {
                        diff = abs(bh - second_candidates[i].height);
                        if (diff > hhalf)
                            continue;
                        if (diff < min_diff)
                        {
                            min_diff = diff;
                            imin = i;
                        }
                    }
                    if (imin >= 0) // Prevents selecting second foot for erroneous boxes (prefers one candidate).
                        candidates.push_back(second_candidates[imin]);
                }
                else
                {
                    candidates.push_back(second_candidates[conn_candidate]);
                }
            }
        }

    } // End second feet candidate incorporation trials

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::Mat ft_color;
    cv::cvtColor(pmask, ft_color, cv::COLOR_GRAY2BGR);
    paintInitialFeetPositionBBoxes(candidates, ft_color, 0, 0, 255);
    cv::resize(ft_color, ft_color, cv::Size(ft_color.cols, ft_color.rows));
    std::cout << "Candidates: " << candidates.size() << std::endl;
    cv::imshow("Candidates", ft_color);
#endif
    candidates_per_frame[frame] = candidates;
}

void FeetTracker::paintInitialFeetPositionBBoxes(std::vector<cv::Rect> &bboxes, cv::Mat &img, uchar B, uchar G, uchar R)
{
    uint bsize = bboxes.size();
    for (uint i = 0; i < bsize; ++i)
        cv::rectangle(img, bboxes[i], cv::Scalar(B, G, R));
}

void FeetTracker::trackPositions(int frame, cv::Rect &pos, cv::Mat &pmask, cv::Mat &current, int msecs, int frame_index)
{
    /*if (frame == 6)
        std::cout << "Stop." << std::endl;*/
    // Control the frames queues to max frames_to_store frames
    if (stored_oframes.size() == frames_to_store)
    {
        stored_oframes.pop_front();
        stored_mframes.pop_front();
        stored_times.pop_front();
    }

    cv::Mat ccopy, mcopy;
    current.copyTo(ccopy);
    pmask.copyTo(mcopy);
    stored_oframes.push_back(ccopy);
    stored_mframes.push_back(mcopy);
    stored_times.push_back(msecs);
    sframes[frame] = ccopy;
    smasks[frame] = mcopy;
    spos[frame] = pos;
    ++frame_count;

    if (frame_count <= frames_to_store / 2) // Do nothing if no enough frames to process are available
        return;
    int fs_2 = frames_to_store / 2,
        cindex = frame_count <= frames_to_store ? frame_count - fs_2 - 1 : fs_2,
        cframe = frame - fs_2, kframe,
        fframe = cframe + fs_2;
    cv::Rect left, right;
    std::vector<cv::Rect> &cbboxes = candidates_per_frame[cframe];
    std::vector<cv::Rect> left_future_frames, right_future_frames;
    std::vector<int> future_msec;
    int n = cbboxes.size();
    num_candidates.push_back(n);
    cv::Mat &mask = stored_mframes[cindex];

    if (cindex == 0)
    { // 1. The first frame to process
        if (n == 1)
        { // 1.1Current has just one candidate
            // We can suppose feet sizes are malformed by occlusion
            // 1.1.1. Associate left and right on window for future frames
            //    and check if there are future frames with two or more candidates:
            bool search_start = true;
            int start_frame, num_multicandidate = 0;
            std::vector<cv::Rect> first_bboxes;
            for (kframe = cframe + 1; kframe <= fframe; ++kframe)
            {
                std::vector<cv::Rect> &fbboxes = candidates_per_frame[kframe];
                if (fbboxes.size() >= 2)
                { // We have (at least) two candidates
                    ++num_multicandidate;
                    if (search_start)
                    {
                        search_start = false;
                        first_bboxes = fbboxes;
                        associateLeftAndRight(fbboxes, left, right);
                        start_frame = kframe;
                    }
                }
            }

            // 1.1.2. Use or reestimate position according to each case.
            if (num_multicandidate == 0)
            {
                // 1.1.2.1 If no two-rect candidate, just duplicate position of the sole candidate,
                //     and estimate velocities according to sole positions
                cv::Rect r_cur = cbboxes[0];
                cv::Rect fother = r_cur;
                --fother.y; // Displace second a little bit
                cv::Point2i ccenter = getRectCenter(r_cur),
                            ocenter = ccenter, prev, cur;
                --ocenter.y; // Displace second a little bit
                // Init previous with current position:
                prev = ccenter;
                float dx = 0, dy = 0, wd = 0, w = 1.0;
                // Get weighted velocity (same for both feet), according to distance to previous frame.
                // Weight divides by two for each frame (farthest contributes less).
                // We are assuming that future velocity is similar for the first frame.
                for (kframe = cframe + 1; kframe <= fframe; ++kframe, w /= 2.0)
                {
                    std::vector<cv::Rect> &fbboxes = candidates_per_frame[kframe];
                    if (fbboxes.size() == 1)
                    {
                        cur = getRectCenter(fbboxes[0]);
                        dx += w * (cur.x - prev.x);
                        dy += w * (cur.y - prev.y);
                        wd += w;
                        prev = cur;
                    }
                }
                if (wd == 0)
                {
                    dx = dy = 0;
                }
                else
                {
                    dx /= wd;
                    dy /= wd;
                }

                left_rects.push_back(r_cur);
                Dx_left.push_back(dx);
                Dy_left.push_back(dy);
                right_rects.push_back(fother);
                Dx_right.push_back(dx);
                Dy_right.push_back(dy);
                from_one_candidate.push_back(true);
            }
            else if (num_multicandidate == 1)
            {
                // 1.1.2.2 If one two-rect candidate, check if there are more unique in between and set
                //     estimated velocities. Consider the nearest to unique candidate and adjust position of
                //     second according to the nearest highest region match (displace one in 1_y position).
                int diff = start_frame - cframe;
                cv::Rect r_cur = cbboxes[0];
                cv::Rect fother = r_cur;
                --fother.y;
                cv::Point2i ccenter = getRectCenter(r_cur),
                            lcenter = getRectCenter(left),
                            rcenter = getRectCenter(right),
                            ocenter = ccenter;
                --ocenter.y;
                // Set nearest feet as grounded one and second slightly displaced up.
                // Suppose that initial velocities from second frame maintain in first.
                // Consider frame distance to adjust to real velocity
                if (firstNearest(ccenter, lcenter, rcenter))
                {
                    left_rects.push_back(r_cur);
                    Dx_left.push_back((lcenter.x - ccenter.x) / diff);
                    Dy_left.push_back((lcenter.y - ccenter.y) / diff);
                    right_rects.push_back(fother);
                    Dx_right.push_back((rcenter.x - ocenter.x) / diff);
                    Dy_right.push_back((rcenter.y - ocenter.y) / diff);
                }
                else
                {
                    left_rects.push_back(fother);
                    Dx_left.push_back((lcenter.x - ocenter.x) / diff);
                    Dy_left.push_back((lcenter.y - ocenter.y) / diff);
                    right_rects.push_back(r_cur);
                    Dx_right.push_back(rcenter.x - ccenter.x);
                    Dy_right.push_back(rcenter.y - ccenter.y);
                }
                from_one_candidate.push_back(true);
            }
            else
            {
                // 1.1.2.3 If more than one two-rect candidate

                // 1.1.2.3.1 Get mean future left and right feet, and their velocities
                cv::Rect r_cur = cbboxes[0], cleft, cright;
                // Use two nearest multicandidate frames. Estimate size and velocities from them.
                for (kframe = start_frame + 1; kframe <= fframe; ++kframe)
                {
                    std::vector<cv::Rect> &fbboxes = candidates_per_frame[kframe];
                    if (fbboxes.size() >= 2)
                    { // We have (at least) two candidates
                        associateLeftAndRight(first_bboxes, start_frame, fbboxes, kframe,
                                              r_cur, cframe, cleft, cright, stored_mframes.front());
                        break;
                    }
                }
            }
        }
        else if (n >= 2)
        { // 1.2 If current has more than one candidate
            // 1.2.1. Associate left and right on window for future frames
            //    and check if there are future frames with two or more candidates:
            bool search_start = true;
            int start_frame, num_multicandidate = 0;
            std::vector<cv::Rect> first_bboxes;
            for (kframe = cframe + 1; kframe <= fframe; ++kframe)
            {
                std::vector<cv::Rect> &fbboxes = candidates_per_frame[kframe];
                if (fbboxes.size() >= 2)
                { // We have (at least) two candidates
                    ++num_multicandidate;
                    if (search_start)
                    {
                        search_start = false;
                        first_bboxes = fbboxes;
                        associateLeftAndRight(fbboxes, left, right);
                        start_frame = kframe;
                    }
                }
            }

            // 1.2.2. Use or reestimate position according to each case.
            if (num_multicandidate == 0)
            {
                // 1.2.2.1 If no two-rect candidate, use nearest frame bbox to find the two nearest current candidates,
                //     and estimate velocities according to this position
                cv::Rect cleft, cright;
                associateLeftAndRight(cbboxes, cleft, cright, candidates_per_frame[cframe + 1][0]);
            }
            else if (num_multicandidate == 1)
            {
                // 1.2.2.2 If one two-rect candidate, two cases:
                cv::Rect cleft, cright;
                int diff = start_frame - cframe;

                if (diff == 1)
                {
                    // 1.2.2.2.1 If two(or more)-rect candidates frame is next one, associate one with the best combination
                    //           to the next one-candidate frame, and the second as the lowest paired with the nearest remaining.
                    associateLeftAndRight(cbboxes, first_bboxes, candidates_per_frame[start_frame + 1][0], cleft, cright);
                }
                else
                {
                    // 1.2.2.2.2 If two(or more)-rect candidates frame is NOT next one, associate one with the best combination
                    //           to the next one-candidate frame. The second current will be the lowest, and will fit to
                    //           best mask position, in the line of the nearest remaining in the two(or more)-rect candidates frame.
                    associateLeftAndRight(cbboxes, first_bboxes, candidates_per_frame[cframe + 1][0], diff, cleft, cright);
                }
            }
            else
            {
                // 1.2.2.3 If more than one two-rect candidate

                // 1.2.2.3.1 Get second two(more)-candidate frame. Assume lowest candidate is valid, and search best velocity match.
                //           For second, chose current with minimal projection error among best combinations of the next two frames.
                cv::Rect r_cur = cbboxes[0], cleft, cright;
                // Use two nearest multicandidate frames. Estimate size and velocities from them.
                for (kframe = start_frame + 1; kframe <= fframe; ++kframe)
                {
                    std::vector<cv::Rect> &fbboxes = candidates_per_frame[kframe];
                    if (fbboxes.size() >= 2)
                    { // We have (at least) two candidates
                        //                        associateLeftAndRight(cbboxes, first_bboxes, start_frame - cframe, fbboxes, kframe - cframe,
                        //                                              cleft, cright);
                        associateLeftAndRight(cbboxes, first_bboxes, start_frame, fbboxes, kframe,
                                              cleft, cright);
                        break;
                    }
                }
            }
        }
    }
    else
    { // 2. Previous frame history available
        float dx_left, dy_left, dx_right, dy_right;
        if (left_rects.size() == 0)
        {
            std::cerr << "Error: Feet array empty." << std::endl;
            return;
        }
        cv::Rect pleft = left_rects.back(), pright = right_rects.back();
        if (cindex == 1)
        { // 2.1 Just one from history: Use one from future to adjust
            dx_left = Dx_left[0];
            dy_left = Dy_left[0];
            dx_right = Dx_right[0];
            dy_right = Dy_right[0];
            pleft.x += dx_left;
            pleft.y += dy_left;
            pright.x += dx_right;
            pright.y += dy_right;

            if (n == 1)
            { // 2.1.1 Set candidate association first and adjust second to best mask position
                bool is_left = true;
                cv::Rect ccur = cbboxes[0];
                cv::Point2i cur = getRectCenter(ccur), pl = getRectCenter(pleft), pr = getRectCenter(pright), po;

                cv::Rect cother;
                if (abs(pr.x - cur.x) + abs(pr.y - cur.y) < abs(pl.x - cur.x) + abs(pl.y - cur.y))
                {
                    is_left = false;
                    fitRectToMask(pleft, dx_left, dy_left, cother, mask, ccur);
                }
                else
                    fitRectToMask(pright, dx_right, dy_right, cother, mask, ccur);
                cv::Point2i pother = getRectCenter(cother);
                if (is_left)
                {
                    left_rects.push_back(ccur);
                    Dx_left.push_back(cur.x - pl.x);
                    Dy_left.push_back(cur.y - pl.y);
                    right_rects.push_back(cother);
                    Dx_right.push_back(pother.x - pr.x);
                    Dy_right.push_back(pother.y - pr.y);
                }
                else
                {
                    left_rects.push_back(cother);
                    Dx_left.push_back(cother.x - pl.x);
                    Dy_left.push_back(cother.y - pl.y);
                    right_rects.push_back(ccur);
                    Dx_right.push_back(cur.x - pr.x);
                    Dy_right.push_back(cur.y - pr.y);
                }
                from_one_candidate.push_back(true);
            }
            else
            { // 2.1.2 Find best associations
                // Start with lowest (normally moves less) and then find nearest to second
                bool left_first = true;
                cv::Rect first, second, eval1, eval2;
                if (pleft.y + pleft.height > pright.y + pright.height)
                {
                    eval1 = pleft;
                    eval2 = pright;
                }
                else
                {
                    left_first = false;
                    eval1 = pright;
                    eval2 = pleft;
                }
                int i, csize = cbboxes.size(), first_idx;
                float dist, min_dist = FLT_MAX;
                cv::Point2i p1 = getRectCenter(eval1), p2 = getRectCenter(eval2), p, fp, sp;
                for (i = 0; i < csize; ++i)
                {
                    cv::Rect &r = cbboxes[i];
                    p = getRectCenter(r);
                    dist = abs(p.x - p1.x) + abs(p.y - p1.y);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        first = r;
                        fp = p;
                        first_idx = i;
                    }
                }
                min_dist = FLT_MAX;
                for (i = 0; i < csize; ++i)
                {
                    if (i == first_idx)
                        continue;
                    cv::Rect &r = cbboxes[i];
                    p = getRectCenter(r);
                    dist = abs(p.x - p2.x) + abs(p.y - p2.y);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        second = r;
                        sp = p;
                    }
                }
                if (left_first)
                {
                    left_rects.push_back(first);
                    Dx_left.push_back(fp.x - p1.x);
                    Dy_left.push_back(fp.y - p1.y);
                    right_rects.push_back(second);
                    Dx_right.push_back(sp.x - p2.x);
                    Dy_right.push_back(sp.y - p2.y);
                }
                else
                {
                    left_rects.push_back(second);
                    Dx_left.push_back(sp.x - p2.x);
                    Dy_left.push_back(sp.y - p2.y);
                    right_rects.push_back(first);
                    Dx_right.push_back(fp.x - p1.x);
                    Dy_right.push_back(fp.y - p1.y);
                }
                from_one_candidate.push_back(false);
            }
        }
        else
        { // 2.2 Longer history, allows to project potential position and estimate feet size
            // This part projects last left/right foot to potential current position, with velocity taken from two previous

            int rsize = left_rects.size();
            cv::Rect pleft2 = left_rects[rsize - 2], pright2 = right_rects[rsize - 2];
            float acceptable_dim_variation = 0.2;
            bool prev_from_one = from_one_candidate.back(),
                 prevprev_from_one = from_one_candidate[rsize - 2];
            int mw_l, mh_l, mw_r, mh_r;

            // Same nature of boxes considers mean dimensions, else those of the previous.
            if (prev_from_one == prevprev_from_one)
            {
                mw_l = round((pleft.width + pleft2.width) / 2.0);
                mh_l = round((pleft.height + pleft2.height) / 2.0);
                mw_r = round((pright.width + pright2.width) / 2.0);
                mh_r = round((pright.height + pright2.height) / 2.0);
            }
            else
            { //(!prev_from_one && prevprev_from_one)
                mw_l = pleft.width;
                mh_l = pleft.height;
                mw_r = pright.width;
                mh_r = pright.height;
            }

            // Project feet with two previous frame associations:
            cv::Point2i pl = getRectCenter(pleft),
                        pr = getRectCenter(pright),
                        pl2 = getRectCenter(pleft2),
                        pr2 = getRectCenter(pright2);

            if (prev_from_one == prevprev_from_one)
            { // Same nature, so comparable
                dx_left = pl.x - pl2.x;
                dy_left = pl.y - pl2.y;
                dx_right = pr.x - pr2.x;
                dy_right = pr.y - pr2.y;
            }
            else if (prev_from_one && !prevprev_from_one)
            { // Prev is one, so consider position of prev prev as merge
                cv::Rect paux_left, paux_right;
                int xl2 = pleft2.x + pleft2.width - 1, xr2 = pright2.x + pright2.width - 1, new_x2;
                new_x2 = xl2 > xr2 ? xl2 : xr2;
                paux_left.x = paux_right.x = pleft2.x < pright2.x ? pleft2.x : pright2.x;
                paux_left.width = paux_right.width = new_x2 - paux_left.x + 1;

                int yl2 = pleft2.y + pleft2.height - 1, yr2 = pright2.y + pright2.height - 1, new_y2;
                new_y2 = yl2 > yr2 ? yl2 : yr2;
                paux_left.y = paux_right.y = pleft2.y < pright2.y ? pleft2.y : pright2.y;
                paux_left.height = paux_right.height = new_y2 - paux_left.y + 1;
                paux_right.y--; // One up
                pl2 = getRectCenter(paux_left),
                pr2 = getRectCenter(paux_right);
                dx_left = pl.x - pl2.x;
                dy_left = pl.y - pl2.y;
                dx_right = pr.x - pr2.x;
                dy_right = pr.y - pr2.y;
            }
            else
            { //(!prev_from_one && prevprev_from_one)
                // We got to first separate candidates, proportional to each part
                cv::Rect paux_left, paux_right;
                // For x coordinate:

                // First we check and get real order for x:
                int prlx1, prrx1, prlx2, prrx2;
                if (pleft.x < pright.x)
                {
                    prlx1 = pleft.x;
                    prrx1 = pright.x;
                    prlx2 = prlx1 + pleft.width - 1;
                    prrx2 = prrx1 + pright.width - 1;
                }
                else
                {
                    prlx1 = pright.x;
                    prrx1 = pleft.x;
                    prlx2 = prlx1 + pright.width - 1;
                    prrx2 = prrx1 + pleft.width - 1;
                }
                // Then build the three cases
                paux_left.x = prlx1;
                if (prlx2 < prrx1)
                { // Case 1: Not intersecting in x
                    float full_width = prrx2 - prlx1 + 1, prop = pleft2.width / full_width;
                    int a = prlx2 - prlx1 + 1, b = prrx1 - prlx1, c = prrx2 - prrx1 + 1;
                    paux_left.width = prop * a;
                    paux_right.x = prlx1 + prop * b;
                    paux_right.width = prop * c;
                }
                else if (prlx2 > prrx2)
                { // Case 2: right contained in left
                    float full_width = prlx2 - prlx1 + 1, prop = pleft2.width / full_width;
                    int a = prrx1 - prlx1, b = prrx2 - prrx1 + 1;
                    paux_left.width = pleft2.width;
                    paux_right.x = prlx1 + prop * a;
                    paux_right.width = prop * b;
                }
                else
                { // Case 3: Partial intersection
                    float full_width = prrx2 - prlx1 + 1, prop = pleft2.width / full_width;
                    int a = prrx1 - prlx1, b = prlx2 - prlx1 + 1, c = prrx2 - prrx1 + 1;
                    paux_left.width = prop * b;
                    paux_right.x = prlx1 + prop * a;
                    paux_right.width = prop * c;
                }
                // For y coordinate:
                // First we check and get real order for x:
                int prly1, prry1, prly2, prry2;
                if (pleft.y < pright.y)
                {
                    prly1 = pleft.y;
                    prry1 = pright.y;
                    prly2 = prly1 + pleft.height - 1;
                    prry2 = prry1 + pright.height - 1;
                }
                else
                {
                    prly1 = pright.y;
                    prry1 = pleft.y;
                    prly2 = prly1 + pright.height - 1;
                    prry2 = prry1 + pleft.height - 1;
                }
                // Then build the three cases
                paux_left.y = prly1;
                if (prly2 < prry1)
                { // Case 1: Not intersecting in y
                    float full_height = prry2 - prly1 + 1, prop = pleft2.height / full_height;
                    int a = prly2 - prly1 + 1, b = prry1 - prly1, c = prry2 - prry1 + 1;
                    paux_left.height = prop * a;
                    paux_right.y = prly1 + prop * b;
                    paux_right.height = prop * c;
                }
                else if (prly2 > prry2)
                { // Case 2: right contained in left
                    float full_height = prly2 - prly1 + 1, prop = pleft2.height / full_height;
                    int a = prry1 - prly1, b = prry2 - prry1 + 1;
                    paux_left.height = pleft2.height;
                    paux_right.y = prly1 + prop * a;
                    paux_right.height = prop * b;
                }
                else
                { // Case 3: Partial intersection
                    float full_height = prry2 - prly1 + 1, prop = pleft2.height / full_height;
                    int a = prry1 - prly1, b = prly2 - prly1 + 1, c = prry2 - prry1 + 1;
                    paux_left.height = prop * b;
                    paux_right.y = prly1 + prop * a;
                    paux_right.height = prop * c;
                }

                pl2 = getRectCenter(paux_left),
                pr2 = getRectCenter(paux_right);
                dx_left = pl.x - pl2.x;
                dy_left = pl.y - pl2.y;
                dx_right = pr.x - pr2.x;
                dy_right = pr.y - pr2.y;
            }

            cv::Rect rleft = pleft, rright = pright;
            pleft.x += dx_left;
            pleft.y += dy_left;
            pright.x += dx_right;
            pright.y += dy_right;

            if (n == 1)
            { // 2.2.1 Set candidate association first and adjust second to best mask position
                cv::Rect &cur = cbboxes[0], first, second;
                cv::Point2i p = getRectCenter(cur), p2,
                            ppl = getRectCenter(pleft), ppr = getRectCenter(pright);
                // Set initial association to current sole candidate, and second to mask
                if (abs(p.x - ppl.x) + abs(p.y - ppl.y) < abs(p.x - ppr.x) + abs(p.y - ppr.y))
                {
                    if (abs(cur.width - mw_l) / (float)mw_l > acceptable_dim_variation || abs(cur.height - mh_l) / (float)mh_l > acceptable_dim_variation)
                    {
                        fitRectToMask((mw_l + cur.width) / 2, (mh_l + cur.height) / 2, cur, mask, first);
                        p = getRectCenter(first);
                    }
                    else
                    {
                        first = cur;
                    }
                    // fitRectToMask(rright, dx_right, dy_right, second, mask, cur);
                    fitRectToMask(mw_r, mh_r, cv::Point2i(pr.x + dx_right, pr.y + dy_right), second, mask, cur);
                    p2 = getRectCenter(second);

                    left_rects.push_back(first);
                    Dx_left.push_back(p.x - pl.x);
                    Dy_left.push_back(p.y - pl.y);
                    right_rects.push_back(second);
                    Dx_right.push_back(p2.x - pr.x);
                    Dy_right.push_back(p2.y - pr.y);
                    from_one_candidate.push_back(false);
                }
                else
                {
                    if (abs(cur.width - mw_r) / (float)mw_r > acceptable_dim_variation || abs(cur.height - mh_r) / (float)mh_r > acceptable_dim_variation)
                    {
                        fitRectToMask((mw_r + cur.width) / 2, (mh_r + cur.height) / 2, cur, mask, first);
                        p = getRectCenter(first);
                    }
                    else
                    {
                        first = cur;
                    }
                    // fitRectToMask(rleft, dx_left, dy_left, second, mask, cur);
                    fitRectToMask(mw_l, mh_l, cv::Point2i(pl.x + dx_left, pl.y + dy_left), second, mask, cur);
                    p2 = getRectCenter(second);

                    left_rects.push_back(second);
                    Dx_left.push_back(p2.x - pl.x);
                    Dy_left.push_back(p2.y - pl.y);
                    right_rects.push_back(first);
                    Dx_right.push_back(p.x - pr.x);
                    Dy_right.push_back(p.y - pr.y);
                    from_one_candidate.push_back(false);
                }
            }
            else
            { // 2.2.2 Find best associations
                // Start with lowest (normally moves less) and then find nearest to second
                bool left_first = true;
                cv::Rect first, second, eval1, eval2, reval1, reval2;
                if (pleft.y + pleft.height > pright.y + pright.height)
                {
                    eval1 = pleft;
                    eval2 = pright;
                    reval1 = rleft;
                    reval2 = rright;
                }
                else
                {
                    left_first = false;
                    eval1 = pright;
                    eval2 = pleft;
                    reval1 = rright;
                    reval2 = rleft;
                }
                int i, csize = cbboxes.size(), first_idx;
                float dist, min_dist = FLT_MAX;
                cv::Point2i p1 = getRectCenter(eval1), p2 = getRectCenter(eval2),
                            rp1 = getRectCenter(reval1), rp2 = getRectCenter(reval2), p, fp, sp;
                for (i = 0; i < csize; ++i)
                {
                    cv::Rect &r = cbboxes[i];
                    p = getRectCenter(r);
                    dist = abs(p.x - p1.x) + abs(p.y - p1.y);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        first = r;
                        fp = p;
                        first_idx = i;
                    }
                }
                min_dist = FLT_MAX;
                for (i = 0; i < csize; ++i)
                {
                    if (i == first_idx)
                        continue;
                    cv::Rect &r = cbboxes[i];
                    p = getRectCenter(r);
                    dist = abs(p.x - p2.x) + abs(p.y - p2.y);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        second = r;
                        sp = p;
                    }
                }

                // Improve fit for each box
                cv::Rect ffirst, fsecond;
                if (left_first)
                {
                    if (abs(first.width - mw_l) / (float)mw_l > acceptable_dim_variation || abs(first.height - mh_l) / (float)mh_l > acceptable_dim_variation)
                    {
                        fitRectToMask(pleft, mw_l, mh_l, first.x - rleft.x, first.y - rleft.y, mask, first, ffirst);
                        fp = getRectCenter(ffirst);
                    }
                    else
                    {
                        ffirst = first;
                    }
                    if (abs(second.width - mw_r) / (float)mw_r > acceptable_dim_variation || abs(second.height - mh_r) / (float)mh_r > acceptable_dim_variation)
                    {
                        fitRectToMask(pright, mw_r, mh_r, second.x - rright.x, second.y - rright.y, mask, second, fsecond);
                        sp = getRectCenter(fsecond);
                    }
                    else
                    {
                        fsecond = second;
                    }

                    left_rects.push_back(ffirst);
                    Dx_left.push_back(fp.x - rp1.x);
                    Dy_left.push_back(fp.y - rp1.y);
                    right_rects.push_back(fsecond);
                    Dx_right.push_back(sp.x - rp2.x);
                    Dy_right.push_back(sp.y - rp2.y);
                    from_one_candidate.push_back(false);
                }
                else
                {
                    if (abs(second.width - mw_l) / (float)mw_l > acceptable_dim_variation || abs(second.height - mh_l) / (float)mh_l > acceptable_dim_variation)
                    {
                        fitRectToMask(pleft, mw_l, mh_l, second.x - rleft.x, second.y - rleft.y, mask, second, fsecond);
                        sp = getRectCenter(fsecond);
                    }
                    else
                    {
                        fsecond = second;
                    }
                    if (abs(first.width - mw_r) / (float)mw_r > acceptable_dim_variation || abs(first.height - mh_r) / (float)mh_r > acceptable_dim_variation)
                    {
                        fitRectToMask(pright, mw_r, mh_r, first.x - rright.x, first.y - rright.y, mask, first, ffirst);
                        fp = getRectCenter(ffirst);
                    }
                    else
                    {
                        ffirst = first;
                    }

                    left_rects.push_back(fsecond);
                    Dx_left.push_back(sp.x - rp2.x);
                    Dy_left.push_back(sp.y - rp2.y);
                    right_rects.push_back(ffirst);
                    Dx_right.push_back(fp.x - rp1.x);
                    Dy_right.push_back(fp.y - rp1.y);
                    from_one_candidate.push_back(false);
                }
            }
        }
    }
#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "Processed tracking frame: " << cframe << std::endl;
#endif
    cv::Mat current_phantom, cur_copy;
    if (saveTracking || showTracking) // Save results
        current_phantom = getPhantomTrackingImage(cindex, 3);
    if (saveTracking) // Save results
        saveResult(current_phantom, cframe);
    if (showTracking)
    { // Save results
        current_phantom.copyTo(cur_copy);
        // cv::resize(cur_copy, cur_copy, cv::Size(current_phantom.cols*3, current_phantom.rows*3));
        cv::imshow("Tracking", cur_copy);
        //       cv::waitKey(0);
    }
}

void FeetTracker::completeTracking(int frame_to_start)
{
    int fs_2 = frames_to_store / 2,
        cframe = frame_to_start,
        sfsize = stored_mframes.size(),
        cindex = sfsize - fs_2;
    cv::Rect left, right;
    float dx_left, dy_left, dx_right, dy_right;
    float acceptable_dim_variation = 0.2;

    for (; cindex < sfsize; ++cindex, ++cframe)
    {
        std::vector<cv::Rect> &cbboxes = candidates_per_frame[cframe];
        int n = cbboxes.size();
        num_candidates.push_back(n);
        cv::Mat &mask = stored_mframes[cindex];
        cv::Rect pleft = left_rects.back(), pright = right_rects.back();
        int rsize = left_rects.size();
        cv::Rect pleft2 = left_rects[rsize - 2], pright2 = right_rects[rsize - 2];
        int mw_l = round((pleft.width + pleft2.width) / 2.0),
            mh_l = round((pleft.height + pleft2.height) / 2.0),
            mw_r = round((pright.width + pright2.width) / 2.0),
            mh_r = round((pright.height + pright2.height) / 2.0);
        // Project feet with two previous frame associations:
        cv::Point2i pl = getRectCenter(pleft),
                    pr = getRectCenter(pright),
                    pl2 = getRectCenter(pleft2),
                    pr2 = getRectCenter(pright2);
        dx_left = pl.x - pl2.x;
        dy_left = pl.y - pl2.y;
        dx_right = pr.x - pr2.x;
        dy_right = pr.y - pr2.y;
        cv::Rect rleft = pleft, rright = pright;
        pleft.x += dx_left;
        pleft.y += dy_left;
        pright.x += dx_right;
        pright.y += dy_right;

        if (n == 1)
        { // 2.2.1 Set candidate association first and adjust second to best mask position
            cv::Rect &cur = cbboxes[0], first, second;
            cv::Point2i p = getRectCenter(cur), p2,
                        ppl = getRectCenter(pleft), ppr = getRectCenter(pright);
            // Set initial association to current sole candidate, and second to mask
            if (abs(p.x - ppl.x) + abs(p.y - ppl.y) < abs(p.x - ppr.x) + abs(p.y - ppr.y))
            {
                if (abs(cur.width - mw_l) / (float)mw_l > acceptable_dim_variation || abs(cur.height - mh_l) / (float)mh_l > acceptable_dim_variation)
                {
                    fitRectToMask((mw_l + cur.width) / 2, (mh_l + cur.height) / 2, cur, mask, first);
                    p = getRectCenter(first);
                }
                else
                {
                    first = cur;
                }
                // fitRectToMask(rright, dx_right, dy_right, second, mask, cur);
                fitRectToMask(mw_r, mh_r, cv::Point2i(pr.x + dx_right, pr.y + dy_right), second, mask, cur);
                p2 = getRectCenter(second);

                left_rects.push_back(first);
                Dx_left.push_back(p.x - pl.x);
                Dy_left.push_back(p.y - pl.y);
                right_rects.push_back(second);
                Dx_right.push_back(p2.x - pr.x);
                Dy_right.push_back(p2.y - pr.y);
            }
            else
            {
                if (abs(cur.width - mw_r) / (float)mw_r > acceptable_dim_variation || abs(cur.height - mh_r) / (float)mh_r > acceptable_dim_variation)
                {
                    fitRectToMask((mw_r + cur.width) / 2, (mh_r + cur.height) / 2, cur, mask, first);
                    p = getRectCenter(first);
                }
                else
                {
                    first = cur;
                }
                // fitRectToMask(rleft, dx_left, dy_left, second, mask, cur);
                fitRectToMask(mw_l, mh_l, cv::Point2i(pl.x + dx_left, pl.y + dy_left), second, mask, cur);
                p2 = getRectCenter(second);

                left_rects.push_back(second);
                Dx_left.push_back(p2.x - pl.x);
                Dy_left.push_back(p2.y - pl.y);
                right_rects.push_back(first);
                Dx_right.push_back(p.x - pr.x);
                Dy_right.push_back(p.y - pr.y);
            }
        }
        else
        { // 2.2.2 Find best associations
            // Start with lowest (normally moves less) and then find nearest to second
            bool left_first = true;
            cv::Rect first, second, eval1, eval2, reval1, reval2;
            if (pleft.y + pleft.height > pright.y + pright.height)
            {
                eval1 = pleft;
                eval2 = pright;
                reval1 = rleft;
                reval2 = rright;
            }
            else
            {
                left_first = false;
                eval1 = pright;
                eval2 = pleft;
                reval1 = rright;
                reval2 = rleft;
            }
            int i, csize = cbboxes.size(), first_idx;
            float dist, min_dist = FLT_MAX;
            cv::Point2i p1 = getRectCenter(eval1), p2 = getRectCenter(eval2),
                        rp1 = getRectCenter(reval1), rp2 = getRectCenter(reval2), p, fp, sp;
            for (i = 0; i < csize; ++i)
            {
                cv::Rect &r = cbboxes[i];
                p = getRectCenter(r);
                dist = abs(p.x - p1.x) + abs(p.y - p1.y);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    first = r;
                    fp = p;
                    first_idx = i;
                }
            }
            min_dist = FLT_MAX;
            for (i = 0; i < csize; ++i)
            {
                if (i == first_idx)
                    continue;
                cv::Rect &r = cbboxes[i];
                p = getRectCenter(r);
                dist = abs(p.x - p2.x) + abs(p.y - p2.y);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    second = r;
                    sp = p;
                }
            }

            // Improve fit for each box
            cv::Rect ffirst, fsecond;
            if (left_first)
            {
                if (abs(first.width - mw_l) / (float)mw_l > acceptable_dim_variation || abs(first.height - mh_l) / (float)mh_l > acceptable_dim_variation)
                {
                    fitRectToMask(pleft, mw_l, mh_l, first.x - rleft.x, first.y - rleft.y, mask, first, ffirst);
                    fp = getRectCenter(ffirst);
                }
                else
                {
                    ffirst = first;
                }
                if (abs(second.width - mw_r) / (float)mw_r > acceptable_dim_variation || abs(second.height - mh_r) / (float)mh_r > acceptable_dim_variation)
                {
                    fitRectToMask(pright, mw_r, mh_r, second.x - rright.x, second.y - rright.y, mask, second, fsecond);
                    sp = getRectCenter(fsecond);
                }
                else
                {
                    fsecond = second;
                }

                left_rects.push_back(ffirst);
                Dx_left.push_back(fp.x - rp1.x);
                Dy_left.push_back(fp.y - rp1.y);
                right_rects.push_back(fsecond);
                Dx_right.push_back(sp.x - rp2.x);
                Dy_right.push_back(sp.y - rp2.y);
            }
            else
            {
                if (abs(second.width - mw_l) / (float)mw_l > acceptable_dim_variation || abs(second.height - mh_l) / (float)mh_l > acceptable_dim_variation)
                {
                    fitRectToMask(pleft, mw_l, mh_l, second.x - rleft.x, second.y - rleft.y, mask, second, fsecond);
                    sp = getRectCenter(fsecond);
                }
                else
                {
                    fsecond = second;
                }
                if (abs(first.width - mw_r) / (float)mw_r > acceptable_dim_variation || abs(first.height - mh_r) / (float)mh_r > acceptable_dim_variation)
                {
                    fitRectToMask(pright, mw_r, mh_r, first.x - rright.x, first.y - rright.y, mask, first, ffirst);
                    fp = getRectCenter(ffirst);
                }
                else
                {
                    ffirst = first;
                }

                left_rects.push_back(fsecond);
                Dx_left.push_back(sp.x - rp2.x);
                Dy_left.push_back(sp.y - rp2.y);
                right_rects.push_back(ffirst);
                Dx_right.push_back(fp.x - rp1.x);
                Dy_right.push_back(fp.y - rp1.y);
            }
        }

#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Processed tracking frame: " << cframe << std::endl;
#endif
        cv::Mat current_phantom, cur_copy;
        if (saveTracking || showTracking) // Save results
            current_phantom = getPhantomTrackingImage(cindex, 3);
        if (saveTracking) // Save results
            saveResult(current_phantom, cframe);
        if (showTracking)
        { // Save results
            current_phantom.copyTo(cur_copy);
            //            cv::resize(cur_copy, cur_copy, cv::Size(current_phantom.cols*3, current_phantom.rows*3));
            cv::imshow("Tracking", cur_copy);
            //           cv::waitKey(0);
        }
    }
}

void FeetTracker::smoothDisplacement(int index)
{
    float sum_dx_left = weights[0] * Dx_left[index], sum_dy_left = weights[0] * Dy_left[index],
          sum_dx_right = weights[0] * Dx_right[index], sum_dy_right = weights[0] * Dy_right[index];
    int s = Dx_left.size(), norm = weights[0];
    for (int i = index + 1, j = 1; i < s && j <= 2; ++i, ++j)
    {
        sum_dx_left += weights[j] * Dx_left[i];
        sum_dy_left += weights[j] * Dy_left[i];
        sum_dx_right += weights[j] * Dx_right[i];
        sum_dy_right += weights[j] * Dy_right[i];
        norm += weights[j];
    }
    for (int i = index - 1, j = 1; i >= 0 && j <= 2; --i, ++j)
    {
        sum_dx_left += weights[j] * Dx_left[i];
        sum_dy_left += weights[j] * Dy_left[i];
        sum_dx_right += weights[j] * Dx_right[i];
        sum_dy_right += weights[j] * Dy_right[i];
        norm += weights[j];
    }
    Dx_left_s[index] = sum_dx_left / norm;
    Dy_left_s[index] = sum_dy_left / norm;
    Dx_right_s[index] = sum_dx_right / norm;
    Dy_right_s[index] = sum_dy_right / norm;
}

void FeetTracker::smoothBBoxes(int index)
{
    cv::Rect &left = left_rects[index], &right = right_rects[index];
    cv::Point2i pl = getRectCenter(left), pr = getRectCenter(right);
    float sum_x_left = weights[0] * pl.x, sum_y_left = weights[0] * pl.y,
          sum_x_right = weights[0] * pr.x, sum_y_right = weights[0] * pr.y,
          sum_w_left = weights[0] * left.width, sum_h_left = weights[0] * left.height,
          sum_w_right = weights[0] * right.width, sum_h_right = weights[0] * right.height;
    int s = left_rects.size(), norm = weights[0];
    for (int i = index + 1, j = 1; i < s && j <= 2; ++i, ++j)
    {
        cv::Rect &left = left_rects[i], &right = right_rects[i];
        cv::Point2i pl = getRectCenter(left), pr = getRectCenter(right);
        sum_x_left += weights[j] * pl.x;
        sum_y_left += weights[j] * pl.y;
        sum_x_right += weights[j] * pr.x;
        sum_y_right += weights[j] * pr.y;
        sum_w_left += weights[j] * left.width;
        sum_h_left += weights[j] * left.height;
        sum_w_right += weights[j] * right.width;
        sum_h_right += weights[j] * right.height;
        norm += weights[j];
    }
    for (int i = index - 1, j = 1; i >= 0 && j <= 2; --i, ++j)
    {
        cv::Rect &left = left_rects[i], &right = right_rects[i];
        cv::Point2i pl = getRectCenter(left), pr = getRectCenter(right);
        sum_x_left += weights[j] * pl.x;
        sum_y_left += weights[j] * pl.y;
        sum_x_right += weights[j] * pr.x;
        sum_y_right += weights[j] * pr.y;
        sum_w_left += weights[j] * left.width;
        sum_h_left += weights[j] * left.height;
        sum_w_right += weights[j] * right.width;
        sum_h_right += weights[j] * right.height;
        norm += weights[j];
    }
    sum_x_left /= norm;
    sum_y_left /= norm;
    sum_x_right /= norm;
    sum_y_right /= norm;
    sum_w_left /= norm;
    sum_h_left /= norm;
    sum_w_right /= norm;
    sum_h_right /= norm;

    left_rects_s[index] = cv::Rect(sum_x_left - sum_w_left / 2, sum_y_left - sum_h_left / 2, sum_w_left, sum_h_left);
    right_rects_s[index] = cv::Rect(sum_x_right - sum_w_right / 2, sum_y_right - sum_h_right / 2, sum_w_right, sum_h_right);
}

bool FeetTracker::leftStepCriteria(int index)
{
    if (abs(Dx_left_s[index]) + abs(Dy_left_s[index]) < min_displacement)
        return true;
    return false;
}

bool FeetTracker::rightStepCriteria(int index)
{
    if (abs(Dx_right_s[index]) + abs(Dy_right_s[index]) < min_displacement)
        return true;
    return false;
}

float FeetTracker::distance(cv::Point2f &p1, cv::Point2f &p2)
{
    float dx = p1.x - p2.x, dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

cv::Point2f FeetTracker::transformInv(cv::Point2f p)
{
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0, 0) = (p.x * calib_w) / real_w;
    pin.at<double>(1, 0) = (p.y * calib_h) / real_h;
    pin.at<double>(2, 0) = 1;
    cv::Mat pout = pin;
    return cv::Point2f(pout.at<double>(0, 0) / pout.at<double>(2, 0),
                       pout.at<double>(1, 0) / pout.at<double>(2, 0));
}

cv::Point2f FeetTracker::getStepPosition(int frame, cv::Rect &feet)
{
    cv::Point2f pixel_pos;
    cv::Mat &mask = smasks[frame];
    cv::imshow("Mask in Pos", mask);

    if (feet.x < 0)
    {
        feet.width += feet.x; // substract correction
        feet.x = 0;
    }
    if (feet.x + feet.width >= mask.cols)
        feet.width = mask.cols - feet.x;

    if (feet.y < 0)
    {
        feet.height += feet.y; // substract correction
        feet.y = 0;
    }
    if (feet.y + feet.height >= mask.rows)
        feet.height = mask.rows - feet.y;

    cv::Mat roi(mask, feet), big_roi, br_color;

    int i, j, w = feet.width, w2 = rint(w / 2.0), h = feet.height, num_lines = 3, lim = h - num_lines, count = 0;
    cv::Point2f ref(feet.x + w2, feet.y + h - 1), comp = ref, ref_cm = transformInv(ref), comp_cm;
    for (i = ref.y - 1, j = 1; i >= feet.y; --i, ++j)
    {
        comp.y = i;
        comp_cm = transformInv(comp);
        if (distance(ref_cm, comp_cm) < cm_to_feet_contact)
            continue;
        num_lines = j - 1;
    }
    if (num_lines < 2) // Min considered lines
        num_lines = 2;

    float mx = 0, my = 0;
    for (i = h - 1; i >= lim; --i)
        for (j = 0; j < w; ++j)
            if (roi.at<uchar>(i, j) == 255)
            {
                ++count;
                mx += j;
                my += i;
            }
    if (count > 0)
    {
        mx /= count;
        my /= count;
    }
    else
    {
        mx = w / 2.0;
        my = (h - 1 + lim) / 2.0;
    }
    pixel_pos.x = mx + feet.x;
    pixel_pos.y = my + feet.y;

    /*    cv::cvtColor(roi, br_color, cv::COLOR_GRAY2BGR);
        cv::circle(br_color, cv::Point(rint(mx), rint(my)), 1, cv::Scalar(0,0,255));
        cv::resize(br_color, big_roi, cv::Size(roi.cols*4, roi.rows*4));
        cv::imshow("Feet ROI", big_roi);
        std::cout << "Count: " << count << std::endl;
        std::cout << "(mx, my): " << mx << ", " << my << std::endl;
        cv::waitKey(0); */

    return pixel_pos;
}

// void FeetTracker::intersectsObjective(int index, int frame, cv::Rect &left, bool lstep, cv::Rect &right, bool rstep) {
//     if(!lstep && !rstep) {
//         return;
//     }

//     if(lstep) {
//         //First intersections
//         std::vector<cv::Point2i> leftCont;
//         leftCont.push_back(cv::Point2i(left.x, left.y));
//         leftCont.push_back(cv::Point2i(left.x + left.width - 1, left.y));
//         leftCont.push_back(cv::Point2i(left.x + left.width - 1, left.y + left.height - 1));
//         leftCont.push_back(cv::Point2i(left.x, left.y + left.height - 1));
//         bool intersection_ok = false;
//         for(uint i=1; i<=9; ++i) {
//             std::vector<cv::Point2i> imPos;
//             imPos.push_back(objImPos[i][0]);
//             imPos.push_back(objImPos[i][1]);
//             imPos.push_back(objImPos[i][2]);
//             imPos.push_back(objImPos[i][3]);
//             std::vector<cv::Point2i> inter = intersectConvexPolygons(imPos, leftCont);
//             if(inter.size() > 0) {
//                 double area = cv::contourArea(inter), obj_area = cv::contourArea(imPos);
//                 if(area > obj_area*objective_coverage_rate) {
//                     in_objective1[index] = i;
//                     this->odist1[index] = 0;
//                     left_foot[index] = getStepPosition(frame, left);
//                     intersection_ok = true;
//                 }
//                 break; //If not enough still stop (a step is not that big)
//             }
//         }

//         if(!intersection_ok) {
//             float d, min_dist = FLT_MAX;
//             int min_ind = 0;
//             cv::Point2f lpos = getStepPosition(frame, left), lpos_cm = transformInv(lpos);
//             left_foot[index] = lpos;

//             for(uint i=0; i<9; ++i) {
//                 cv::Point2f &scene_pos = scenePoints[i];
//                 d = distance(lpos_cm, scene_pos);
//                 if(d < min_dist) {
//                     min_dist = d;
//                     min_ind = i + 1;
//                 }
//             }
//             in_objective1[index] = min_ind;
//             this->odist1[index] = min_dist;
//         }
//     }

//     if(rstep) {
//         std::vector<cv::Point2i> rightCont;
//         rightCont.push_back(cv::Point2i(right.x, right.y));
//         rightCont.push_back(cv::Point2i(right.x + right.width - 1, right.y));
//         rightCont.push_back(cv::Point2i(right.x + right.width - 1, right.y + right.height - 1));
//         rightCont.push_back(cv::Point2i(right.x, right.y + right.height - 1));
//         bool intersection_ok = false;
//         for(uint i=1; i<=9; ++i) {
//             std::vector<cv::Point2i> imPos;
//             imPos.push_back(objImPos[i][0]);
//             imPos.push_back(objImPos[i][1]);
//             imPos.push_back(objImPos[i][2]);
//             imPos.push_back(objImPos[i][3]);
//             std::vector<cv::Point2i> inter = intersectConvexPolygons(imPos, rightCont);
//             if(inter.size() > 0) {
//                 double area = cv::contourArea(inter), obj_area = cv::contourArea(imPos);
//                 if(area > obj_area*objective_coverage_rate) {
//                     in_objective2[index] = i;
//                     this->odist2[index] = 0;
//                     right_foot[index] = getStepPosition(frame, right);
//                     intersection_ok = true;
//                 }
//                 break; //If not enough still stop (a step is not that big)
//             }
//         }

//         if(!intersection_ok) {
//             float d, min_dist = FLT_MAX;
//             int min_ind = 0;
//             cv::Point2f rpos = getStepPosition(frame, right), rpos_cm = transformInv(rpos);
//             right_foot[index] = rpos;

//             for(uint i=0; i<9; ++i) {
//                 cv::Point2f &scene_pos = scenePoints[i];
//                 d = distance(rpos_cm, scene_pos);
//                 if(d < min_dist) {
//                     min_dist = d;
//                     min_ind = i + 1;
//                 }
//             }
//             in_objective2[index] = min_ind;
//             this->odist2[index] = min_dist;
//         }
//     }
// }

// Calculates the distance between a point and a line segment
float calculatePointToLineDistance(const cv::Point2f &pointA, const cv::Point2f &pointB, const cv::Point2f &point)
{
    float segmentLength = cv::norm(pointB - pointA);
    if (segmentLength == 0.0)
        return cv::norm(point - pointA);

    float t = ((point.x - pointA.x) * (pointB.x - pointA.x) + (point.y - pointA.y) * (pointB.y - pointA.y)) / (segmentLength * segmentLength);
    t = std::max(0.0f, std::min(1.0f, t));
    cv::Point2f projection = pointA + t * (pointB - pointA);
    return cv::norm(point - projection);
}

/*bool doLineSegmentsIntersect(const cv::Point2f &p1, const cv::Point2f &q1, const cv::Point2f &p2, const cv::Point2f &q2)
{
    auto orientation = [](const cv::Point2f &p, const cv::Point2f &q, const cv::Point2f &r)
    {
        float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if (val == 0)
            return 0;             // colinear
        return (val > 0) ? 1 : 2; // clock or counterclock wise
    };

    // Check the four orientations needed for general and special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4)
        return true;

    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && cv::boundingRect({p1, q1, p2}).contains(p2))
        return true;

    // p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && cv::boundingRect({p1, q1, q2}).contains(q2))
        return true;

    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && cv::boundingRect({p2, q2, p1}).contains(p1))
        return true;

    // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && cv::boundingRect({p2, q2, q1}).contains(q1))
        return true;

    return false; // Doesn't fall in any of the above cases
}*/

// Función de utilidad para encontrar la orientación de un trío ordenado de puntos
int orientation(const cv::Point2f &p, const cv::Point2f &q, const cv::Point2f &r) {
    float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;  // colinear
    return (val > 0) ? 1 : 2;  // clock or counterclock wise
}

// Función para verificar si dos segmentos de línea (p1, q1) y (p2, q2) se cruzan
bool doIntersect(const cv::Point2f &p1, const cv::Point2f &q1, const cv::Point2f &p2, const cv::Point2f &q2) {
    // Encuentra las cuatro orientaciones necesarias para los casos generales y especiales
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // Caso general
    if (o1 != o2 && o3 != o4) {
        return true;
    }

    // Casos especiales no se aplican en este contexto simplificado
    return false;
}

// Verifica si el contorno está completamente dentro de una forma definida por puntos
bool isContourFullyInside(const std::vector<cv::Point2f> &shapePoints, const std::vector<cv::Point2f> &contour) {
    for (const auto &point : contour) {
        if (cv::pointPolygonTest(shapePoints, point, false) < 0) {
            return false;  // Si algún punto está fuera, el contorno no está completamente dentro
        }
    }
    return true;  // Todos los puntos están dentro de la forma
}

// Función principal que verifica si un rectángulo intersecta con un contorno
bool rectIntersectsContour(const cv::Rect &rect, const std::vector<cv::Point2f> &contour) {
    // Convertir rectángulo a contorno de puntos
    std::vector<cv::Point2f> rectPoints = {
        {rect.x, rect.y},
        {rect.x + rect.width, rect.y},
        {rect.x + rect.width, rect.y + rect.height},
        {rect.x, rect.y + rect.height}
    };

    // Verificar si alguna esquina del rectángulo está dentro del contorno
    for (const auto &point : rectPoints) {
        if (cv::pointPolygonTest(contour, point, false) > 0) {
            return true;
        }
    }

    // Verificar si el contorno está completamente dentro de la forma definida por rectPoints
    if (isContourFullyInside(rectPoints, contour)) {
        return true;
    }

    // Verificar intersecciones entre las aristas del rectángulo y las del contorno
    std::vector<cv::Point2f> edges = {rectPoints[0], rectPoints[1], rectPoints[2], rectPoints[3], rectPoints[0]};
    for (size_t i = 0; i < edges.size() - 1; ++i) {
        for (size_t j = 0; j < contour.size(); ++j) {
            cv::Point2f nextContourPoint = contour[(j + 1) % contour.size()];
            if (doIntersect(edges[i], edges[i + 1], contour[j], nextContourPoint)) {
                return true;
            }
        }
    }

    return false;  // No se encontró intersección
}

// Threshold for determining closeness to a contour
const float CLOSENESS_THRESHOLD = 10.0f;

/*std::vector<cv::Point2f> convertToPoint2f(const std::vector<Point>& points) {
    std::vector<cv::Point2f> points2f;
    for (const auto& p : points) {
        points2f.emplace_back(static_cast<float>(p.a), static_cast<float>(p.b));
    }
    return points2f;
}*/

// // Determines if a step intersects with or is close to a contour
// int FeetTracker::intersectsObjective(cv::Mat &img, int index, int frame, cv::Rect &leftStep, bool leftStepOccurred, cv::Rect &rightStep, bool rightStepOccurred)
// {
//     auto minDistanceToRectContour = [&](const cv::Rect &stepRect, const std::vector<cv::Point2f> &contour, cv::Mat &img) -> float
//     {
//         std::vector<cv::Point2f> rectPoints = {
//             cv::Point2f(stepRect.tl()),
//             cv::Point2f(stepRect.br().x, stepRect.tl().y),
//             cv::Point2f(stepRect.br()),
//             cv::Point2f(stepRect.tl().x, stepRect.br().y)};

//         float minDistance = FLT_MAX;
//         for (const auto &rectPoint : rectPoints)
//         {
//             for (int i = 0; i < contour.size(); i++)
//             {
//                 float distance = calculatePointToLineDistance(contour[i], contour[(i + 1) % contour.size()], rectPoint);
//                 minDistance = std::min(minDistance, distance);
//             }
//         }
//         return minDistance;
//     };

    
//     for (int contourIndex = 0; contourIndex < this->contours.size(); contourIndex++)
//     {
//         const Contour &contour = this->contours[contourIndex];

//         if (true)
//         {
//             float minDistance = FLT_MAX;
//             bool intersects = false;
//             for (int contourIndex = 0; contourIndex < this->contours.size(); contourIndex++)
//             {
//                 std::vector<cv::Point2f> contourPoints2f = convertToPoint2f(contour.points);
//                 float distance = minDistanceToRectContour(leftStep, contourPoints2f, img);
//                 minDistance = std::min(minDistance, distance);
//                 intersects = rectIntersectsContour(leftStep, contourPoints2f);

//             }

//             if (intersects && leftStepOccurred)
//             {
//                 // Intersection or closeness logic for left step
//                 // this->leftClosestContourIndex[index] = contourIndex;
//                 // this->leftIntersectionDistance[index] = intersects ? 0 : distance; // Use 0 to indicate intersection
//                 this->odist1[index] = 0;
//                 this->left_foot[index] = getStepPosition(frame, leftStep);
//                 std::cout << "Intersect left_foot frame: " << frame << " contourIndex: "<< contourIndex <<std::endl;
//                 return contourIndex;
//             }
//             else {
//                 this->odist1[index] = minDistance;
//                 this->left_foot[index] = getStepPosition(frame, leftStep);
//             }
//         }

//         if (true)
//         {
//             float minDistance = FLT_MAX;
//             bool intersects = false;
//             for (int contourIndex = 0; contourIndex < this->contours.size(); contourIndex++)
//             {
//                 std::vector<cv::Point2f> contourPoints2f = convertToPoint2f(contour.points);
//                 float distance = minDistanceToRectContour(rightStep, contourPoints2f, img);
//                 minDistance = std::min(minDistance, distance);
//                 intersects = rectIntersectsContour(rightStep, contourPoints2f);

//             }
//             if (intersects && rightStepOccurred)
//             {
//                 // Intersection or closeness logic for right step
//                 // this->rightClosestContourIndex[index] = contourIndex;
//                 // this->rightIntersectionDistance[index] = intersects ? 0 : distance; // Use 0 to indicate intersection
//                 this->odist2[index] = 0;
//                 this->right_foot[index] = getStepPosition(frame, rightStep);
//                 std::cout << "Intersect right_foot frame: " << frame << " contourIndex: "<< contourIndex <<std::endl;
//                 return contourIndex;
//             }
//             else {
//                 this->odist2[index] = minDistance;
//                 this->right_foot[index] = getStepPosition(frame, rightStep);
//             }
//         }
//     }
//     return 1000;
// }




// Determines if a step intersects with or is close to a contour
int FeetTracker::intersectsObjective(cv::Mat &img, int index, int frame, cv::Rect &leftStep, bool leftStepOccurred, cv::Rect &rightStep, bool rightStepOccurred)
{
    auto minDistanceToRectContour = [&](const cv::Rect &stepRect, const std::vector<cv::Point2f> &contour) -> float
    {
        std::vector<cv::Point2f> rectPoints = {
            cv::Point2f(stepRect.tl()),
            cv::Point2f(stepRect.br().x, stepRect.tl().y),
            cv::Point2f(stepRect.br()),
            cv::Point2f(stepRect.tl().x, stepRect.br().y)};

        float minDistance = FLT_MAX;
        for (const auto &rectPoint : rectPoints)
        {
            for (int i = 0; i < contour.size(); i++)
            {
                float distance = calculatePointToLineDistance(contour[i], contour[(i + 1) % contour.size()], rectPoint);
                minDistance = std::min(minDistance, distance);
            }
        }
        return minDistance;
    };

    float distance_left = 0;
    float minDistance_left = FLT_MAX;
    float id_minDist_left = FLT_MAX;
    bool intersects_left = false;

    float distance_right = 0;
    float minDistance_right = FLT_MAX;
    float id_minDist_right = FLT_MAX;
    bool intersects_right = false;

    bool flagIntersect = false;
    bool flagFirst = true;

    for (int contourIndex = 0; contourIndex < this->contours.size(); contourIndex++)
    {
        const Contour &contour = this->contours[contourIndex];
        //std::vector<cv::Point2f> contourPoints2f = convertToPoint2f(contour.points);
        distance_left = minDistanceToRectContour(leftStep, contour.points);
        if (distance_left < minDistance_left) {
            minDistance_left = distance_left;
            id_minDist_left = contourIndex;
        }

        if (id_minDist_left != 4 && flagFirst) {
            flagFirst = false;
        }

        intersects_left = rectIntersectsContour(leftStep, contour.points);
        
        if ((flagFirst && (minDistance_left < CLOSENESS_THRESHOLD) && leftStepOccurred) || (!flagFirst && (intersects_left && leftStepOccurred)))
        {
            // Intersection or closeness logic for left step
            this->odist1[index] = 0;
            this->left_foot[index] = getStepPosition(frame, leftStep);
            this->in_objective1[index] = contourIndex;
            std::cout << "Intersect left_foot frame: " << frame << " contourIndex: "<< contourIndex <<std::endl;
            flagIntersect = true;
            this->left_intersects[index] = 1;
            this->right_intersects[index] = 1;
        }
        else {
            this->odist1[index] = minDistance_left;
            this->in_objective1[index] = id_minDist_left;
            this->left_foot[index] = getStepPosition(frame, leftStep);
            this->left_intersects[index] = 0;
        }

        distance_right = minDistanceToRectContour(rightStep, contour.points);
        if (distance_right < minDistance_right) {
            minDistance_right = distance_right;
            id_minDist_right = contourIndex;
        }

        if (id_minDist_right != 4 && flagFirst) {
            flagFirst = false;
        }

        intersects_right = rectIntersectsContour(rightStep, contour.points);

        if ((flagFirst && (minDistance_right < CLOSENESS_THRESHOLD) && rightStepOccurred) || (!flagFirst && (intersects_right && rightStepOccurred)))
        {
            // Intersection or closeness logic for right step
            this->odist2[index] = 0;
            this->right_foot[index] = getStepPosition(frame, rightStep);
            this->in_objective2[index] = contourIndex;
            std::cout << "Intersect right_foot frame: " << frame << " contourIndex: "<< contourIndex <<std::endl;
            flagIntersect = true;
            this->left_intersects[index] = 1;
            this->right_intersects[index] = 1;
        }
        else {
            this->odist2[index] = minDistance_right;
            this->in_objective2[index] = id_minDist_right;
            this->right_foot[index] = getStepPosition(frame, rightStep);
            this->right_intersects[index] = 0;
        }
        if (flagIntersect) {
            return contourIndex;
        }
    }
    return 1000;
}








// Note: You need to define how close a step needs to be to a Contorno to be considered intersecting or in proximity.
// Adjust the threshold in the condition accordingly.

void FeetTracker::insideObjective(int index, int frame, cv::Rect &left, bool lstep, cv::Rect &right, bool rstep)
{

    auto minDistanceToPointContour = [&](const cv::Point2f &stepPoint, const std::vector<cv::Point2f> &contour) -> float
    {
        float minDistance = FLT_MAX;
        for (int i = 0; i < contour.size(); i++)
        {
            float distance = calculatePointToLineDistance(contour[i], contour[(i + 1) % contour.size()], stepPoint);
            minDistance = std::min(minDistance, distance);
        }
        return minDistance;
    };

    if (!lstep && !rstep)
    {
        return;
    }

    if (lstep)
    {
        float distance = 0;
        float min_dist = FLT_MAX;
        int min_ind = 0;
        cv::Point2f lpos = getStepPosition(frame, left), lpos_cm = transformInv(lpos);
        left_foot[index] = lpos;

        for (int contourIndex = 0; contourIndex < this->contours.size(); contourIndex++)
        {
            const Contour &contour = this->contours[contourIndex];
            //std::vector<cv::Point2f> contourPoints2f = convertToPoint2f(contour.points);
            distance = minDistanceToPointContour(lpos_cm, contour.points);
            if (distance < min_dist)
            {
                min_dist = distance;
                min_ind = contourIndex + 1;
            }
            // minDistance = std::min(min_dist, distance);
            // intersects = rectIntersectsContour(left, contourPoints2f);
        }
        // for (uint i = 0; i < 9; ++i)
        // {
        //     cv::Point2f &scene_pos = scenePoints[i];
        //     d = distance(lpos_cm, scene_pos);
        //     if (d < min_dist)
        //     {
        //         min_dist = d;
        //         min_ind = i + 1;
        //     }
        // }
        in_objective1[index] = min_ind;
        this->odist1[index] = min_dist;
    }

    if (rstep)
    {
        float distance = 0;
        float min_dist = FLT_MAX;
        int min_ind = 0;
        cv::Point2f rpos = getStepPosition(frame, right), rpos_cm = transformInv(rpos);
        left_foot[index] = rpos;

        for (int contourIndex = 0; contourIndex < this->contours.size(); contourIndex++)
        {
            const Contour &contour = this->contours[contourIndex];
            //std::vector<cv::Point2f> contourPoints2f = convertToPoint2f(contour.points);
            distance = minDistanceToPointContour(rpos_cm, contour.points);
            if (distance < min_dist)
            {
                min_dist = distance;
                min_ind = contourIndex + 1;
            }
        }
        in_objective2[index] = min_ind;
        this->odist2[index] = min_dist;
    }
}

// int FeetTracker::insideObjective(cv::Rect &left, bool lstep, cv::Rect &right, bool rstep, std::map<int, std::vector<cv::Point2i>> &objectiveImPos)
// {
//     if (!lstep && !rstep)
//         return -1;
//     cv::Point2i pl, pr;
//     double ldist, rdist, ref_dist = 0.5;
//     for (uint i = 1; i <= 9; ++i)
//     {
//         std::vector<cv::Point2i> &pos = objectiveImPos[i];
//         if (lstep)
//         {
//             pl = getRectCenter(left);
//             ldist = cv::pointPolygonTest(pos, pl, true);
//             if (ldist > ref_dist)
//                 return i;
//         }
//         if (rstep)
//         {
//             pr = getRectCenter(right);
//             rdist = cv::pointPolygonTest(pos, pr, true);
//             if (rdist > ref_dist)
//                 return i;
//         }
//     }
//     return -1;
// }

void FeetTracker::drawObjectives(cv::Mat &img, int id)
{
    for (const auto& contour : contours) {
        std::vector<cv::Point> cv_contour;
        for (const auto& point : contour.points) {
            cv_contour.emplace_back(point.x, point.y);
        }
        if (contour.z == id){
            cv::Rect boundingBox = cv::boundingRect(cv_contour);
            cv::Point center = cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            std::string text = "ID: " + std::to_string(id);
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 1.0;
            int thickness = 2;
            cv::putText(img, text, center, fontFace, fontScale, cv::Scalar(255, 0, 255), thickness);
            cv::polylines(img, cv_contour, true, cv::Scalar(255, 0, 255), 2);
        }else{
            cv::polylines(img, cv_contour, true, cv::Scalar(255, 0, 0), 2);
        }
    }

    /*for (int i = 1; i <= 9; ++i)
    {
        std::vector<cv::Point2i> &pos = objImPos[i];
        if ((i == obj1 && d1 < max_cm_to_center) || (i == obj2 && d2 < max_cm_to_center))
            cv::polylines(img, pos, true, cv::Scalar(0, 255, 0), 2);
        else
            cv::polylines(img, pos, true, cv::Scalar(255, 255, 0));
    }*/
}

/*void FeetTracker::drawObjectives(cv::Mat &img, int obj, std::map<int, std::vector<cv::Point2i>> &objectiveImPos)
{
    for (int i = 1; i <= 9; ++i)
    {
        std::vector<cv::Point2i> &pos = objectiveImPos[i];
        if (i == obj)
            cv::polylines(img, pos, true, cv::Scalar(0, 255, 0), 2);
        else
            cv::polylines(img, pos, true, cv::Scalar(255, 255, 0));
    }
}*/

// Blind association of left and right
void FeetTracker::associateLeftAndRight(std::vector<cv::Rect> &fbboxes, cv::Rect &left, cv::Rect &right)
{
    if (fbboxes.size() == 1)
    { // One candidate
        left = fbboxes[0];
        right = fbboxes[0];
    }
    else if (fbboxes.size() == 2)
    { // Exactly two candidates
        if (fbboxes[0].x < fbboxes[1].x)
        {
            left = fbboxes[0];
            right = fbboxes[1];
        }
        else
        {
            left = fbboxes[1];
            right = fbboxes[0];
        }
    }
    else
    {
        int i, bsize = fbboxes.size(), lowest_ind, second_ind, lowest_y = 0, y;
        // First lowest
        for (i = 0; i < bsize; ++i)
        {
            cv::Rect &r = fbboxes[i];
            y = r.y + r.height;
            if (y > lowest_y)
            {
                lowest_y = y;
                lowest_ind = i;
            }
        }
        // Second lowest
        lowest_y = 0;
        for (i = 0; i < bsize; ++i)
        {
            if (i != lowest_ind)
            {
                cv::Rect &r = fbboxes[i];
                y = r.y + r.height;
                if (y > lowest_y)
                {
                    lowest_y = y;
                    second_ind = i;
                }
            }
        }
        cv::Rect &r1 = fbboxes[lowest_ind], &r2 = fbboxes[second_ind];
        if (r1.x < r2.x)
        {
            left = r1;
            right = r2;
        }
        else
        {
            left = r2;
            right = r1;
        }
    }
}

// Associate feet to one-candidate current frame
void FeetTracker::associateLeftAndRight(std::vector<cv::Rect> &near_bboxes, int near_frame,
                                        std::vector<cv::Rect> &far_bboxes, int far_frame,
                                        cv::Rect &current, int cframe, cv::Rect &cleft, cv::Rect &cright, cv::Mat &mask)
{
    // First associate frames with lowest projected distance to current bbox
    float min_dx, min_dy, dx, dy, dist, min_dist = FLT_MAX;
    cv::Point2i ccenter = getRectCenter(current), aux;
    int i, j, far_c, near_c, nfar = far_bboxes.size(), nnear = near_bboxes.size(),
                             near_fdist = near_frame - cframe, far_fdist = far_frame - near_frame,
                             cx = ccenter.x, cy = ccenter.y;
    int x_near[nnear], y_near[nnear], x_far[nfar], y_far[nfar], xf, yf, xn, yn,
        w_near[nnear], h_near[nnear], w_far[nfar], h_far[nfar];

    for (i = 0; i < nfar; ++i)
    {
        cv::Rect &r_aux = far_bboxes[i];
        aux = getRectCenter(r_aux);
        x_far[i] = aux.x;
        y_far[i] = aux.y;
        w_far[i] = r_aux.width;
        h_far[i] = r_aux.height;
    }
    for (i = 0; i < nnear; ++i)
    {
        cv::Rect &r_aux = near_bboxes[i];
        aux = getRectCenter(r_aux);
        x_near[i] = aux.x;
        y_near[i] = aux.y;
        w_near[i] = r_aux.width;
        h_near[i] = r_aux.height;
    }

    // 1. Find min distance to current combination:
    for (i = 0; i < nfar; ++i)
    {
        xf = x_far[i];
        yf = y_far[i];
        for (j = 0; j < nnear; ++j)
        {
            xn = x_near[j];
            yn = y_near[j];
            // Displacement vector (normalized by frame distance)
            dx = xn - xf;
            dy = yn - yf;
            dx /= far_fdist;
            dy /= far_fdist;
            // Distance between predicted and current center (displaced according to distance near<->current)
            dist = abs(xn + dx * near_fdist - cx) + abs(yn + dy * near_fdist - cy);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_dx = dx;
                min_dy = dy;
                far_c = i;
                near_c = j;
            }
        }
    }

    // 2. Find min distance of second association to current box, considering that second foot shall be near
    //    as only one candidate was found:
    float min_dx2, min_dy2, min_dist2 = FLT_MAX, xpos, ypos, min_xpos, min_ypos;
    int far_c2, near_c2;

    for (i = 0; i < nfar; ++i)
    {
        if (i == far_c)
            continue;
        xf = x_far[i];
        yf = y_far[i];
        for (j = 0; j < nnear; ++j)
        {
            if (j == near_c)
                continue;
            xn = x_near[j];
            yn = y_near[j];
            // Displacement vector (normalized by frame distance)
            dx = xn - xf;
            dy = yn - yf;
            dx /= far_fdist;
            dy /= far_fdist;
            // Distance between predicted and current center (displaced according to distance near<->current)
            xpos = xn + dx * near_fdist;
            ypos = yn + dy * near_fdist;
            dist = abs(xpos - cx) + abs(ypos - cy);
            if (dist < min_dist2)
            {
                min_dist2 = dist;
                min_dx2 = dx;
                min_dy2 = dy;
                far_c2 = i;
                near_c2 = j;
                min_xpos = xpos;
                min_ypos = ypos;
            }
        }
    }

    // 3. Adjust second box (as shall be occluded), to the best fit to mask, according to velocity vector, and
    //    considering mean size of best remaining candidates:
    int mw = round((w_far[far_c2] + w_near[near_c2]) / 2.0),
        mh = round((w_far[far_c2] + w_near[near_c2]) / 2.0),
        mx = min_xpos - mw / 2,
        my = min_ypos - mh / 2;
    cv::Rect best_fit;
    fitRectToMask(mx, my, mw, mh, min_dx2, min_dy2, best_fit, mask, current);
    cv::Point best_center = getRectCenter(best_fit);
    if (x_near[near_c] < x_near[near_c2])
    { // First is left
        cleft = current;
        cright = best_fit;
        left_rects.push_back(cleft);
        Dx_left.push_back((x_near[near_c] - cx) / near_fdist);
        Dy_left.push_back((y_near[near_c] - cy) / near_fdist);
        right_rects.push_back(cright);
        Dx_right.push_back((x_near[near_c2] - best_center.x) / near_fdist);
        Dy_right.push_back((y_near[near_c2] - best_center.y) / near_fdist);
    }
    else
    {
        cright = current;
        cleft = best_fit;
        left_rects.push_back(cleft);
        Dx_left.push_back((x_near[near_c2] - best_center.x) / near_fdist);
        Dy_left.push_back((y_near[near_c2] - best_center.y) / near_fdist);
        right_rects.push_back(cright);
        Dx_right.push_back((x_near[near_c] - cx) / near_fdist);
        Dy_right.push_back((y_near[near_c] - cy) / near_fdist);
    }
}

// Associate feet to two (or more)-candidate current frame, with next with one candidate (no other next with two candidates)
void FeetTracker::associateLeftAndRight(std::vector<cv::Rect> &cbboxes, cv::Rect &cleft, cv::Rect &cright,
                                        cv::Rect &next_bbox)
{
    // First select current two candidates with lowest distance to next bbox
    int i, csize = cbboxes.size();
    cv::Point2i p = getRectCenter(next_bbox);

    if (csize == 2)
    { // If there are just two, set left and right:
        cv::Rect &r1 = cbboxes[0];
        cv::Rect &r2 = cbboxes[1];
        if (r1.x < r2.x)
        {
            cleft = r1;
            cright = r2;
        }
        else
        {
            cleft = r2;
            cright = r1;
        }

        // Set velocities and store data:
        cv::Point2i pl = getRectCenter(cleft), pr = getRectCenter(cright);
        left_rects.push_back(cleft);
        right_rects.push_back(cright);
        Dx_left.push_back(p.x - pl.x);
        Dx_right.push_back(p.x - pr.x);
        // Displace velocity vector a little bit up for farthest feet
        if (abs(p.x - pl.x) + abs(p.y - pl.y) < abs(p.x - pr.x) + abs(p.y - pr.y))
        {
            Dy_left.push_back(p.y - pl.y);
            Dy_right.push_back((p.y - 1) - pr.y);
        }
        else
        {
            Dy_left.push_back((p.y - 1) - pl.y);
            Dy_right.push_back(p.y - pr.y);
        }
    }
    else
    { // If more, search two nearest to only next candidate
        // 1. Find two min distances to next:
        cv::Rect first, second;
        int min_first = INT_MAX, min_second = INT_MAX, dist;
        cv::Point2i cp, p1, p2;
        for (i = 0; i < csize; ++i)
        {
            cv::Rect &r = cbboxes[i];
            cp = getRectCenter(r);
            dist = abs(p.x - cp.x) + abs(p.y - cp.y);
            if (dist < min_first)
            {
                second = first;
                min_second = min_first;
                p2 = p1;
                first = r;
                min_first = dist;
                p1 = cp;
            }
            else if (dist < min_second)
            {
                second = r;
                min_second = dist;
                p2 = cp;
            }
        }
        // 2. Set velocities and store data
        // estimate velocity vector as rough estimation of very near to ground association
        if (p1.x < p2.x)
        { // First is left
            cleft = first;
            cright = second;
            left_rects.push_back(cleft);
            Dx_left.push_back(p.x - p1.x);
            Dy_left.push_back(p.y - p1.y);
            right_rects.push_back(cright);
            Dx_right.push_back(p.x - p2.x);
            Dy_right.push_back((p.y - 1) - p2.y);
        }
        else
        {
            cleft = second;
            cright = first;
            left_rects.push_back(cleft);
            Dx_left.push_back(p.x - p2.x);
            Dy_left.push_back((p.y - 1) - p2.y);
            right_rects.push_back(cright);
            Dx_right.push_back(p.x - p1.x);
            Dy_right.push_back(p.y - p1.y);
        }
    }
}

// Associate left and right: in case of several current candidates, if two(or more)-rect candidates frame is next one,
//     associate one with the best combination to the next one-candidate frame,
//     and the second as the lowest paired with the nearest remaining.
void FeetTracker::associateLeftAndRight(std::vector<cv::Rect> &cbboxes, std::vector<cv::Rect> &next_bboxes,
                                        cv::Rect &sole_bbox, cv::Rect &cleft, cv::Rect &cright)
{
    cv::Point2i p = getRectCenter(sole_bbox), aux;
    float min_dx, min_dy, dx, dy, dist, min_dist = FLT_MAX;
    int i, j, csize = cbboxes.size(), nsize = next_bboxes.size(),
              px = p.x, py = p.y;
    int xc[csize], yc[csize], xn[nsize], yn[nsize], xxc, yyc, xxn, yyn, bestc, bestn;
    for (i = 0; i < csize; ++i)
    {
        cv::Rect &r_aux = cbboxes[i];
        aux = getRectCenter(r_aux);
        xc[i] = aux.x;
        yc[i] = aux.y;
    }
    for (i = 0; i < nsize; ++i)
    {
        cv::Rect &r_aux = next_bboxes[i];
        aux = getRectCenter(r_aux);
        xn[i] = aux.x;
        yn[i] = aux.y;
    }

    // 1. Find min distance to current combination:
    for (i = 0; i < nsize; ++i)
    {
        xxn = xn[i];
        yyn = yn[i];
        for (j = 0; j < csize; ++j)
        {
            xxc = xc[j];
            yyc = yc[j];
            // Displacement vector (normalized by frame distance)
            dx = xxn - xxc;
            dy = yyn - yyc;
            // Distance between predicted and current center (displaced according to distance near<->current)
            dist = abs(xxn + dx - px) + abs(yyn + dy - py);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_dx = dx;
                min_dy = dy;
                bestc = j;
                bestn = i;
            }
        }
    }

    // 2. Set first and second bboxes:
    cv::Rect first = cbboxes[bestc], second;
    int lowest = 0, low_ind;
    for (j = 0; j < csize; ++j)
    {
        if (j == bestc)
            continue;
        cv::Rect &s = cbboxes[j];
        yyc = s.y + s.height - 1;
        if (yyc > lowest)
        {
            lowest = yyc;
            second = s;
            low_ind = j;
        }
    }

    xxc = xc[low_ind];
    yyc = yc[low_ind];

    // 3. Find min distance to second:
    int min_dx2, min_dy2, secn;
    min_dist = FLT_MAX;
    for (i = 0; i < nsize; ++i)
    {
        if (i == bestn)
            continue;
        xxn = xn[i];
        yyn = yn[i];
        // Displacement vector (normalized by frame distance)
        dx = xxn - xxc;
        dy = yyn - yyc;
        // Distance between predicted and current center (displaced according to distance near<->current)
        dist = abs(xxn + dx - px) + abs(yyn + dy - py);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_dx2 = dx;
            min_dy2 = dy;
            secn = i;
        }
    }

    // Set velocities and store
    if (xc[bestc] < xxc)
    { // First is left
        cleft = first;
        cright = second;
        left_rects.push_back(cleft);
        Dx_left.push_back(xn[bestn] - xc[bestc]);
        Dy_left.push_back(yn[bestn] - yc[bestc]);
        right_rects.push_back(cright);
        Dx_right.push_back(xn[secn] - xxc);
        Dy_right.push_back(yn[secn] - yyc);
    }
    else
    {
        cleft = second;
        cright = first;
        left_rects.push_back(cleft);
        Dx_left.push_back(xn[secn] - xxc);
        Dy_left.push_back(yn[secn] - yyc);
        right_rects.push_back(cright);
        Dx_right.push_back(xn[bestn] - xc[bestc]);
        Dy_right.push_back(yn[bestn] - yc[bestc]);
    }
    from_one_candidate.push_back(false);
}

// Associate left and right: in case of several current candidates, if two(or more)-rect candidates frame is NOT next one,
//           associate one with the best combination to the next one-candidate frame. The second current will be the lowest,
//           and will fit to best mask position, in the line of the nearest remaining in the two(or more)-rect candidates frame.
//           cbboxes (current) --> sole_bbox --> next_bboxes
void FeetTracker::associateLeftAndRight(std::vector<cv::Rect> &cbboxes, std::vector<cv::Rect> &next_bboxes,
                                        cv::Rect &sole_bbox, int diff, cv::Rect &cleft, cv::Rect &cright)
{
    // First associate frames with lowest projected distance to sole bbox
    float min_dx, min_dy, dx, dy, dist, min_dist = FLT_MAX;
    cv::Point2i scenter = getRectCenter(sole_bbox), aux;
    int i, j, cindex, nindex, ncur = cbboxes.size(), nnext = next_bboxes.size(),
                              sx = scenter.x, sy = scenter.y;
    int cx[ncur], cy[ncur], nx[nnext], ny[nnext], cxx, cyy, nxx, nyy;

    for (i = 0; i < ncur; ++i)
    {
        cv::Rect &r_aux = cbboxes[i];
        aux = getRectCenter(r_aux);
        cx[i] = aux.x;
        cy[i] = aux.y;
    }
    for (i = 0; i < nnext; ++i)
    {
        cv::Rect &r_aux = next_bboxes[i];
        aux = getRectCenter(r_aux);
        nx[i] = aux.x;
        ny[i] = aux.y;
    }

    // 1. Find min distance to sole combination:
    for (i = 0; i < nnext; ++i)
    {
        nxx = nx[i];
        nyy = ny[i];
        for (j = 0; j < ncur; ++j)
        {
            cxx = cx[j];
            cyy = cy[j];
            // Displacement vector (normalized by frame distance)
            dx = nxx - cxx;
            dy = nyy - cyy;
            dx /= diff;
            dy /= diff;
            // Distance between predicted and sole bbox center (displaced according to distance current<->next)
            dist = abs(cxx + dx - sx) + abs(cyy + dy - sy);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_dx = dx;
                min_dy = dy;
                cindex = j;
                nindex = i;
            }
        }
    }

    // 2. Find min distance of second association to current box, considering that second foot shall be near
    //    to sole bbox:
    float min_dx2, min_dy2, min_dist2 = FLT_MAX;
    int cindex2, nindex2;

    for (i = 0; i < nnext; ++i)
    {
        if (i == nindex)
            continue;
        nxx = nx[i];
        nyy = ny[i];
        for (j = 0; j < ncur; ++j)
        {
            if (j == cindex)
                continue;
            cxx = cx[j];
            cyy = cy[j];
            // Displacement vector (normalized by frame distance)
            dx = nxx - cxx;
            dy = nyy - cyy;
            dx /= diff;
            dy /= diff;
            // Distance between predicted and sole bbox center (displaced according to distance current<->next)
            dist = abs(cxx + dx - sx) + abs(cyy + dy - sy);
            if (dist < min_dist2)
            {
                min_dist2 = dist;
                min_dx2 = dx;
                min_dy2 = dy;
                cindex2 = j;
                nindex2 = i;
            }
        }
    }

    // 3. Set velocities and store data:
    if (cx[cindex] < cx[cindex2])
    { // First is left
        cleft = cbboxes[cindex];
        cright = cbboxes[cindex2];
        left_rects.push_back(cleft);
        Dx_left.push_back((nx[nindex] - cx[cindex]) / diff);
        Dy_left.push_back((ny[nindex] - cy[cindex]) / diff);
        right_rects.push_back(cright);
        Dx_right.push_back((nx[nindex2] - cx[cindex2]) / diff);
        Dy_right.push_back((ny[nindex2] - cy[cindex2]) / diff);
    }
    else
    {
        cleft = cbboxes[cindex2];
        cright = cbboxes[cindex];
        left_rects.push_back(cleft);
        Dx_left.push_back((nx[nindex2] - cx[cindex2]) / diff);
        Dy_left.push_back((ny[nindex2] - cy[cindex2]) / diff);
        right_rects.push_back(cright);
        Dx_right.push_back((nx[nindex] - cx[cindex]) / diff);
        Dy_right.push_back((ny[nindex] - cy[cindex]) / diff);
    }
    from_one_candidate.push_back(false);
}

// Associate left and right: in case of several current candidates, if more than one two-rect candidate, assume lowest
//           candidate is valid, and search best velocity match.
//           For second, chose current with minimal projection error among best combinations of the next two frames.
void FeetTracker::associateLeftAndRight(std::vector<cv::Rect> &cbboxes,
                                        std::vector<cv::Rect> &n1_bboxes, int d1,
                                        std::vector<cv::Rect> &n2_bboxes, int d2,
                                        cv::Rect &cleft, cv::Rect &cright)
{
    // 1. Set first as lowest:
    int i, j, yyc, csize = cbboxes.size(), n1size = n1_bboxes.size(), n2size = n2_bboxes.size();
    cv::Rect first, second;
    int lowest = 0, low_ind;
    for (j = 0; j < csize; ++j)
    {
        cv::Rect &s = cbboxes[j];
        yyc = s.y + s.height - 1;
        if (yyc > lowest)
        {
            lowest = yyc;
            first = s;
            low_ind = j;
        }
    }
    cv::Point2i cp = getRectCenter(first), aux;

    // 2. Search best association to first
    int n1x[n1size], n1y[n1size], n2x[n2size], n2y[n2size], n1xx, n1yy, n2xx, n2yy, first_n1, first_n2;

    for (i = 0; i < n1size; ++i)
    {
        cv::Rect &r_aux = n1_bboxes[i];
        aux = getRectCenter(r_aux);
        n1x[i] = aux.x;
        n1y[i] = aux.y;
    }
    for (i = 0; i < n2size; ++i)
    {
        cv::Rect &r_aux = n2_bboxes[i];
        aux = getRectCenter(r_aux);
        n2x[i] = aux.x;
        n2y[i] = aux.y;
    }

    float min_dx, min_dy, dx, dy, dist, min_dist = FLT_MAX;
    for (i = 0; i < n1size; ++i)
    {
        n1xx = n1x[i];
        n1yy = n1y[i];
        for (j = 0; j < n2size; ++j)
        {
            n2xx = n2x[j];
            n2yy = n2y[j];
            // Displacement vector (normalized by frame distance)
            dx = n1xx - cp.x;
            dy = n1yy - cp.y;
            dx /= d1;
            dy /= d1;

            dist = abs(n1xx + dx * (d2 - d1) - n2xx) + abs(n1yy + dy * (d2 - d1) - n2yy);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_dx = dx;
                min_dy = dy;
                first_n1 = i;
                first_n2 = j;
            }
        }
    }

    // 3. Associate second, as the one of minimal projection error
    int sec_n1, sec_n2, min_dx2, min_dy2;
    cv::Point2i cp2;
    min_dist = FLT_MAX;

    if (csize == 2)
    { // If only one candidate remaining, just choose that and find best association from remaining nexts.
        second = cbboxes[1 - low_ind];
        cp2 = getRectCenter(second);
        for (i = 0; i < n1size; ++i)
        {
            if (i == first_n1)
                continue;
            n1xx = n1x[i];
            n1yy = n1y[i];
            for (j = 0; j < n2size; ++j)
            {
                if (j == first_n2)
                    continue;
                n2xx = n2x[j];
                n2yy = n2y[j];
                // Displacement vector (normalized by frame distance)
                dx = n1xx - cp2.x;
                dy = n1yy - cp2.y;
                dx /= d1;
                dy /= d1;
                // Distance between predicted and sole bbox center (displaced according to distance current<->next)
                dist = abs(n1xx + dx * (d2 - d1) - n2xx) + abs(n1yy + dy * (d2 - d1) - n2yy);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_dx2 = dx;
                    min_dy2 = dy;
                    sec_n1 = i;
                    sec_n2 = j;
                }
            }
        }
    }
    else
    { // Else, find the best projection combination among remaining ones
        int k;
        cv::Point2i paux;
        for (k = 0; k < csize; ++k)
        {
            if (k == low_ind)
                continue;
            cv::Rect &raux = cbboxes[k];
            paux = getRectCenter(raux);
            for (i = 0; i < n1size; ++i)
            {
                if (i == first_n1)
                    continue;
                n1xx = n1x[i];
                n1yy = n1y[i];
                for (j = 0; j < n2size; ++j)
                {
                    if (j == first_n2)
                        continue;
                    n2xx = n2x[j];
                    n2yy = n2y[j];

                    // Displacement vector (normalized by frame distance)
                    dx = n1xx - paux.x;
                    dy = n1yy - paux.y;
                    dx /= d1;
                    dy /= d1;

                    // Distance between predicted and sole bbox center (displaced according to distance current<->next)
                    dist = abs(n1xx + dx * (d2 - d1) - n2xx) + abs(n1yy + dy * (d2 - d1) - n2yy);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        min_dx2 = dx;
                        min_dy2 = dy;
                        sec_n1 = i;
                        sec_n2 = j;
                        second = raux;
                        cp2 = paux;
                    }
                }
            }
        }
    }

    // 3. Set velocities and store data:
    if (cp.x < cp2.x)
    { // First is left
        cleft = first;
        cright = second;
        left_rects.push_back(cleft);
        Dx_left.push_back((n1x[first_n1] - cp.x) / d1);
        Dy_left.push_back((n1y[first_n1] - cp.y) / d1);
        right_rects.push_back(cright);
        Dx_right.push_back((n1x[sec_n1] - cp2.x) / d1);
        Dy_right.push_back((n1y[sec_n1] - cp2.y) / d1);
    }
    else
    {
        cleft = second;
        cright = first;
        left_rects.push_back(cleft);
        Dx_left.push_back((n1x[sec_n1] - cp2.x) / d1);
        Dy_left.push_back((n1y[sec_n1] - cp2.y) / d1);
        right_rects.push_back(cright);
        Dx_right.push_back((n1x[first_n1] - cp.x) / d1);
        Dy_right.push_back((n1y[first_n1] - cp.y) / d1);
    }
    from_one_candidate.push_back(false);
}

//  If sole candidate position x is not in the interval, extend properly.
// Controlled search interval: Fit feet considering previous feet estimation to best mask fit [(dx,dy),2*(dx,dy)], following displacement vector.
//  If sole candidate position x is not in the interval, extend properly.
void FeetTracker::fitRectToMask(cv::Rect &prev, float dx, float dy, cv::Rect &best_fit, cv::Mat &mask, cv::Rect &current)
{
    int i, j, xi, yi, xf, yf, score, max_score = -1, best_x, best_y,
                                     rows = mask.rows, cols = mask.cols, step = mask.step, ind, prev_score1 = -1, prev_score2 = -1,
                                     w = prev.width, h = prev.height;
    uchar *data = mask.data;
    float xx, yy, norm = sqrt(dx * dx + dy * dy);
    if (norm == 0)
    {
        best_fit = prev;
        return;
    }
    cv::Point2f pprev(prev.x + dx, prev.y + dy);
    float x_min = std::fmin(pprev.x - dx / 2, pprev.x + dx),
          y_min = std::fmin(pprev.y - dy / 2, pprev.y + dy),
          x_max = std::fmax(pprev.x - dx / 2, pprev.x + dx),
          y_max = std::fmax(pprev.y - dy / 2, pprev.y + dy);
    if (x_min == x_max)
    { // Prevent precision problems
        x_min -= 0.01;
        x_max += 0.01;
    }
    if (y_min == y_max)
    { // Prevent precision problems
        y_min -= 0.01;
        y_max += 0.01;
    }
    dx /= norm;
    dy /= norm;
    int sign = dy > 0 ? 1 : -1;
    // xx = current.x;
    // yy = prev.y + sign*(fabs((xx - prev.x)*dy/dx));
    xx = (x_min + x_max) / 2.0;
    yy = (y_min + y_max) / 2.0;
    // Direction 1
    bool dir1_ready = false;
    for (; xx >= x_min && xx <= x_max && yy >= y_min && yy <= y_max; xx += dx, yy += dy)
    {
        xi = round(xx);
        yi = round(yy);
        if (xi < 0 || xi >= cols || yi < 0 || yi >= rows)
            break;
        xf = xi + w - 1;
        yf = yi + h - 1;
        if (xf >= cols)
            xf = cols - 1;
        if (yf >= rows)
            yf = rows - 1;
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score < prev_score1)
        {
            dir1_ready = true;
            break;
        }
        prev_score1 = score;
    }

    // Direction 2
    bool dir2_ready = false;
    //    xx = current.x - dx;
    //    yy = prev.y + sign*(fabs((xx - prev.x)*dy/dx)) - dy;
    xx = (x_min + x_max) / 2.0;
    yy = (y_min + y_max) / 2.0;
    for (; xx >= x_min && xx <= x_max && yy >= y_min && yy <= y_max; xx -= dx, yy -= dy)
    {
        xi = round(xx);
        yi = round(yy);
        if (xi < 0 || xi >= cols || yi < 0 || yi >= rows)
            break;
        xf = xi + w - 1;
        yf = yi + h - 1;
        if (xf >= cols)
            xf = cols - 1;
        if (yf >= rows)
            yf = rows - 1;
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score < prev_score2)
        {
            dir2_ready = true;
            break;
        }
        prev_score2 = score;
    }
    best_fit.x = best_x;
    best_fit.y = best_y;
    best_fit.width = w;
    best_fit.height = h;
}

// Fit current one-candidate frame bbox, considering size of previous
void FeetTracker::fitRectToMask(int w, int h, cv::Rect &current, cv::Mat &mask, cv::Rect &best_fit)
{
    int i, j, xx, xi, yi, xf, yf, score, max_score = -1, best_x, best_y,
                                         cols = mask.cols, step = mask.step, ind, prev_score1 = -1, prev_score2 = -1;
    uchar *data = mask.data;
    xx = current.x + current.width / 2 - w / 2;
    yi = current.y + current.height - h;
    yf = yi + h - 1;
    // Direction 1
    bool dir1_ready = false;
    for (xi = xx; true; ++xi)
    {
        if (xi >= cols)
        {
            dir1_ready = true;
            break;
        }
        xf = xi + w - 1;
        if (xf >= cols)
        {
            dir1_ready = true;
            break;
        }
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score < prev_score1)
        {
            dir1_ready = true;
            break;
        }
        prev_score1 = score;
    }

    // Direction 2
    bool dir2_ready = false;
    for (xi = xx - 1; true; --xi)
    {
        if (xi < 0)
        {
            dir2_ready = true;
            break;
        }
        xf = xi + w - 1;
        yf = yi + h - 1;
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score < prev_score2)
        {
            dir2_ready = true;
            break;
        }
        prev_score2 = score;
    }
    best_fit.x = best_x;
    best_fit.y = best_y;
    best_fit.width = w;
    best_fit.height = h;
}

// Controlled search interval: Start from sole current and follow leg searching for nearest point to projected
//   previous bbox. Tolerance of pixel match is relative to max found, but leaving percentual flexibility for
//   exploring.
void FeetTracker::fitRectToMask(int w, int h, cv::Point2i pprev, cv::Rect &best_fit, cv::Mat &mask, cv::Rect &current)
{
    int i, j, k, l, xi, yi, xf, yf, score, max_score = -1, best_x, best_y, min_y,
                                           rows = mask.rows, cols = mask.cols, step = mask.step, ind, dist,
                                           min_dist = INT_MAX, prev_dist = INT_MAX, xc, yc;
    uchar *data = mask.data;

    min_y = std::min(pprev.y - h / 2, current.y);
    xi = current.x;
    yi = current.y;

    for (i = yi; i >= min_y; --i)
    {
        if (i < 0 || i >= rows)
            break;
        yc = round(i + w / 2.0);
        // Direction 1
        prev_dist = INT_MAX; // Try at least one distance, then stop if gets far
        for (j = xi; true; ++j)
        {
            if (j < 0 || j >= cols)
                break;
            xf = j + w - 1;
            yf = i + h - 1;
            if (xf >= cols)
                xf = cols - 1;
            if (yf >= rows)
                yf = rows - 1;
            // Get score (best fit of mask following displacement vector):
            score = 0;
            for (k = i; k <= yf; ++k)
            { // Process score
                ind = k * step;
                for (l = j; l <= xf; ++l)
                    if (data[ind + l])
                        ++score;
            }
            if (score > max_score) // Store reference max score
                max_score = score;

            if (score < max_score * max_score_rate) // Stop (assuming that initial is reasonable)
                break;                              // Stop this branch

            xc = round(j + w / 2.0);
            dist = abs(xc - pprev.x) + abs(yc - pprev.y);
            if (dist < min_dist)
            { // Store good score of min distance
                min_dist = dist;
                best_x = xi;
                best_y = yi;
            }

            if (dist > prev_dist)
                break; // Stop this branch if goes farther

            prev_dist = dist;
        }

        // Direction 2
        prev_dist = INT_MAX; // Try at least one distance, then stop if gets far
        for (j = xi; true; --j)
        {
            if (j < 0 || j >= cols || i < 0 || i >= rows)
                break;
            xf = j + w - 1;
            yf = i + h - 1;
            if (xf >= cols)
                xf = cols - 1;
            if (yf >= rows)
                yf = rows - 1;
            // Get score (best fit of mask following displacement vector):
            score = 0;
            for (k = i; k <= yf; ++k)
            { // Process score
                ind = k * step;
                for (l = j; l <= xf; ++l)
                    if (data[ind + l])
                        ++score;
            }
            if (score > max_score) // Store reference max score
                max_score = score;

            if (score < max_score * max_score_rate) // Stop (assuming that initial is reasonable)
                break;                              // Stop this branch

            xc = round(j + w / 2.0);
            dist = abs(xc - pprev.x) + abs(yc - pprev.y);
            if (dist < min_dist)
            { // Store good score of min distance
                min_dist = dist;
                best_x = xi;
                best_y = yi;
            }

            if (dist > prev_dist)
                break; // Stop this branch if goes farther

            prev_dist = dist;
        }
    }
    best_fit.x = best_x;
    best_fit.y = best_y;
    best_fit.width = w;
    best_fit.height = h;
}

// Controlled search interval: Fit bbox on best possible position given previous size
//  for two against two candidates (current is real candidate position of current)
void FeetTracker::fitRectToMask(cv::Rect &prev, int w, int h, float dx, float dy, cv::Mat &mask,
                                cv::Rect &current, cv::Rect &best_fit)
{
    int i, j, xi, yi, xf, yf, score, max_score = -1, best_x, best_y,
                                     rows = mask.rows, cols = mask.cols, step = mask.step, ind, prev_score1 = -1, prev_score2 = -1;
    ;
    uchar *data = mask.data;
    float xx, yy, norm = sqrt(dx * dx + dy * dy);
    if (norm == 0)
    {
        best_fit = current;
        return;
    }

    cv::Point2f adapted_pcur(current.x + dx / 2, current.y + dy / 2); // Extra search in current direction
    float x_min = std::fmin(prev.x, adapted_pcur.x),
          y_min = std::fmin(prev.y, adapted_pcur.y),
          x_max = std::fmax(prev.x, adapted_pcur.x),
          y_max = std::fmax(prev.y, adapted_pcur.y);
    if (x_min == x_max)
    { // Prevent precision problems
        x_min -= 0.01;
        x_max += 0.01;
    }
    if (y_min == y_max)
    { // Prevent precision problems
        y_min -= 0.01;
        y_max += 0.01;
    }
    dx /= norm;
    dy /= norm;
    xx = current.x + current.width / 2 - w / 2;
    if (xx < x_min)
        x_min = xx - 0.01;
    else if (xx > x_max)
        x_max = xx + 0.01;
    yy = current.y + current.height / 2 - h / 2;
    if (yy < y_min)
        y_min = yy - 0.01;
    else if (yy > y_max)
        y_max = yy + 0.01;
    // Direction 1
    bool dir1_ready = false;
    for (; xx >= x_min && xx <= x_max && yy >= y_min && yy <= y_max; xx += dx, yy += dy)
    {
        xi = round(xx);
        yi = round(yy);
        xf = xi + w - 1;
        yf = yi + h - 1;
        if (xf >= cols)
            xf = cols - 1;
        if (yf >= rows)
            yf = rows - 1;
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score <= prev_score1)
        {
            dir1_ready = true;
            break;
        }
        prev_score1 = score;
    }

    // Direction 2
    bool dir2_ready = false;
    xx = current.x + current.width / 2 - w / 2 - dx;
    yy = current.y + current.height / 2 - h / 2 - dy;

    for (; xx >= x_min && xx <= x_max && yy >= y_min && yy <= y_max; xx -= dx, yy -= dy)
    {
        xi = round(xx);
        yi = round(yy);
        if (xi < 0 || xi >= cols || yi < 0 || yi >= rows)
            break;
        xf = xi + w - 1;
        yf = yi + h - 1;
        if (xf >= cols)
            xf = cols - 1;
        if (yf >= rows)
            yf = rows - 1;
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score <= prev_score2)
        {
            dir2_ready = true;
            break;
        }
        prev_score2 = score;
    }
    best_fit.x = best_x;
    best_fit.y = best_y;
    best_fit.width = w;
    best_fit.height = h;
}

// Fit initial rectangle to best mask fit, following displacement vector, and constrained by the candidate:
void FeetTracker::fitRectToMask(int x, int y, int w, int h, float dx, float dy,
                                cv::Rect &best_fit, cv::Mat &mask, cv::Rect &current)
{
    int i, j, xi, yi, xf, yf, score, max_score = -1, best_x, best_y, x_min, y_min, x_max, y_max,
                                     rows = mask.rows, cols = mask.cols, step = mask.step, ind, prev_score = -1;
    uchar *data = mask.data;
    float xx, yy, norm = sqrt(dx * dx + dy * dy);
    x_min = std::min(x, current.x);
    y_min = std::min(y, current.y);
    x_max = std::max(x, current.x);
    y_max = std::max(y, current.y);
    dx /= norm;
    dy /= norm;
    if (norm == 0)
    {
        best_fit.x = x;
        best_fit.y = y;
        best_fit.width = w;
        best_fit.height = h;
        return;
    }

    int sign = dy > 0 ? 1 : -1;
    xx = current.x;
    yy = y + sign * (fabs((xx - x) * dy / dx));
    // Direction 1
    for (; true; xx += dx, yy += dy)
    {
        xi = round(xx);
        yi = round(yy);
        if (xi < 0 || xi >= cols || yi < 0 || yi >= rows)
            break;
        xf = xi + w - 1;
        yf = yi + h - 1;
        if (xf >= cols)
            xf = cols - 1;
        if (yf >= rows)
            yf = rows - 1;
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score < prev_score)
            break;
        prev_score = score;
    }

    // Direction 2
    prev_score = -1;
    xx = current.x - dx;
    yy = y + sign * (fabs((xx - x) * dy / dx)) - dy;
    for (; true; xx -= dx, yy -= dy)
    {
        xi = round(xx);
        yi = round(yy);
        if (xi < 0 || xi >= cols || yi < 0 || yi >= rows)
            break;
        xf = xi + w - 1;
        yf = yi + h - 1;
        if (xf >= cols)
            xf = cols - 1;
        if (yf >= rows)
            yf = rows - 1;
        // Get score (best fit of mask following displacement vector):
        score = 0;
        for (i = yi; i <= yf; ++i)
        {
            ind = i * step;
            for (j = xi; j <= xf; ++j)
                if (data[ind + j])
                    ++score;
        }
        if (score > max_score)
        {
            max_score = score;
            best_x = xi;
            best_y = yi;
        }
        if (score < prev_score)
            break;
        prev_score = score;
    }
    best_fit.x = best_x;
    best_fit.y = best_y;
    best_fit.width = w;
    best_fit.height = h;
}

void FeetTracker::saveResult(cv::Mat &image, int frame)
{
    std::string sframe = std::to_string(frame), zeros;
    switch (sframe.size())
    {
    case 1:
        zeros = "0000";
        break;
    case 2:
        zeros = "000";
        break;
    case 3:
        zeros = "00";
        break;
    case 4:
        zeros = "0";
        break;
    default:
        zeros = "";
    }
    std::string fname = saveTrackingDir + "/" + zeros + sframe + ".jpg";
    // cv::Mat resized;
    // cv::resize(image, resized, cv::Size(2*image.cols, 2*image.rows));
    // cv::imwrite(fname, resized);
    cv::imwrite(fname, image);
}

void FeetTracker::saveStepsResult(cv::Mat &image, int frame)
{
    std::string sframe = std::to_string(frame), zeros;
    switch (sframe.size())
    {
    case 1:
        zeros = "0000";
        break;
    case 2:
        zeros = "000";
        break;
    case 3:
        zeros = "00";
        break;
    case 4:
        zeros = "0";
        break;
    default:
        zeros = "";
    }
    std::string fname = saveStepsDir + "/" + zeros + sframe + ".jpg";
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(2 * image.cols, 2 * image.rows));
    cv::imwrite(fname, resized);
}

cv::Mat FeetTracker::getPhantomTrackingImage(int cindex, int num)
{
    int init_frame = cindex - num < 0 ? 0 : cindex - num;
    cv::Mat initial = stored_oframes[init_frame], phantom;
    initial.copyTo(phantom);
    if (cindex == 0)
    {
        float dx_l = Dx_left.back(), dy_l = Dy_left.back(),
              dx_r = Dx_right.back(), dy_r = Dy_right.back();
        cv::Rect left = left_rects.back(), right = right_rects.back();
        cv::Point2i pleft = getRectCenter(left), pright = getRectCenter(right);
        cv::rectangle(phantom, left, cv::Scalar(0, 0, 255));                                             // Red for left
        cv::rectangle(phantom, right, cv::Scalar(0, 255, 255));                                          // Yellow for right
        cv::line(phantom, cv::Point(pleft.x - dx_l, pleft.y - dy_l), pleft, cv::Scalar(255, 255, 0));    // Cyan for left velocities
        cv::line(phantom, cv::Point(pright.x - dx_r, pright.y - dy_r), pright, cv::Scalar(255, 0, 255)); // Magenta for left velocities
    }
    else
    {
        int i, j, k, l, last = Dx_left.size() - 1, init_mark = last - num < 0 ? 0 : last - num;
        cv::Rect left = left_rects[init_mark], right = right_rects[init_mark];
        cv::Point2i pleft, pright;
        int ind, ind2, aux, cols = initial.cols, rows = initial.rows, step = initial.step;
        uchar *pdata = phantom.data, *data;
        float t = 1.0 - phantom_current_weight;

        cv::rectangle(phantom, left, cv::Scalar(0, 0, 255));    // Red for left
        cv::rectangle(phantom, right, cv::Scalar(0, 255, 255)); // Yellow for right

        for (i = init_mark + 1, j = init_frame + 1; i <= last; ++i, ++j)
        {
            cv::Mat current;
            stored_oframes[j].copyTo(current);
            left = left_rects[i];
            right = right_rects[i];
            cv::rectangle(current, left, cv::Scalar(0, 0, 255));    // Red for left
            cv::rectangle(current, right, cv::Scalar(0, 255, 255)); // Yellow for right
            data = current.data;
            for (k = 0; k < rows; ++k)
            {
                ind = step * k;
                for (l = 0; l < cols; ++l)
                {
                    ind2 = ind + 3 * l;
                    aux = t * pdata[ind2] + phantom_current_weight * data[ind2];
                    pdata[ind2] = aux > 255 ? 255 : (aux < 0 ? 0 : (uchar)aux);
                    aux = t * pdata[ind2 + 1] + phantom_current_weight * data[ind2 + 1];
                    pdata[ind2 + 1] = aux > 255 ? 255 : (aux < 0 ? 0 : (uchar)aux);
                    aux = t * pdata[ind2 + 2] + phantom_current_weight * data[ind2 + 2];
                    pdata[ind2 + 2] = aux > 255 ? 255 : (aux < 0 ? 0 : (uchar)aux);
                }
            }
        }

        float dx_l = Dx_left.back(), dy_l = Dy_left.back(),
              dx_r = Dx_right.back(), dy_r = Dy_right.back();
        pleft = getRectCenter(left);
        pright = getRectCenter(right);
        cv::line(phantom, cv::Point(pleft.x - dx_l, pleft.y - dy_l), pleft, cv::Scalar(255, 255, 0));    // Cyan for left velocities
        cv::line(phantom, cv::Point(pright.x - dx_r, pright.y - dy_r), pright, cv::Scalar(255, 0, 255)); // Magenta for left velocities
    }

    return phantom;
}

// STATIC
void FeetTracker::getSamplesAndBBoxesExceptLabel(cv::Mat &labels, cv::Mat &stats,
                                                 std::vector<cv::Point2i> &sample_points,
                                                 std::vector<cv::Rect> &bboxes, int i_x, int i_y, int label)
{
    int i, x, xx, xf, y, yy, w, h, cnum = stats.rows;
    for (i = 1; i < cnum; ++i)
    {
        if (i != label)
        {
            x = stats.at<int>(i, 0);
            y = stats.at<int>(i, 1);
            w = stats.at<int>(i, 2);
            h = stats.at<int>(i, 3);
            xx = x + i_x;
            yy = y + i_y;
            cv::Rect r(xx, yy, w, h);
            bboxes.push_back(r);
            xf = x + w - 1;
            for (; x <= xf; ++x)
            {
                if (labels.at<int>(y, x) == i)
                {
                    sample_points.push_back(cv::Point(x, y));
                    break;
                }
            }
        }
    }
}

void FeetTracker::getBBoxesExceptLabel(cv::Mat &stats, std::vector<cv::Rect> &bboxes, int i_x, int i_y, int label)
{
    int i, x, y, w, h, cnum = stats.rows;
    for (i = 1; i < cnum; ++i)
    {
        if (i != label)
        {
            x = stats.at<int>(i, 0);
            y = stats.at<int>(i, 1);
            w = stats.at<int>(i, 2);
            h = stats.at<int>(i, 3);
            cv::Rect r(x + i_x, y + i_y, w, h);
            bboxes.push_back(r);
        }
    }
}

void FeetTracker::getBBoxes(cv::Mat &labels, cv::Mat &stats, std::vector<cv::Rect> &bboxes, int i_x, int i_y, std::vector<cv::Point2i> &pointInRegion)
{
    int i, x, y, w, h, cnum = stats.rows, xf;
    for (i = 1; i < cnum; ++i)
    {
        x = stats.at<int>(i, 0);
        y = stats.at<int>(i, 1);
        w = stats.at<int>(i, 2);
        h = stats.at<int>(i, 3);
        cv::Rect r(x + i_x, y + i_y, w, h);
        bboxes.push_back(r);
        xf = x + w - 1;
        for (int xx = x; xx <= xf; ++xx)
            if (labels.at<int>(y, xx) == i)
            {
                pointInRegion.push_back(cv::Point2i(xx + i_x, y + i_y));
                break;
            }
    }
}