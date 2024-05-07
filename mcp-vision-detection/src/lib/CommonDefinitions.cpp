#include "CommonDefinitions.h"

//Dot product between vectors u and v: 
float dotProduct(const cv::Point2f &u, const cv::Point2f &v) {
    return u.x * v.x + u.y * v.y;    
}

// Function to calculate the magnitude of a vector
float magnitude(const cv::Point2f &v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Function to calculate the projection of v1 onto v2
cv::Point2f projectVector(cv::Point2f v1, cv::Point2f v2) {
    float dot = dotProduct(v1, v2);
    float mag_v2_squared = magnitude(v2) * magnitude(v2);
    return (dot / mag_v2_squared) * v2;
}


// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool segmentIntersection(cv::Point2i &o1, cv::Point2i &p1, cv::Point2i &o2, cv::Point2i &p2, cv::Point2i &r) {
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


std::vector<cv::Point2i> searchSegmentIntersections(cv::Point2i &pp1, cv::Point2i &pp2,
                                                    std::vector<cv::Point2i> &p,
                                                    std::vector<uint> &p_inter_id) {
    cv::Point2i r;
    std::vector<cv::Point2i> rr;
    uint psize = p.size();
    for(uint i=0; i<psize; ++i)
        if(segmentIntersection(pp1, pp2, p[i], p[(i+1)%psize], r)) { //There is intersectoin
            p_inter_id.push_back(i);
            rr.push_back(r);
        }
    return rr;
}

bool isPolygonIntersection(std::vector<cv::Point2i> &p1, std::vector<cv::Point2i> &p2) {
    std::vector<cv::Point2i> inter = intersectConvexPolygons(p1, p2);
    int l = inter.size();
    //None, point or only segment intersection, considered false
    if(l <= 2)
        return false;
    return true;
}

//For Convex polygon, max 2 intersections per segment...
std::vector<cv::Point2i> intersectConvexPolygons(std::vector<cv::Point2i> &p1, std::vector<cv::Point2i> &p2) {

     uint p2size = p2.size(), num_in = 0;
     int first_in = -1;
    std::vector<int> in(p2size);

    // Print points in p1
    std::cout << "Polygon p1 points:" << std::endl;
    for (const auto& point : p1) {
        std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
    }

    // Print points in p2
    std::cout << "Polygon p2 points:" << std::endl;
    for (const auto& point : p2) {
        std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
    }

    for(uint i=0; i<p2size; ++i)
        if((in[i] = cv::pointPolygonTest(p1, p2[i], true)) >= 0) {
            if(first_in == -1)
                first_in = i;
            ++num_in;
        }

    //Non inside: 
    if(num_in == 0) {
        uint p1size = p1.size(), num_in2 = 0;
        std::vector<int> in2(p1size);
        for(uint i=0; i<p1size; ++i)
            if( (in2[i] = cv::pointPolygonTest(p2, p1[i], true)) > 0)
                num_in2++;

        //Return wall if all wall points inside
        if(num_in2 == p1size)
            return p1;

        //If some point of wall inside, and non of the polygon p, case is covered inverting order.
        if(num_in2 > 0)
            return intersectConvexPolygons(p2, p1);

        //Else, check for intersection points, and add them (no point of one polygon into the other)
        std::vector<cv::Point2i> r, r2;
        bool added = false;
        for(uint j=0; j < p2size; ++j) {
            std::vector<uint> p1_inter_id;
            uint next = (j+1)%p2size;
            std::vector<cv::Point2i> intersections = searchSegmentIntersections(p2[j], p2[next], p1, p1_inter_id);
            uint num_int = intersections.size();
            for(uint k=0; k < num_int; ++k) {
                r.push_back(intersections[k]);
                added = true;
            }
        }

        if(added) {
            cv::convexHull(r, r2);
            return r2;
        }
        //If no input, return empty polygon
        return std::vector<cv::Point2i>();
    } else if(num_in == p2size)
        return p2;

    
    uint p1size = p1.size(), num_in2 = 0;
    std::vector<int> in2(p1size);
    for(uint i=0; i<p1size; ++i)
        if( (in2[i] = cv::pointPolygonTest(p2, p1[i], true)) > 0)
            num_in2++;

    uint i = first_in, j = 0;
    int other_p_out = -1;
    std::vector<cv::Point2i> r;

    while (j <= p2size) {
        if(in[i] >= 0) { //If current point in, add it
            if(other_p_out != -1) { //It comes from the outside
                std::vector<uint> p1_inter_id;
                uint prev = i == 0? p2size-1 : i-1;
                std::vector<cv::Point2i> intersections = searchSegmentIntersections(p2[prev], p2[i], p1, p1_inter_id);
                uint num_int = intersections.size();
                if(num_int != 1) { //If there is no just one, coming from inside, the other polygon is not convex?
                    //std::cerr << "Something is wrong!! Should have intersected!" << std::endl;
                    return r;
                }

                if(other_p_out != p1_inter_id[0]) { //It means we need to add wall points...
                    //Check begginning, end and sense:
                    uint beggining;
                    if(in2[other_p_out] < 0) //Starting point of wall segment is out, start from next;
                        beggining = (other_p_out + 1)%p1size;
                    else
                        beggining = other_p_out;
                    if(in2[(beggining+1)%p1size] > 0)
                        for(uint k=beggining; in2[k%p1size] > 0; ++k)
                            r.push_back(p1[k%p1size]);
                    else
                        for(uint k=beggining; in2[k%p1size] > 0; --k)
                            r.push_back(p1[k%p1size]);
                }
                //Reaching here, we add intersection point
                r.push_back(intersections[0]);
            }
            //Finally we add current inner point
            if(j < p2size)
                r.push_back(p2[i]);
            other_p_out = -1;
        } else { //If not... add intersections and inner points of the other polygons
            std::vector<uint> p1_inter_id;
            cv::Point2i last = r.back();
            uint prev = i==0 ? p2size-1 : i-1;
            std::vector<cv::Point2i> intersections = searchSegmentIntersections(p2[prev], p2[i], p1, p1_inter_id);
            uint num_int = intersections.size();
            if(other_p_out == -1) { //Comes from inside
                if(num_int != 1) { //If there is no just one, coming from inside, the other polygon is not convex?
                    //std::cerr << "Something is wrong!! Should have intersected!" << std::endl;
                    return r;
                }
                //Reaching here, we add intersection point
                r.push_back(intersections[0]);
                other_p_out = p1_inter_id[0];
            } else { //Comes from outside: Could be double intersection or none (if none, ignore)
                if(num_int == 2) { // Add both intersections, nearest first
                    cv::Point2i pp1 = intersections[0], pp2 = intersections[1];
                    float d1 = sqrt((last.x - pp1.x)*(last.x - pp1.x) + (last.y - pp1.y)*(last.y - pp1.y)),
                          d2 = sqrt((last.x - pp2.x)*(last.x - pp2.x) + (last.y - pp2.y)*(last.y - pp2.y));
                    if(d1 < d2) {
                        r.push_back(pp1);
                        r.push_back(pp2);
                        other_p_out = p1_inter_id[1];
                    } else if(d1 > d2) {
                        r.push_back(pp2);
                        r.push_back(pp1);
                        other_p_out = p1_inter_id[0];
                    } else { //Same point. A corner? Add one...
                        r.push_back(pp1);
                        other_p_out = p1_inter_id[0] > p1_inter_id[1] ? p1_inter_id[0] : p1_inter_id[1];
                    }
                } else if (num_int != 0) { // If is not 0 or 2, the other polygon is not convex?
                    //std::cerr << "Something is wrong!! Not possible in convex polygons!" << std::endl;
                    return r;
                }
            }
        }
        i = (i+1)%p2size;
        ++j;
    }

    return r;
}
