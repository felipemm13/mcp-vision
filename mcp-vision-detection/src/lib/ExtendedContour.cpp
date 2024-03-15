#include "ExtendedContour.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

//#define SHOW_INTERMEDIATE_RESULTS

extendedContour::extendedContour() {}

extendedContour::extendedContour(std::vector<cv::Point> c): contour(c) {
    findExtremeIndexes();
}

cv::Mat extendedContour::extendContour(cv::Mat &fg, cv::Mat &labels, cv::Mat &rr, cv::Rect &result_rect) {
    cv::Mat rfg = cv::Mat::zeros(fg.size(), CV_8UC1),
            rcont = cv::Mat::zeros(fg.size(), CV_8UC1);

    cv::Rect r(rr.at<int>(0), rr.at<int>(1), rr.at<int>(2), rr.at<int>(3));

    float margin_rate = 0.05; //Margin for extended search of near objects

    int i, j, w, h, label, blabel = rr.at<int>(4),
        wmargin = fg.cols*margin_rate, hmargin = fg.rows*margin_rate,
        xmin = r.x - wmargin, xmax = r.x + r.width - 1 + wmargin,
        ymin = r.y - hmargin, ymax = r.y + r.height - 1 + hmargin;
    if(xmin < 0) xmin = 0;
    if(ymin < 0) ymin = 0;
    if(xmax >= fg.cols) xmax = fg.cols - 1;
    if(ymax >= fg.rows) ymax = fg.rows - 1;

    std::map<int, cv::Rect> near_rects; //Use it first to store x1, y1, x2, y2, then convert
    uchar *rdata = rcont.data, *data = rfg.data;
    int step = rcont.step;

    //Inspect extended zone around big object
    for(i=ymin; i<=ymax; ++i)
        for(j=xmin; j<=xmax; ++j) {
            label = labels.at<int>(i,j);
            if(label != 0) {
                if(label != blabel) { //Different labels to the big object are stored separately to obtain their contours...
                    rdata[i*step + j] = 255;
                    if(near_rects.count(label) == 0) {
                        cv::Rect rnew(j,i,1,1);
                        near_rects[label] = rnew;
                    } else {
                        cv::Rect &rex = near_rects[label];
                        if(j < rex.x)
                            rex.x = j;
                        if(j > rex.x + rex.width - 1)
                            rex.width = j - rex.x + 1;
                        if(i < rex.y)
                            rex.y = i;
                        if(i > rex.y + rex.height - 1)
                            rex.height = i - rex.y + 1;
                    }
                } else { //It is the big object
                    data[i*step + j] = 255;
                }
            }
        }

    //Complete pixels of new regions:
    std::map<int, cv::Rect>::iterator nr_iter = near_rects.begin();
    int k, l, isize, n_xmin, n_xmax, n_ymin, n_ymax;
    for(; nr_iter != near_rects.end(); ++nr_iter) {
        label = nr_iter->first;
        cv::Rect &prev_rect = nr_iter->second;
        if(prev_rect.y == ymin) { //Complete upper pixels of new regions:
            n_ymin = ymin; //Modifiable
            n_ymax = prev_rect.y + prev_rect.height - 1; //Same
            i = ymin - 1;
            n_xmin = j = prev_rect.x;
            w = prev_rect.width;
            n_xmax = n_xmin + w - 1;
            while(true) {
                if(i<0) //Just in case, if reach image boundary
                    break;
                std::vector<cv::Point2i> cur_segs;
                setHIntervals(labels, label, j, i, w, cur_segs);
                isize = cur_segs.size();
                if(isize == 0) { //If no current, stop
                    prev_rect.x = n_xmin;
                    prev_rect.y = n_ymin;
                    prev_rect.width = n_xmax - n_xmin + 1;
                    prev_rect.height = n_ymax - n_ymin + 1;
                    break;
                }
                //Add to image
                for(k=0; k<isize; ++k)
                    memset(rdata + step*i + cur_segs[k].x, 255, cur_segs[k].y);
                //Adjust for next iteration
                n_ymin = i--; //Keep going up
                j = cur_segs[0].x; //Beggining of first segment
                if(j < n_xmin)
                    n_xmin = j;
                w = (cur_segs[isize-1].x + cur_segs[isize-1].y - 1) - cur_segs[0].x + 1;
                if(j + w - 1 > n_xmax)
                    n_xmax = j + w - 1;
            }
        } else if(prev_rect.y + prev_rect.height - 1 == ymax) { //Complete lower pixels of new regions:
            n_ymin = prev_rect.y;
            n_ymax = ymax;
            i = ymax + 1; //Going down
            n_xmin = j = prev_rect.x;
            w = prev_rect.width;
            n_xmax = n_xmin + w - 1;
            while(true) {
                if(i >= fg.rows) //Just in case, if reach image boundary
                    break;
                std::vector<cv::Point2i> cur_segs;
                setHIntervals(labels, label, j, i, w, cur_segs);
                isize = cur_segs.size();
                if(isize == 0) { //If no current, stop
                    prev_rect.x = n_xmin;
                    prev_rect.y = n_ymin;
                    prev_rect.width = n_xmax - n_xmin + 1;
                    prev_rect.height = n_ymax - n_ymin + 1;
                    break;
                }
                //Add to image
                for(k=0; k<isize; ++k)
                    memset(rdata + step*i + cur_segs[k].x, 255, cur_segs[k].y);
                //Adjust for next iteration
                n_ymax = i++; //Keep going down
                j = cur_segs[0].x; //Beggining of first segment
                if(j < n_xmin)
                    n_xmin = j;
                w = (cur_segs[isize-1].x + cur_segs[isize-1].y - 1) - cur_segs[0].x + 1;
                if(j + w - 1 > n_xmax)
                    n_xmax = j + w - 1;
            }
        }

        //Treat vertical separatelly as modifications could have changed the situation
        if(prev_rect.x <= xmin) { //Complete leftmost pixels of new regions:
            n_xmin = prev_rect.x; //Modifiable
            n_xmax = prev_rect.x + prev_rect.width - 1; //Same
            j = n_xmin - 1; //To the left
            n_ymin = i = prev_rect.y;
            h = prev_rect.height;
            n_ymax = n_ymin + h - 1;
            while(true) {
                if(j<0) //Just in case, if reach image boundary
                    break;
                std::vector<cv::Point2i> cur_segs;
                setVIntervals(labels, label, j, i, h, cur_segs);
                isize = cur_segs.size();
                if(isize == 0) { //If no current, stop
                    prev_rect.x = n_xmin;
                    prev_rect.y = n_ymin;
                    prev_rect.width = n_xmax - n_xmin + 1;
                    prev_rect.height = n_ymax - n_ymin + 1;
                    break;
                }
                //Add to image
                for(k=0; k<isize; ++k) {
                    l=cur_segs[k].x;
                    int ylim = l + cur_segs[k].y - 1;
                    for(; l<ylim; ++l)
                        rdata[step*l + j] = 255;
                }

                //Adjust for next iteration
                n_xmin = j--; //Keep going left
                i = cur_segs[0].x; //Beggining of first segment
                if(i < n_ymin)
                    n_ymin = i;
                h = (cur_segs[isize-1].x + cur_segs[isize-1].y - 1) - cur_segs[0].x + 1;
                if(i + h - 1 > n_ymax)
                    n_ymax = i + h - 1;
            }
        } else if(prev_rect.x + prev_rect.width - 1 >= xmax) { //Complete rightmost pixels of new regions:
            n_xmin = prev_rect.x; //Same
            n_xmax = prev_rect.x + prev_rect.width - 1;
            j = n_xmax + 1; //Going right
            n_ymin = i = prev_rect.y;
            h = prev_rect.height;
            n_ymax = n_ymin + h - 1;
            while(true) {
                if(j >= fg.cols) //Just in case, if reach image boundary
                    break;
                std::vector<cv::Point2i> cur_segs;
                setVIntervals(labels, label, j, i, h, cur_segs);
                isize = cur_segs.size();
                if(isize == 0) { //If no current, stop
                    prev_rect.x = n_xmin;
                    prev_rect.y = n_ymin;
                    prev_rect.width = n_xmax - n_xmin + 1;
                    prev_rect.height = n_ymax - n_ymin + 1;
                    break;
                }
                //Add to image
                for(k=0; k<isize; ++k) {
                    l=cur_segs[k].x;
                    int ylim = l + cur_segs[k].y - 1;
                    for(; l<ylim; ++l)
                        rdata[step*l + j] = 255;
                }
                //Adjust for next iteration
                n_xmax = j++; //Keep going down
                i = cur_segs[0].y; //Beggining of first segment
                if(i < n_ymin)
                    n_ymin = i;
                h = (cur_segs[isize-1].x + cur_segs[isize-1].y - 1) - cur_segs[0].x + 1;
                if(i + h - 1 > n_ymax)
                    n_ymax = i + h - 1;
            }

        }
    }

    //Get big contour
    std::vector<std::vector<cv::Point> > cbig;
    std::vector<std::vector<cv::Point> > cnear;
    std::vector<cv::Vec4i> hi;
    cv::findContours(rfg, cbig, hi, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //Get near contours
    cv::findContours(rcont, cnear, hi, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::Mat cont_view = cv::Mat::zeros(rfg.size(), CV_8UC3);
    cv::drawContours(cont_view, cbig, 0, cv::Scalar(0,0,255));
    cv::drawContours(cont_view, cnear, -1, cv::Scalar(0,255,255));
    cv::imshow("Contours", cont_view);
    cv::waitKey(0);
#endif

    //Prepare for distance calculations
    extendedContour ext_big(cbig[0]);
    std::vector<extendedContour> ext_near;
    for(uint i=0; i<cnear.size(); ++i)
        ext_near.push_back(extendedContour(cnear[i]));

    cv::Point np_big, np_cont;
    float distanceCriterion = rfg.cols*margin_rate/2.0; //Contour should be really near
    distanceCriterion *= distanceCriterion; //Work with square distance
    //std::cout << "Distance criterion: " << distanceCriterion << " pixels." << std::endl;
    for(uint i=0; i<ext_near.size(); ++i) {
        if(ext_big.getMinimalDistanceToContourSimple(ext_near[i], np_big, np_cont) < distanceCriterion) {
            nps_big.push_back(np_big);
            nps_near.push_back(np_cont);
            cintegrated.push_back(ext_near[i].contour);
        }
    }

    cv::Mat result;
    rfg.copyTo(result);
    cv::fillPoly(result, cintegrated, cv::Scalar(255));

    cintegrated.push_back(cbig[0]);
#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::Mat cont_final = cv::Mat::zeros(rfg.size(), CV_8UC3);
    cv::drawContours(cont_final, cintegrated, -1, cv::Scalar(0,0,255));
    for(uint i=0; i<nps_big.size(); ++i) {
        cv::circle(cont_final, nps_big[i], 3, cv::Scalar(0,255,0));
        cv::circle(cont_final, nps_near[i], 3, cv::Scalar(0,255,255));
    }
    cv::imshow("Contours Final", cont_final);
#endif

    getBoundingBox(cintegrated, result_rect);

    result.copyTo(final_mask);
    getFinalContour(final_mask);

    return final_mask;

}

void extendedContour::findExtremeIndexes() {
    uint i, size = contour.size();
    int x = contour[0].x, y = contour[0].y, minx = x, miny = y, maxx = x, maxy = y,
        i_minx = 0, i_miny = 0, i_maxx = 0, i_maxy = 0;
    for(i=1; i<size; ++i) {
        x = contour[i].x;
        y = contour[i].y;
        if(x < minx) {
            minx = x;
            i_minx = i;
        }
        if(x > maxx) {
            maxx = x;
            i_maxx = i;
        }
        if(y < miny) {
            miny = y;
            i_miny = i;
        }
        if(y > maxy) {
            maxy = y;
            i_maxy = i;
        }
    }
}

void extendedContour::setHIntervals(cv::Mat &labels, int label, int x, int y, int w,
                                    std::vector<cv::Point2i> &segs) {
    int i, l_int, x2 = x + w - 1, x_int;
    bool in_process = false;
    for(i=x; i<=x2; ++i) {
        if(labels.at<int>(y, i) == label) {
            if(!in_process) { //Interval starting
                in_process = true;
                x_int = i;
                l_int = 1;
            } else { //Interval continuing
                l_int++;
            }
        } else {
            if(in_process) { //Interval finishing
                in_process = false;
                segs.push_back(cv::Point(x_int, l_int));
            }
        }
    }

    if(i == x2) { //Close final interval if reached the end
        segs.push_back(cv::Point(x_int, l_int));
    }

    if(segs.size()!=0 && segs[0].x == x) { //Extend first to the left if starts at border
        int x_ext = segs[0].x, l_ext = segs[0].y;
        for(i=x_ext-1;i>=0;i--) {
            if(labels.at<int>(y, i) == label) {
                x_ext--;
                l_ext++;
            } else {
                segs[0].x = x_ext;
                segs[0].y = l_ext;
                break;
            }
        }
        if(i==0) {
            segs[0].x = x_ext;
            segs[0].y = l_ext;
        }
    }
}

void extendedContour::setVIntervals(cv::Mat &labels, int label, int x, int y, int h,
                                    std::vector<cv::Point2i> &segs) {
    int i, l_int, y2 = y + h - 1, y_int;
    bool in_process = false;
    for(i=y; i<=y2; ++i) {
        if(labels.at<int>(i, x) == label) {
            if(!in_process) { //Interval starting
                in_process = true;
                y_int = i;
                l_int = 1;
            } else { //Interval continuing
                l_int++;
            }
        } else {
            if(in_process) { //Interval finishing
                in_process = false;
                segs.push_back(cv::Point(y_int, l_int));
            }
        }
    }

    if(i == y2) { //Close final interval if reached the end
        segs.push_back(cv::Point(y_int, l_int));
    }

    if(segs.size()!=0 && segs[0].x == x) { //Extend first up if starts at border
        int y_ext = segs[0].x, l_ext = segs[0].y;
        for(i=y_ext-1;i>=0;i--) {
            if(labels.at<int>(i, x) == label) {
                y_ext--;
                l_ext++;
            } else {
                segs[0].x = y_ext;
                segs[0].y = l_ext;
                break;
            }
        }
        if(i==0) {
            segs[0].x = y_ext;
            segs[0].y = l_ext;
        }
    }
}

//It operates with square distance
float extendedContour::getMinimalDistanceToContourSimple(extendedContour &c, cv::Point &own, cv::Point &other) {
    uint i, j, own_size = contour.size(), c_size = c.contour.size();
    float d, dx, dy, min_dist = FLT_MAX;
    cv::Point near_own, near_other;
    int i_own, i_other;
    std::vector<cv::Point> &cc = c.contour;
    for(i=0; i<own_size; ++i) {
        cv::Point &pown = contour[i];
        for(j=0; j<c_size; ++j) {
            cv::Point &pcont = cc[j];
            dx = pown.x - pcont.x;
            dy = pown.y - pcont.y;
            d = dx*dx + dy*dy;
            if(d < min_dist) {
                min_dist = d;
                own.x = pown.x; own.y = pown.y; i_own = i;
                other.x = pcont.x; other.y = pcont.y; i_other = j;
            }
        }
    }

    return min_dist;
}

void extendedContour::getBoundingBox(std::vector<std::vector<cv::Point> > contours, cv::Rect &result) {
    int csize = contours.size(), psize, i, j, x, y, x1 = INT_MAX, x2 = 0, y1 = INT_MAX, y2 = 0;
    if(csize == 0) {
        result.x = result.y = result.width = result.height = 0;
        return;
    }
    for(i=0; i<csize; ++i) {
        std::vector<cv::Point> &points = contours[i];
        psize = points.size();
        for(j=0; j<psize; ++j) {
            x = points[j].x;
            y = points[j].y;
            if(x < x1) x1 = x;
            if(x > x2) x2 = x;
            if(y < y1) y1 = y;
            if(y > y2) y2 = y;
        }
    }

    result.x = x1;
    result.y = y1;
    result.width = x2 - x1 + 1;
    result.height = y2 - y1 + 1;
}

void extendedContour::getFinalContour(cv::Mat &mask) {
    uint size = nps_big.size();
    for(uint i=0; i<size; ++i)
        cv::line(mask, nps_big[i], nps_near[i], cv::Scalar(255), 5);
    std::vector<std::vector<cv::Point> > ccfinal;
    std::vector<cv::Vec4i> h;
    cv::findContours(mask, ccfinal, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
    size = ccfinal.size();
    int ind, area, max_size = 0;
    for(uint i=0; i<size; ++i) {
        area = cv::contourArea(ccfinal[i]);
        if(area > max_size) {
            ind = i;
            max_size = area;
        }
    }
    cfinal = ccfinal[ind];
}

extendedTrackedContours::extendedTrackedContours() {}

void extendedTrackedContours::addContour(int frame, extendedContour e) {
    contours[frame] = e;
}

