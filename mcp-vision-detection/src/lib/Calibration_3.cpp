#include "Calibration_3.h"

Calibration_3::Calibration_3(){

}

//If only cut boxes remaining
bool Calibration_3::cutBBoxes(std::map<int, cv::Rect> &bboxes) {
    if(bboxes.size()==0)
        return true;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit)
       if(bit->second.y != 0)
           return false;
    return true;
}

//Suppress uable bboxes based on normal shape features from markers
void Calibration_3::suppressUnviable(std::map<int, cv::Rect> &bboxes, int calibImgW, int calibImgH, cv::Mat &curImgMask) {
    int i, j, ilim, jlim, step = curImgMask.step;
    uchar *data = curImgMask.data;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    float times_diff = 0.2,
          min_p_im_w = 0.0226563,
          max_p_im_w = 0.0539063,
          min_p_bb = 1.60465,
          max_p_bb = 3.41176,
          min_c = 0.636842,
          max_c = 1,
          diff_p_im_w = fabs(max_p_im_w-min_p_im_w),
          diff_p_bb   = fabs(max_p_bb-min_p_bb),
          diff_c      = fabs(max_c-min_c);
    min_p_im_w -= times_diff*diff_p_im_w;
    max_p_im_w += times_diff*diff_p_im_w;
    min_p_bb -= times_diff*diff_p_bb;
    max_p_bb += times_diff*diff_p_bb;
    min_c -= times_diff*diff_c;
    max_c += times_diff*diff_c;

#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "p_im_w: [" << min_p_im_w << "; " << max_p_im_w << "]" << std::endl;
    std::cout << "p_bb: [" << min_p_bb << "; " << max_p_bb << "]" << std::endl;
    std::cout << "compactness: [" << min_c << "; " << max_c << "]" << std::endl;
#endif
    std::vector<int> to_suppress;

    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0) //Do not discard those cut
            continue;
        ilim = bb.y + bb.height - 1;
        jlim = bb.x + bb.width - 1;

        //Proportion to image width:
        float p_im_w = bb.width/(float)calibImgW;

        if(p_im_w < min_p_im_w || p_im_w > max_p_im_w) {
            to_suppress.push_back(bit->first);
            continue;
        }

        float p_bb = bb.width/(float)bb.height;
        if(p_bb < min_p_bb || p_bb > max_p_bb) {
            to_suppress.push_back(bit->first);
            continue;
        }

        //Pixel stats (num pixels, compatness):
        int num = 0;
        for(i=bb.y; i<ilim; ++i)
            for(j=bb.x; j<jlim; ++j)
                if(data[i*step + j] == 255) ++num;
        float compactness = num/(float)(bb.width*bb.height);
        if(compactness < min_c || compactness > max_c) {
            to_suppress.push_back(bit->first);
            continue;
        }
    }
    //Suppress
    for(uint i=0; i<to_suppress.size();++i)
        bboxes.erase(to_suppress[i]);
}

//Establish the correction factor for new algorithm iteration, based on lowest bound of remaining bboxes
int Calibration_3::getLowestFromBBoxes(std::map<int, cv::Rect> &bboxes) {
    if(bboxes.size()==0)
        return 0;
    int correction = 0, cur;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        cur = bb.y + bb.height - 1;
        if(cur > correction)
            correction = cur;
    }
    return correction > 0 ? correction + 3 : correction; //To leave lower border
}

//Establish the correction factor for new algorithm iteration, based on lowest bound of probably cut bboxes
int Calibration_3::getCorrectionToCutBoxes(std::map<int, cv::Rect> &bboxes) {
    int correction= 0, cur;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0) { //Could be cut
            cur = bb.y + bb.height - 1;
            if(cur > correction)
                correction = cur;
        }
    }
    return correction > 0 ? correction + 3 : correction; //To leave lower border
}

//Get potential marker position 5, using previously detected markers 1, 2, and 3.
int Calibration_3::getViableCentralBBox(std::map<int, cv::Rect> &bboxes, int y_start,
                         cv::Point2i &ml, cv::Point2i &mc, cv::Point2i &mr, cv::Mat &curImgMask) {
    int index = -1, cur, y, nearest = curImgMask.cols;
    float d, m, n;
    bool singular = ml.y == mr.y ? true : false;
    if(!singular) {
        m = (ml.x - mr.x)/(float)(mr.y - ml.y); //perpendicular: opposite reciprocal: -1/m
        n = mc.y - m*mc.x;
    }
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0) //Ignore those that might be cut
            continue;
        cur = bb.x+bb.width/2;
        if(cur > ml.x && cur < mr.x) { //Viable: lies between left and right markers
            if(singular) { //Singular case: perpendicular is 90° --> just compare distance to central x
                d = fabs(mc.x - cur);
                if(d < nearest) {
                    nearest = d;
                    index = bit->first;
                }
            } else {
                y = bb.y + bb.height/2 + y_start;
                d = fabs((y - n)/m - cur); //Distance to perpendicular line to ml<->mr, with incidence in mc
                if(d < nearest) {
                    nearest = d;
                    index = bit->first;
                }
            }
        }
    }
    return index;
}

//Get potential marker position 3, using previously detected marker 2 (it also works for detecting position 6, from marker 5).
int Calibration_3::getViableBBoxNearestToMarkerLeft(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
    int lindex = -1, nearest = curImgMask.cols;
    float d;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0)
            continue;
        if(bb.x+bb.width-1 < marker.x) { //Viable
            d = marker.x - bb.x+bb.width-1;
            if(d < nearest) {
                nearest = d;
                lindex = bit->first;
            }
        }
    }
    return lindex;
}

int Calibration_3::getViableBBoxHorizontalmostToMarkerRight(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
    int rindex = -1;
    float d, min_d = 360;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0)
            continue;
        if(bb.x > marker.x) { //Viable
            d = fabs(atan2(fabs(marker.y - bb.y), bb.x - marker.x))*180/M_PI;
            std::cout << "BB ID: " << bit->first << ": angle --> " << d << std::endl;
            if(d < min_d) {
                min_d = d;
                rindex = bit->first;
            }
        }
    }
    return rindex;
}

int Calibration_3::getViableBBoxHorizontalmostToMarkerLeft(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
    int rindex = -1;
    float d, min_d = 360;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0)
            continue;
        if(bb.x < marker.x) { //Viable
            d = fabs(atan2(fabs(marker.y - bb.y), marker.x - bb.x))*180/M_PI;
            if(d < min_d) {
                min_d = d;
                rindex = bit->first;
            }
        }
    }
    return rindex;
}


//Get potential marker position 1, using previously detected marker 2 (it also works for detecting position 4, from marker 5).
int Calibration_3::getViableBBoxNearestToMarkerRight(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
    int rindex = -1, nearest = curImgMask.cols;
    float d;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0)
            continue;
        if(bb.x > marker.x) { //Viable
            d = bb.x - marker.x;
            if(d < nearest) {
                nearest = d;
                rindex = bit->first;
            }
        }
    }
    return rindex;
}

//Gets centroid from fg mask inside a rect bbox
cv::Point2i Calibration_3::getMarker(cv::Rect &r, cv::Mat &curImgMask) {
    double x=0, y=0;
    int i, j,
        ilim = r.y + r.height - 1,
        jlim = r.x + r.width - 1,
        step = curImgMask.step, num = 0;
    uchar *data = curImgMask.data;

    for(i=r.y; i<ilim; ++i)
        for(j=r.x; j<jlim; ++j)
            if(data[i*step + j] == 255) {
                ++num;
                y+=i;
                x+=j;
            }
    return cv::Point2i(rint(x/num),rint(y/num));
}

//Gets lowest bbox position and bbox index
int Calibration_3::lowestMarker(std::map<int, cv::Rect> &bboxes, int &yl_index) {
    int cur, lowest = 0;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        cur = bb.y + bb.height - 1;
        if(cur > lowest) {
            lowest = cur;
            yl_index = bit->first;
        }
    }
    return lowest;
}

//Calculates and prints stats for bboxes to determine viability parameters
void Calibration_3::candidateStats(std::map<int, cv::Rect> &bboxes, int calibImgW, int calibImgH, cv::Mat &curImgMask) {
    if(bboxes.size()==0)
        return;
    int i, j, ilim, jlim, step = curImgMask.step;
    uchar *data = curImgMask.data;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();

    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        ilim = bb.y + bb.height - 1;
        jlim = bb.x + bb.width - 1;

        std::cout << "BBox " << bit->first
                  << " (x,y,w,h): (" << bb.x << ", " << bb.y
                  << ", " << bb.width << ", " << bb.height << ")" << std::endl;
        //Proportion to image width:
        std::cout << "Proportion to image width:" << bb.width/(float)calibImgW << std::endl;
        std::cout << "Proportion of bbox w/h:" << bb.width/(float)bb.height << std::endl;

        //Pixel stats (num pixels, compatness):
        int num = 0;
        for(i=bb.y; i<ilim; ++i)
            for(j=bb.x; j<jlim; ++j)
                if(data[i*step + j] == 255) ++num;
        std::cout << "Proportion to bbox pixels num/(w*h):" << num/(float)(bb.width*bb.height) << std::endl;

    }

}

//Draw bboxes and id (for debugging)
void Calibration_3::paintRectangles(cv::Mat &img, std::map<int, cv::Rect> &bboxes, cv::Scalar &color) {
    std::map<int, cv::Rect>::iterator it, it_end = bboxes.end();
    for(it = bboxes.begin(); it != it_end; it++) {
        cv::rectangle(img, it->second, color, 1);
        //Draw text
        cv::putText(img, std::to_string(it->first),
                    cv::Point(it->second.x+it->second.width+2,it->second.y+it->second.height/2), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    cv::Scalar(0,0,255));

    }
}

//Gets bboxes from connected components image
void Calibration_3::getBlobs(cv::Mat &connected_components, std::map<int, cv::Rect> &bboxes) {
    int r = connected_components.rows, c = connected_components.cols;
    int label, x, y;
    bboxes.clear();
    for(int j=0; j<r; ++j)
        for(int i=0; i<c; ++i) {
            label = connected_components.at<int>(j, i);
            if(label > 0) {
                if(bboxes.count(label) == 0) { //New label
                    cv::Rect r(i,j,1,1);
                    bboxes[label] = r;
                } else { //Update rect
                    cv::Rect &r = bboxes[label];
                    x = r.x + r.width  - 1;
                    y = r.y + r.height - 1;
                    if(i < r.x) r.x = i;
                    if(i > x) x = i;
                    if(j < r.y) r.y = j;
                    if(j > y) y = j;
                    r.width = x - r.x + 1;
                    r.height = y - r.y + 1;
                }
            }
        }
}

//Obtains foreground mask from initial fg mean and bg mean, according to proximity of each pixel.
//It also calculates the new mean fg and bg from actual fg and bg pixels, and returns pixel counts.
cv::Mat Calibration_3::FGMaskFromImage(const cv::Mat &image,
                        cv::Scalar &fg, cv::Scalar &bg,
                        cv::Scalar &bg_adj, cv::Scalar &fg_adj, int &count, int &count2) {
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(0));
    uchar *data = image.data, *mdata = mask.data;
    int i, k, rows_cols=image.rows*image.cols;
    float B, G, R, db, df,
          bg_accum_B = 0.0, bg_accum_G = 0.0, bg_accum_R = 0.0,
          fg_accum_B = 0.0, fg_accum_G = 0.0, fg_accum_R = 0.0,
          bgB = bg.val[0],
          bgG = bg.val[1],
          bgR = bg.val[2],
          fgB = fg.val[0],
          fgG = fg.val[1],
          fgR = fg.val[2];
    count  = 0;
    count2 = 0;
    for(i=0, k=0; i<rows_cols; ++i, k+=3) {
        B = data[k];
        G = data[k+1];
        R = data[k+2];
        db = fabs(B-bgB)+fabs(G-bgG)+fabs(R-bgR);
        df = fabs(B-fgB)+fabs(G-fgG)+fabs(R-fgR);
        if(df < db) {
            mdata[i] = 255;
            ++count2;
            fg_accum_B += B;
            fg_accum_G += G;
            fg_accum_R += R;
        } else {
            ++count;
            bg_accum_B += B;
            bg_accum_G += G;
            bg_accum_R += R;
        }
    }
    if(count == 0)
        bg_adj = cv::Scalar(-1,-1,-1);
    else
        bg_adj = cv::Scalar(bg_accum_B/count,bg_accum_G/count, bg_accum_R/count);

    if(count2 == 0)
        fg_adj = cv::Scalar(-1,-1,-1);
    else
        fg_adj = cv::Scalar(fg_accum_B/count2,fg_accum_G/count2, fg_accum_R/count2);
    return mask;
}

//Masked
//Obtains foreground mask from initial fg mean and bg mean, according to proximity of each pixel.
//It also calculates the new mean fg and bg from actual fg and bg pixels, and returns pixel counts.
cv::Mat Calibration_3::FGMaskFromImage_masked(const cv::Mat &image,
                        cv::Scalar &fg, cv::Scalar &bg,
                        cv::Scalar &bg_adj, cv::Scalar &fg_adj, int &count, int &count2, cv::Mat &gmask) {
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(0));
    uchar *data = image.data, *mdata = mask.data, *gmdata = gmask.data;
    int i, k, rows_cols=image.rows*image.cols;
    float B, G, R, db, df,
          bg_accum_B = 0.0, bg_accum_G = 0.0, bg_accum_R = 0.0,
          fg_accum_B = 0.0, fg_accum_G = 0.0, fg_accum_R = 0.0,
          bgB = bg.val[0],
          bgG = bg.val[1],
          bgR = bg.val[2],
          fgB = fg.val[0],
          fgG = fg.val[1],
          fgR = fg.val[2];
    count  = 0;
    count2 = 0;
    for(i=0, k=0; i<rows_cols; ++i, k+=3) {
        if(gmdata[i]) {
            B = data[k];
            G = data[k+1];
            R = data[k+2];
            db = fabs(B-bgB)+fabs(G-bgG)+fabs(R-bgR);
            df = fabs(B-fgB)+fabs(G-fgG)+fabs(R-fgR);
            if(df < db) {
                mdata[i] = 255;
                ++count2;
                fg_accum_B += B;
                fg_accum_G += G;
                fg_accum_R += R;
            } else {
                ++count;
                bg_accum_B += B;
                bg_accum_G += G;
                bg_accum_R += R;
            }
        }
    }
    if(count == 0)
        bg_adj = cv::Scalar(-1,-1,-1);
    else
        bg_adj = cv::Scalar(bg_accum_B/count,bg_accum_G/count, bg_accum_R/count);

    if(count2 == 0)
        fg_adj = cv::Scalar(-1,-1,-1);
    else
        fg_adj = cv::Scalar(fg_accum_B/count2,fg_accum_G/count2, fg_accum_R/count2);
    return mask;
}



cv::Mat Calibration_3::completeNearlyConnected(cv::Mat &img, cv::Mat initial_mask, int T) {
    cv::Mat glob, checked;
    initial_mask.copyTo(glob);
    initial_mask.copyTo(checked); //Marks if the pixel is already checked
    int rows = img.rows, cols = img.cols, step = img.step, mstep = checked.step;
    uchar *dchecked = checked.data, *dimg = img.data, *dglob = glob.data;
    int num_checked, B, G, R, BB, GG, RR, d;
    bool unchecked_back, added;
    for(int i=rows-1; i>=0; --i) { //Starts from bottom
        unchecked_back = false; //Flag to check
        for(int j=0; j<cols; ++j) {
            added = false;
            if(!dchecked[i*mstep + j]) { //If not checked, check V4 neighbors for short distance to be added
                num_checked = 0;
                B = dimg[i*step + 3*j];
                G = dimg[i*step + 3*j + 1];
                R = dimg[i*step + 3*j + 2];

                //Lower:
                if(i<rows-1) {
                    if(dchecked[(i+1)*mstep + j]) {
                        ++num_checked;
                        if(dglob[(i+1)*mstep + j]) { //Candidate for analysis
                            BB = dimg[(i+1)*step + 3*j];
                            GG = dimg[(i+1)*step + 3*j + 1];
                            RR = dimg[(i+1)*step + 3*j + 2];
                            d = abs(B-BB) + abs(G-GG) + abs(R-RR);
                            if(d < T) { //It is close to a
                                added = true;
                                dchecked[i*mstep + j] = 255;
                                dglob[i*mstep + j] = 255;
                            }
                        }
                    }
                }

                //Left:
                if(j>0) {
                    if(dchecked[i*mstep + j - 1]) {
                        ++num_checked;
                        if(!added && dglob[i*mstep + j - 1]) { //Candidate for analysis
                            BB = dimg[i*step + 3*(j-1)];
                            GG = dimg[i*step + 3*(j-1) + 1];
                            RR = dimg[i*step + 3*(j-1) + 2];
                            d = abs(B-BB) + abs(G-GG) + abs(R-RR);
                            if(d < T) { //It is close to a
                                added = true;
                                dchecked[i*mstep + j] = 255;
                                dglob[i*mstep + j] = 255;
                            }
                        }
                    }
                }

                //Right:
                if(j<cols-1) {
                    if(dchecked[i*mstep + j + 1]) {
                        ++num_checked;
                        if(!added && dglob[i*mstep + j + 1]) { //Candidate for analysis
                            BB = dimg[i*step + 3*(j+1)];
                            GG = dimg[i*step + 3*(j+1) + 1];
                            RR = dimg[i*step + 3*(j+1) + 2];
                            d = abs(B-BB) + abs(G-GG) + abs(R-RR);
                            if(d < T) { //It is close to a
                                added = true;
                                dchecked[i*mstep + j] = 255;
                                dglob[i*mstep + j] = 255;
                            }
                        }
                    }
                }

                //Top:
                if(i>0) {
                    if(dchecked[(i-1)*mstep + j]) {
                        ++num_checked;
                        if(!added && dglob[(i-1)*mstep + j]) { //Candidate for analysis
                            BB = dimg[(i-1)*step + 3*j];
                            GG = dimg[(i-1)*step + 3*j + 1];
                            RR = dimg[(i-1)*step + 3*j + 2];
                            d = abs(B-BB) + abs(G-GG) + abs(R-RR);
                            if(d < T) { //It is close to a
                                added = true;
                                dchecked[i*mstep + j] = 255;
                                dglob[i*mstep + j] = 255;
                            }
                        }
                    }
                }

                //If the connected point was added
                if(added && i<rows-1 && !dchecked[(i+1)*mstep + j])
                    unchecked_back = true;

                //If it was finally not added, but found checked it is considered as checked
                if(!added && num_checked > 0)
                    dchecked[i*mstep + j] = 255;
            }
        }
        if(unchecked_back)
            i += 2; //goes back two rows (really one row, as for decrements one) to add elements which might be added too
    }


    return glob;
}

cv::Mat Calibration_3::setMaskFromColor(cv::Mat &img, cv::Scalar mask_color, double T) {
    cv::Mat result;
    double B = mask_color.val[0], G = mask_color.val[1], R = mask_color.val[2];
    int lB = B - T < 0 ? 0 : B - T,
          lG = G - T < 0 ? 0 : G - T,
          lR = R - T < 0 ? 0 : R - T,
          uB = B + T > 255 ? 255 : B + T,
          uG = G + T > 255 ? 255 : G + T,
          uR = R + T > 255 ? 255 : R + T;
    std::cout << (int)lB << "; " << (int)lG << "; " << (int)lR << std::endl;
    std::cout << (int)uB << "; " << (int)uG << "; " << (int)uR << std::endl;
    cv::inRange(img, cv::Scalar(lB, lG, lR), cv::Scalar(uB, uG, uR), result);
    return result;
}

//Finds histogram peak and adjusts it interpolating on neighborhood (2*nsize)x(2*nsize)x(2*nsize)
double Calibration_3::histoFindAdjustedMax(const cv::Mat &histo, int bins, int nsize, cv::Scalar &best_max) {
    int bsize = 256/bins, bb=bins*bins, bB, bG, bR;
    float hval, best = -1;
    int bi, bj, bk;
    //Get max frequency bin:
    for(int i=0, mB=bsize/2; i<bins; ++i, mB+=bsize)
        for(int j=0, mG=bsize/2; j<bins; ++j, mG+=bsize)
            for(int k=0, mR=bsize/2; k<bins; ++k, mR+=bsize) {
                hval = histo.at<float>(i*bb + j*bins + k);
                if(hval > best) {
                    best = hval;
                    bB = mB; bG = mG; bR = mR;
                    bi = i; bj = j; bk = k;
                }
            }
    //Correction of peak weighted by frequency in a (2*nsize+1) window:
    double accumB=0.0, accumG=0.0, accumR=0.0, waccum = 0.0;
    int nbsize = nsize*bsize;
    for(int i=bi-nsize, mB=bB-nbsize; i<=bi+nsize; ++i, mB+=bsize)
        for(int j=bj-nsize, mG=bG-nbsize; j<=bj+nsize; ++j, mG+=bsize)
            for(int k=bk-nsize, mR=nbsize/2; k<=bk+nsize; ++k, mR+=bsize)
                if(i>=0 && j>=0 && k>=0 && i<bins && j<bins && k<bins) {
                    hval = histo.at<float>(i*bb + j*bins + k);
                    waccum += hval;
                    accumB += hval*mB;
                    accumG += hval*mG;
                    accumR += hval*mR;
                }

    //Return weighted mean:
    best_max = cv::Scalar(accumB/waccum, accumG/waccum, accumR/waccum);
    return waccum;
}

void Calibration_3::histogram3D_masked(const cv::Mat &im, int bins, cv::Mat *histo, cv::Mat &mask) {
    int i, j, k, bsize = 256/bins, bb = bins*bins, bbb = bb*bins,
        rows = im.rows, cols = im.cols, step = im.step, mstep = mask.step;
    uchar *data = im.data, *dmask = mask.data;
    int iC1, iC2, iC3;
    int count = 0;
    histo->create(1, bbb, CV_32FC1);
    for(i=0; i<bbb; ++i)
         histo->at<float>(i) = 0.0;
    for(i=0; i<rows; ++i)
        for(j=0, k=0; j<cols; ++j, k+=3)
            if(dmask[i*mstep + j]) {
                iC1 = ((int)data[i*step + k    ])/bsize;
                iC2 = ((int)data[i*step + k + 1])/bsize;
                iC3 = ((int)data[i*step + k + 2])/bsize;
                histo->at<float>(iC1*bb + iC2*bins + iC3) += 1.0;
                ++count;
            }

    if(count > 0)
        for(i=0; i<bbb; ++i)
             histo->at<float>(i) /= count;
}

//Calculates the 3D histogram on 3-channel color space
void Calibration_3::histogram3D(const cv::Mat &im, int bins, cv::Mat *histo) {
    int i, j, k, bsize = 256/bins, bb = bins*bins, bbb = bb*bins,
        rows = im.rows, cols = im.cols, step = im.step;
    uchar *data = im.data;
    int iC1, iC2, iC3;
    int count = 0;
    histo->create(1, bbb, CV_32FC1);
    for(i=0; i<bbb; ++i)
         histo->at<float>(i) = 0.0;
    for(i=0; i<rows; ++i)
        for(j=0, k=0; j<cols; ++j, k+=3) {
            iC1 = ((int)data[i*step + k    ])/bsize;
            iC2 = ((int)data[i*step + k + 1])/bsize;
            iC3 = ((int)data[i*step + k + 2])/bsize;
            histo->at<float>(iC1*bb + iC2*bins + iC3) += 1.0;
            ++count;
        }
    if(count > 0)
        for(i=0; i<bbb; ++i)
             histo->at<float>(i) /= count;
}

//Set scene points as star configuration (5 at center, and 8 markers at 200cm distances, 45° distance each
void Calibration_3::setScenePoints(std::vector<cv::Point2f> &scenePoints) {
    //The nine scene points
    scenePoints.resize(9);
    cv::Point2f p;
    p.x = 141.421356237; p.y = 141.421356237;
    scenePoints[0] = p; //Position 1
    p.x =    0; p.y = 200;
    scenePoints[1] = p; //Position 2
    p.x =  -141.421356237; p.y = 141.421356237;
    scenePoints[2] = p; //Position 3
    p.x = 200; p.y = 0;
    scenePoints[3] = p; //Position 4
    p.x =    0; p.y = 0;
    scenePoints[4] = p; //Position 5
    p.x =  -200; p.y = 0;
    scenePoints[5] = p; //Position 6
    p.x = 141.421356237; p.y = -141.421356237;
    scenePoints[6] = p; //Position 7
    p.x =    0; p.y = -200;
    scenePoints[7] = p; //Position 8
    p.x =  -141.421356237; p.y = -141.421356237;
    scenePoints[8] = p; //Position 9
}


//Transforms scene coordinate to image
cv::Point2i Calibration_3::transform(cv::Point2f p, cv::Mat &H) {
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0,0) = p.x;
    pin.at<double>(1,0) = p.y;
    pin.at<double>(2,0) = 1;

    cv::Mat pout = H*pin;

    return cv::Point2i(rint(pout.at<double>(0,0)/pout.at<double>(2,0)),
                       rint(pout.at<double>(1,0)/pout.at<double>(2,0)));
}

//Draws a 15 cm square on image
void Calibration_3::drawRectangle(cv::Mat &img, cv::Point2f &p, cv::Mat &H) {
    float hside = 7.5;
    cv::line(img, transform(cv::Point2f(p.x-hside, p.y-hside), H),
             transform(cv::Point2f(p.x+hside, p.y-hside), H), cv::Scalar(0,255,255));
    cv::line(img, transform(cv::Point2f(p.x+hside, p.y-hside), H),
             transform(cv::Point2f(p.x+hside, p.y+hside), H), cv::Scalar(0,255,255));
    cv::line(img, transform(cv::Point2f(p.x+hside, p.y+hside), H),
             transform(cv::Point2f(p.x-hside, p.y+hside), H), cv::Scalar(0,255,255));
    cv::line(img, transform(cv::Point2f(p.x-hside, p.y+hside), H),
             transform(cv::Point2f(p.x-hside, p.y-hside), H), cv::Scalar(0,255,255));
}

//Get image point from scene coordinate
cv::Point2i Calibration_3::getPoint(cv::Point2f p, cv::Mat &H) {
    //Get center in image coordinates
    return transform(p, H);
}

//Draw circular marker and writes id on image
void Calibration_3::addMarker(cv::Mat &img, cv::Point2i &p, int index) {
    //Draw circle
    cv::circle(img, p, 3, cv::Scalar(0,0,255));

    //Draw text
    cv::putText(img, std::to_string(index),
                cv::Point(p.x+2,p.y), cv::FONT_HERSHEY_DUPLEX, 0.5,
                cv::Scalar(0,255,0));
}

//Adds marker with circle, 15 cm box and id, to image
void Calibration_3::addPoint(cv::Mat &img, cv::Point2f &p, cv::Mat &H, int index) {
    //Get center in image coordinates
    cv::Point2i p_im = transform(p, H);

    //Draw circle
    cv::circle(img, p_im, 3, cv::Scalar(0,0,255));

    //Draw rectangle
    drawRectangle(img, p, H);

    //Draw text
    cv::putText(img, std::to_string(index),
                cv::Point(p_im.x+2,p_im.y), cv::FONT_HERSHEY_DUPLEX, 0.5,
                cv::Scalar(0,255,0));
}


size_t Calibration_3::write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    vector<uchar> *stream = (vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);
    return count;
}

//function to retrieve the image as cv::Mat data type
cv::Mat Calibration_3::curlImg(const char *img_url, int timeout=10)
{
    vector<uchar> stream;
    CURL *curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, img_url); //the img url
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); // timeout if curl_easy hangs,
    CURLcode res = curl_easy_perform(curl); // start curl
    curl_easy_cleanup(curl); // cleanup
    return cv::imdecode(stream, -1); // 'keep-as-is'
}

namespace {
    std::size_t callback(const char* in, std::size_t size, std::size_t num, std::string* out) {
        const std::size_t totalBytes(size * num);
        out->append(in, totalBytes);
        return totalBytes;
    }
}

std::pair<int,std::string> Calibration_3::getMarksAutomatic(std::string email,std::string screenshot) {

    ///Process automatic calibration
    std::chrono::steady_clock::time_point ebegin = std::chrono::steady_clock::now();
    
    ImagemConverter test = ImagemConverter();
    std::string base64_imageFromApp = screenshot.erase(0,23);
    cv::Mat current, M, M1 = test.str2mat(base64_imageFromApp); //Carpeta parcial
    int real_w = M1.cols, real_h = M1.rows,
        calib_w = 1280, calib_h = (real_h * calib_w)/real_w; //Keeps proportion

    cv::resize(M1, M1, cv::Size(calib_w, calib_h));

#ifdef SHOW_MAIN_RESULTS
    cv::imwrite("BG Image.png", M1);
    
#endif

    //Stores bboxes
    std::map<int, cv::Rect> bboxes;
    //Stores positions 1-9 for marker centers and availability of them
    std::map<int, cv::Point2i> marker_position;
    std::map<int, bool> marker_availability;
    marker_availability[1] = false;
    marker_availability[2] = false;
    marker_availability[3] = false;
    marker_availability[4] = false;
    marker_availability[5] = false;
    marker_availability[6] = false;
    marker_availability[7] = false;
    marker_availability[8] = false;
    marker_availability[9] = false;
    cv::Scalar fg_val(255,255,255), bg_val(0,0,0), bg_adj, fg_adj;
    bool first_calib_ok = false;

    int y_start = (2*M1.rows)/3, y_end = M1.rows-1; //The same but initially with 1/3 of image from bottom
    float pct_reduction = 0.15; //15% reduction per iteration.

    //0. Set global mask
    cv::Mat M1c, M1m;
    int gksize = 7;
    int gksize_dil = 5;

    //0.1 Prepare image trying to suppress high texture
    cv::erode(M1, M1c, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(gksize_dil,gksize_dil)));
    cv::dilate(M1c, M1c, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(gksize_dil,gksize_dil)));
    cv::erode(M1c, M1c, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(gksize_dil,gksize_dil)));
    cv::dilate(M1c, M1c, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(gksize_dil,gksize_dil)));
    cv::medianBlur(M1c, M1m, gksize);
#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imwrite("Global Color Median Blur Image.png", M1m);
    
#endif

    //0.2 Get half image
    cv::Mat M1h(M1m, cv::Rect(0,M1.rows/2,M1.cols,M1.rows/2));
#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imwrite("Half Image.png", M1h);
    
#endif

    int gBGRbins = 16, gYUVbins = 16;
    cv::Mat YUV, ghistoBGR, ghistoYUV;
    histogram3D(M1h, gBGRbins, &ghistoBGR);
    cv::cvtColor(M1h, YUV, cv::COLOR_BGR2YCrCb);
    histogram3D(M1h, gBGRbins, &ghistoBGR);
    histogram3D(M1h, gYUVbins, &ghistoYUV);

    float gbest;
    cv::Scalar gbest_max;
    gbest = histoFindAdjustedMax(ghistoBGR, gBGRbins, 1, gbest_max);
#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "Best: "
              << gbest_max.val[0] << ","
              << gbest_max.val[1] << ","
              << gbest_max.val[2] << ": " << gbest << std::endl;
#endif

    //0.4 Get simple 2-KMeans (fg, bg), starting from initial bg and extreme assumed fg:
    cv::Mat ginitial_mask, ginitial_mask2;
    int gcount_bg, gcount_fg;
    ginitial_mask = FGMaskFromImage(M1h, fg_val, gbest_max, bg_adj, fg_adj, gcount_bg, gcount_fg);
    //ginitial_mask = setMaskFromColor(M1m, bg_adj, 20);


    //Another strategy: Get 3 peak UV colors and extend their probability from all descending Y.

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imwrite("GInitial Mask.png", ginitial_mask);
    
#endif

    //global_mask = completeNearlyConnected(M1m, ginitial_mask, 20);
    cv::Mat global_mask = cv::Mat::zeros(M1m.size(), CV_8UC1), partial_gmask(global_mask, cv::Rect(0,M1.rows/2,M1.cols,M1.rows/2));
    partial_gmask = 255 - ginitial_mask;

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imwrite("Global Mask.png", global_mask);
    
#endif

    cv::Mat ext_global_mask = completeNearlyConnected(M1m, global_mask, 10);

    //Check non-convex lower bound
    int first = -1, first_black = -1, last, ecols = ext_global_mask.cols,
            erows = ext_global_mask.rows, eind = (erows-1)*ext_global_mask.step;
    bool non_convex = false;
    for(int i=0; i<ecols; ++i) {
        if(ext_global_mask.data[eind + i]) {
            if(first == -1)
                first = i;
            else if(first_black >= 0) {
                non_convex = true;
            }
        } else {
            if(first >= 0) {
                if(first_black == -1)
                    first_black = i;
                if(ext_global_mask.data[eind + i - 1])
                    last = i-1;
            }
        }
    }
    std::cout << "First: " << first << std::endl;
    std::cout << "First black: " << first_black << std::endl;
    std::cout << "Last: " << last << std::endl;
    std::cout << "Non-convex: " << non_convex << std::endl;
    if(non_convex) {
        if(last < first_black) { //White till the end, after concavity
            memset(ext_global_mask.data + eind + first, 255, ecols - first);
        } else {
            memset(ext_global_mask.data + eind + first, 255, last - first + 1);
        }
    }

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imwrite("Extended Global Mask.png", ext_global_mask);
    
#endif

    std::vector<std::vector<cv::Point> > contours0;
    std::vector<cv::Vec4i> h;
    cv::findContours(ext_global_mask, contours0, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat icontours0 = cv::Mat::zeros(ext_global_mask.size(), CV_8UC1);
    int a, max_area = 0, max_ind;
    for(uint i=0; i<contours0.size(); ++i) {
        int x, y, xl = M1m.cols-1, xr = 0, yt = M1m.rows-1, yb = 0;
        std::vector<cv::Point> c = contours0[i];
        for(uint j=0; j<c.size(); ++j) {
            x = c[j].x;
            y = c[j].y;
            if(x < xl)
                xl = x;
            if(x > xr)
                xr = x;
            if(y < yt)
                yt = y;
            if(y > yb)
                yb = y;
        }
        a = (yb - yt + 1)*(xr - xl + 1);
        if(a > max_area) {
            max_area = a;
            max_ind = i;
        }
    }

    cv::Mat carpet_area0 = cv::Mat::zeros(ext_global_mask.size(), CV_8UC1),
            carpet_mask0 = cv::Mat::zeros(ext_global_mask.size(), CV_8UC1), dcarpet_mask;
    cv::drawContours(carpet_area0, contours0, max_ind, cv::Scalar(255,255,255), 1);
    cv::drawContours(carpet_mask0, contours0, max_ind, cv::Scalar(255,255,255), cv::FILLED);

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imwrite("Carpet area.png", carpet_area0);
    cv::imwrite("Carpet mask.png", carpet_mask0);
    
#endif



    //1. Cycle till first bboxes are found
    int it_count = 0;
    while(true) { //While no valid initial markers found yet
        ++it_count;
        cv::Mat M1r(M1m, cv::Rect(0, y_start, M1.cols, y_end-y_start+1));
        cv::Mat mask_r(carpet_mask0, cv::Rect(0, y_start, M1.cols, y_end-y_start+1));

#ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imwrite("Image Section.png", M1r);
        
#endif

#ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imwrite("Local Mask.png", mask_r);
        
#endif

        cv::Mat initial_mask;
        int count_bg, count_fg, min_count_fg = 10;
        if(!first_calib_ok) {
            //1.3 3D histogram to get initial distribution
            int BGRbins = 16;
            cv::Mat histoBGR;
            histogram3D_masked(M1r, BGRbins, &histoBGR, mask_r);

            //1.4 Interpolate maximum assuming is initial bg
            float best;
            cv::Scalar best_max;
            best = histoFindAdjustedMax(histoBGR, BGRbins, 1, best_max);
#ifdef SHOW_INTERMEDIATE_RESULTS
            std::cout << "Best: "
                      << best_max.val[0] << ","
                      << best_max.val[1] << ","
                      << best_max.val[2] << ": " << best << std::endl;
    #endif
            //1.5 Get simple 2-KMeans (fg, bg), starting from initial bg and extreme assumed fg:
            initial_mask = FGMaskFromImage_masked(M1r, fg_val, best_max, bg_adj, fg_adj, count_bg, count_fg, mask_r);

        } else { //Just adjust the current one:
            cv::Scalar bg_adj2, fg_adj2;
            initial_mask = FGMaskFromImage_masked(M1r, fg_adj, bg_adj, bg_adj2, fg_adj2, count_bg, count_fg, mask_r);

            if(count_bg > min_count_fg)
                bg_adj = bg_adj2;
            if(count_fg > min_count_fg)
                fg_adj = fg_adj2;
            else { //Iteration or stop criterion for too few fg pixels
                if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
                    break;
                y_end = y_start - 1;
                y_start -= pct_reduction*M1.rows;
                if(y_start<0) y_start = 0;
                    continue;
            }
        }
#ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imwrite("Initial mask.png", initial_mask);
        
        std::cout << "BG: "
                  << bg_adj.val[0] << ","
                  << bg_adj.val[1] << ","
                  << bg_adj.val[2] << std::endl;
        std::cout << "FG: "
                  << fg_adj.val[0] << ","
                  << fg_adj.val[1] << ","
                  << fg_adj.val[2] << std::endl;
#endif
        float d = fabs(bg_adj.val[0]-fg_adj.val[0])+fabs(bg_adj.val[1]-fg_adj.val[1])+fabs(bg_adj.val[2]-fg_adj.val[2]),
              two_class_thr = 100;
#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "FG<-->BG distance: " << d << std::endl;
#endif
        //Iteration or stop criterion for single class found
        if(d < two_class_thr) { //No significant fg<-->bg difference, so go for next while iteration with bigger image
            if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
                break;
            y_end = y_start - 1;
            y_start -= pct_reduction*M1.rows;
            if(y_start<0) y_start = 0;
            continue;
        }

        //If we get here, we have two significantly different classes (FG, BG).
        first_calib_ok = true;

        //1.6 Get candidates: connected components from initial mask and bounding boxes
        cv::Mat conn, im_bb;
        cv::connectedComponents(initial_mask, conn);
        getBlobs(conn, bboxes);
        M1r.copyTo(im_bb);
        cv::Scalar green(0,255,0);
        paintRectangles(im_bb, bboxes, green);

#ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imwrite("Initial bboxes.png", im_bb);
        
#endif

        //Iteration or stop criterion for no bbox found
        if(bboxes.size()==0) {
            if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
                break;
            y_end = y_start - 1;
            y_start -= pct_reduction*M1.rows;
            if(y_start<0) y_start = 0;
            continue;
        }


#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Iteration " << it_count << std::endl;
        std::cout << "All candidates: " << std::endl;
        candidateStats(bboxes, M1r.cols, M1r.rows, initial_mask);
#endif
        suppressUnviable(bboxes, M1r.cols, M1r.rows, initial_mask);
#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Viable candidates: " << std::endl;
        candidateStats(bboxes, M1r.cols, M1r.rows, initial_mask);
#endif

        if(cutBBoxes(bboxes)) {
            if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
                break;
            //Correction to cutted boxes, to enter in next iteration:
            int correction = getLowestFromBBoxes(bboxes);
            y_end = y_start - 1 + correction;
            y_start -= pct_reduction*M1.rows;
            if(y_start<0) y_start = 0;
            if(y_end>=M1.rows) y_end = M1.rows-1;
            if(bboxes.size()>0)
                bboxes.clear();
            continue;
        }


        //Set first marker position (assume lowest is marker in position 2):
        if(!marker_availability[2]) { //First set this marker
            int yl_index, y_lowest = lowestMarker(bboxes, yl_index);
            if(y_lowest > 0) {
                marker_position[2] = getMarker(bboxes[yl_index], initial_mask);
                marker_position[2].y += y_start; //Correction to global image
                marker_availability[2] = true;
                bboxes.erase(yl_index); //Delete utilized marker

                if(cutBBoxes(bboxes)) { //If only cut boxes remaining
                    if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
                        break;
                    //Correction to cutted boxes, to enter in next iteration:
                    int correction = getLowestFromBBoxes(bboxes);
                    y_end = y_start - 1 + correction;
                    y_start -= pct_reduction*M1.rows;
                    if(y_start<0) y_start = 0;
                    if(y_end>=M1.rows) y_end = M1.rows-1;
                    if(bboxes.size()>0)
                        bboxes.clear();
                    continue;
                }
            }
        }

        //Set following markers: 1 and 3
        if(marker_availability[2]) { //First set this marker
            cv::Point2i &m2 = marker_position[2];
            if(!marker_availability[1]) {
                int rindex = getViableBBoxNearestToMarkerRight(bboxes, m2, initial_mask);
                if(rindex >= 0) { //Found viable
                    marker_position[1] = getMarker(bboxes[rindex], initial_mask);
                    marker_position[1].y += y_start; //Correction to global image
                    marker_availability[1] = true;
                    bboxes.erase(rindex); //Delete utilized marker

                    if(cutBBoxes(bboxes)) { //If only cut boxes remaining
                        if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
                            break;
                        //Correction to cutted boxes, to enter in next iteration:
                        int correction = getLowestFromBBoxes(bboxes);
                        y_end = y_start - 1 + correction;
                        y_start -= pct_reduction*M1.rows;
                        if(y_start<0) y_start = 0;
                        if(y_end>=M1.rows) y_end = M1.rows-1;
                        if(bboxes.size()>0)
                            bboxes.clear();
                        continue;
                    }

                }
            }


            if(!marker_availability[3]) {
                int lindex = getViableBBoxNearestToMarkerLeft(bboxes, m2, initial_mask);
                if(lindex >= 0) { //Found viable
                    marker_position[3] = getMarker(bboxes[lindex], initial_mask);
                    marker_position[3].y += y_start; //Correction to global image
                    marker_availability[3] = true;
                    bboxes.erase(lindex); //Delete utilized marker

                    if(cutBBoxes(bboxes)) { //If only cut boxes remaining
                        if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
                            break;
                        //Correction to cutted boxes, to enter in next iteration:
                        int correction = getLowestFromBBoxes(bboxes);
                        y_end = y_start - 1 + correction;
                        y_start -= pct_reduction*M1.rows;
                        if(y_start<0) y_start = 0;
                        if(y_end>=M1.rows) y_end = M1.rows-1;
                        if(bboxes.size()>0)
                            bboxes.clear();
                        continue;
                    }

                }
            }
        }

        //Set central marker 5: between x positions of 1 and 3, and the nearest from 2
        if(marker_availability[1] && marker_availability[2] && marker_availability[3]) {
            if(!marker_availability[5]) {
                int index = getViableCentralBBox(bboxes, y_start,
                                                 marker_position[3],
                                                 marker_position[2],
                                                 marker_position[1], initial_mask);
                if(index >= 0) { //Found viable
                    marker_position[5] = getMarker(bboxes[index], initial_mask);
                    marker_position[5].y += y_start; //Correction to global image
                    marker_availability[5] = true;
                    bboxes.erase(index); //Delete utilized marker
                    //With the four markers there might be still bad calibration... prepare special window of analysis

                    //Try to get markers 4 and 6:
                    bool lok = false, rok = false;
                    //Marker 4
                    int rm = getViableBBoxHorizontalmostToMarkerRight(bboxes, marker_position[5], initial_mask);
                    if(rm >= 0 && bboxes[rm].y > 0) { //Check if there is viable marker (bbox with y>0 (non-cutted))
                        marker_position[4] = getMarker(bboxes[rm], initial_mask);
                        marker_position[4].y += y_start; //Correction to global image
                        marker_availability[4] = true;
                        bboxes.erase(index); //Delete utilized marker
                        rok = true;
                    }
                    //Marker 6
                    int lm = getViableBBoxHorizontalmostToMarkerLeft(bboxes, marker_position[5], initial_mask);
                    if(lm >= 0 && bboxes[lm].y > 0) { //Check if there is viable marker (bbox with y>0 (non-cutted))
                        marker_position[6] = getMarker(bboxes[lm], initial_mask);
                        marker_position[6].y += y_start; //Correction to global image
                        marker_availability[6] = true;
                        bboxes.erase(index); //Delete utilized marker
                        lok = true;
                    }

                    if(lok || rok) //If we find at least one more it shall be ok
                        break;
                    else {
                        //Iteration or stop criterion for no bbox or cutted box
                        int times = 3;
                        y_start = bboxes[index].y - bboxes[index].height*times;
                        y_end = bboxes[index].y + - 1 + bboxes[index].height*(times+1);
                        if(y_start<0) y_start = 0;
                        if(y_end >= M1.rows) y_end = M1.rows-1;
                        if(bboxes.size()>0)
                            bboxes.clear();
                        continue;
                    }
                }
            }
        }

        //If marker 5 present, initially tried and failed to find 4 or 6: try again with special analysis:
        if(marker_availability[5]) {
            //1. Suppress bbox representing 5.
            //2. Try to get 4 and 6 for a last try.
            break;
        }

        //Iteration or stop criterion for no bbox or cutted box
        if(y_start == 0) //No initial region candidates found, this calibration is unsuccessful.
            break;
        //Correction to cutted boxes, to enter in next iteration:
        int correction = getCorrectionToCutBoxes(bboxes);
        y_end = y_start - 1 + correction;
        y_start -= pct_reduction*M1.rows;
        if(y_start<0) y_start = 0;
        if(bboxes.size()>0)
            bboxes.clear();

        //Print markers:
        std::map<int, cv::Point2i>::iterator mit, mend = marker_position.end();
        for(mit = marker_position.begin(); mit != mend; mit++) {
            cv::Point2i &p = mit->second;
            std::cout << mit->first << ": (" << p.x << "; " << p.y << ")" << std::endl;
        }
    }

    cv::Mat carpet_area;
    M1.copyTo(carpet_area);

    //Print final markers:
    cv::Mat mout;
    M1.copyTo(mout);
    std::cout << "Final markers: " << std::endl;
    std::map<int, cv::Point2i>::iterator mit, mend = marker_position.end();
    for(mit = marker_position.begin(); mit != mend; mit++) {
        cv::Point2i &p = mit->second;
        std::cout << mit->first << ": (" << p.x << "; " << p.y << ")" << std::endl;
        addMarker(mout, mit->second, mit->first);
    }
#ifdef SHOW_MAIN_RESULTS
        cv::imwrite("Available markers.png", mout);
        
#endif


    //Here we expect 4 points (for the moment...).
    //Calibrate scene
    std::vector<cv::Point2f> scenePoints;
    setScenePoints(scenePoints);

    std::vector<cv::Point2f> calibScenePoints, imagePoints;
    calibScenePoints.push_back(scenePoints[0]);
    imagePoints.push_back(cv::Point2f(marker_position[1].x, marker_position[1].y));
    calibScenePoints.push_back(scenePoints[1]);
    imagePoints.push_back(cv::Point2f(marker_position[2].x, marker_position[2].y));
    calibScenePoints.push_back(scenePoints[2]);
    imagePoints.push_back(cv::Point2f(marker_position[3].x, marker_position[3].y));
    if(marker_availability[4]) {
        calibScenePoints.push_back(scenePoints[3]);
        imagePoints.push_back(cv::Point2f(marker_position[4].x, marker_position[4].y));
    }
    calibScenePoints.push_back(scenePoints[4]);
    imagePoints.push_back(cv::Point2f(marker_position[5].x, marker_position[5].y));
    if(marker_availability[6]) {
        calibScenePoints.push_back(scenePoints[5]);
        imagePoints.push_back(cv::Point2f(marker_position[6].x, marker_position[6].y));
    }
    if(marker_availability[7]) {
        calibScenePoints.push_back(scenePoints[6]);
        imagePoints.push_back(cv::Point2f(marker_position[7].x, marker_position[7].y));
    }
    if(marker_availability[8]) {
        calibScenePoints.push_back(scenePoints[7]);
        imagePoints.push_back(cv::Point2f(marker_position[8].x, marker_position[8].y));
    }
    if(marker_availability[9]) {
        calibScenePoints.push_back(scenePoints[8]);
        imagePoints.push_back(cv::Point2f(marker_position[9].x, marker_position[9].y));
    }

//  cv::Mat H = cv::findHomography(calibScenePoints, imagePoints);
    cv::Mat H = cv::findHomography(calibScenePoints, imagePoints, cv::RANSAC, 5);
    cv::Mat fout;
    M1.copyTo(fout);
    std::map<int, std::vector<cv::Point2i> > objectiveImPos;
    for(int i=1; i<= 9; i++) {
        addPoint(fout, scenePoints[i-1], H, i);
    }
    std::string output = "{";
    output += "\"state\" : \"OK!\" ,";
    output += "\"backgroundImage\":\""+ base64_imageFromApp +"\",";
    output += "\"Wcalib\":"+ std::to_string(calib_w) +",";
    output += "\"Hcalib\":"+ std::to_string(calib_h) +",";
    output += "\"Width\":" + std::to_string(real_w) + ",";
    output += "\"Height\":" + std::to_string(real_h) + ",";
    output += "\"Homography\": [";
    for(int i=0; i<= 8; i++) {
	output += std::to_string(H.at<double>(i)) + ",";
    }
    output.pop_back();
    output += " ] , ";
    output += "\"Floor\": [";
    for(int i=1; i<= 9; i++) {
    	output += "{\"id\":"+ std::to_string(i) + ", \"vertices\": [ ";
        cv::Point2f sp = scenePoints[i-1];
        std::vector<cv::Point2i> square;
        float marker_size = 7.5;
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y+marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y+marker_size), H));
        objectiveImPos[i] = square;
        
        //std::cout<<square[0].x<<std::endl;
        output += "{\"x\":"+ std::to_string((square[0].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[0].y*real_h)/calib_h)+"},"+"{\"x\":"+ std::to_string((square[1].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[1].y*real_h)/calib_h)+"},"+"{\"x\":"+ std::to_string((square[2].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[2].y*real_h)/calib_h)+"},"+"{\"x\":"+ std::to_string((square[3].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[3].y*real_h)/calib_h)+"}";
        output += " ] },";
    }
    output.pop_back();
    output += " ] , ";

    output += "\"Marks\": [";
    std::map<int, cv::Point2i>::iterator mit2, mend2 = marker_position.end();
    for(mit2 = marker_position.begin(); mit2 != mend2; mit2++) {
        cv::Point2i &p = mit2->second;
        output += "{\"id\":"+ std::to_string(mit2->first) + ", \"vertices\": ";
        output += "{\"x\":"+ std::to_string((p.x*real_w)/calib_w)+", \"y\":"+ std::to_string((p.y*real_h)/calib_h)+"}},";
    }
    output.pop_back();
    output += " ] }";


    #ifdef SHOW_MAIN_RESULTS
        std::chrono::steady_clock::time_point eend1 = std::chrono::steady_clock::now();
        std::cout << "Calibrated Scene time = " << std::chrono::duration_cast<std::chrono::seconds>(eend1 - ebegin).count() << "[s]" << std::endl;
        //cv::resize(out, out, cv::Size(M1.cols*4, M1.rows*4));
        cv::imwrite("Full calibration.png", fout);
        std::chrono::steady_clock::time_point ebegin2 = std::chrono::steady_clock::now();
        
    #endif

    return std::make_pair(0,output);
}

std::pair<int,std::string> Calibration_3::getMarksSemiAutomatic(std::string email,std::string screenshot,
                                                                std::string mark1_x,std::string mark1_y,
                                                                std::string mark2_x,std::string mark2_y,
                                                                std::string mark3_x,std::string mark3_y,
                                                                std::string mark4_x,std::string mark4_y,
                                                                std::string mark5_x,std::string mark5_y,
                                                                std::string mark6_x,std::string mark6_y) {

    ///Process automatic calibration
    std::chrono::steady_clock::time_point ebegin = std::chrono::steady_clock::now();


    ImagemConverter test = ImagemConverter();
    std::string base64_imageFromApp = screenshot.erase(0,23);
    cv::Mat current, M, M1 = test.str2mat(base64_imageFromApp); //Carpeta parcial
    int real_w = M1.cols, real_h = M1.rows,
        calib_w = 1280, calib_h = (real_h * calib_w)/real_w; //Keeps proportion

    cv::resize(M1, M1, cv::Size(calib_w, calib_h));

#ifdef SHOW_MAIN_RESULTS
    cv::imwrite("BG Image.png", M1);
    
#endif

    

    cv::Mat carpet_area;
    M1.copyTo(carpet_area);

    //Print final markers:
    cv::Mat mout;
    M1.copyTo(mout);
    std::map<int, cv::Point2i> marker_position;
    marker_position[1] = cv::Point2i(stoi(mark1_x),stoi(mark1_y));
    marker_position[2] = cv::Point2i(stoi(mark2_x),stoi(mark2_y));
    marker_position[3] = cv::Point2i(stoi(mark3_x),stoi(mark3_y));
    marker_position[4] = cv::Point2i(stoi(mark4_x),stoi(mark4_y));
    marker_position[5] = cv::Point2i(stoi(mark5_x),stoi(mark5_y));
    marker_position[6] = cv::Point2i(stoi(mark6_x),stoi(mark6_y));

    std::cout << "Final markers: " << std::endl;
    std::map<int, cv::Point2i>::iterator mit, mend = marker_position.end();
    for(mit = marker_position.begin(); mit != mend; mit++) {
        cv::Point2i &p = mit->second;
        std::cout << mit->first << ": (" << p.x << "; " << p.y << ")" << std::endl;
        addMarker(mout, mit->second, mit->first);
    }
#ifdef SHOW_MAIN_RESULTS
        cv::imwrite("Available markers.png", mout);
        
#endif


    //Here we expect 4 points (for the moment...).
    //Calibrate scene
    std::vector<cv::Point2f> scenePoints;
    setScenePoints(scenePoints);

    std::vector<cv::Point2f> calibScenePoints, imagePoints;
    calibScenePoints.push_back(scenePoints[0]);
    imagePoints.push_back(cv::Point2f(marker_position[1].x, marker_position[1].y));
    calibScenePoints.push_back(scenePoints[1]);
    imagePoints.push_back(cv::Point2f(marker_position[2].x, marker_position[2].y));
    calibScenePoints.push_back(scenePoints[2]);
    imagePoints.push_back(cv::Point2f(marker_position[3].x, marker_position[3].y));
    calibScenePoints.push_back(scenePoints[3]);
    imagePoints.push_back(cv::Point2f(marker_position[4].x, marker_position[4].y));
    calibScenePoints.push_back(scenePoints[4]);
    imagePoints.push_back(cv::Point2f(marker_position[5].x, marker_position[5].y));
    calibScenePoints.push_back(scenePoints[5]);
    imagePoints.push_back(cv::Point2f(marker_position[6].x, marker_position[6].y));
    

//  cv::Mat H = cv::findHomography(calibScenePoints, imagePoints);
    cv::Mat H = cv::findHomography(calibScenePoints, imagePoints, cv::RANSAC, 5);
    cv::Mat fout;
    M1.copyTo(fout);
    std::map<int, std::vector<cv::Point2i> > objectiveImPos;
    for(int i=1; i<= 9; i++) {
        addPoint(fout, scenePoints[i-1], H, i);
    }
    std::string output = "{";
    output += "\"state\" : \"OK!\" ,";
    output += "\"backgroundImage\":\""+ base64_imageFromApp +"\",";
    output += "\"Wcalib\":"+ std::to_string(calib_w) +",";
    output += "\"Hcalib\":"+ std::to_string(calib_h) +",";
    output += "\"Width\":" + std::to_string(real_w) + ",";
    output += "\"Height\":" + std::to_string(real_h) + ",";
    //output += "\"Floor\": [ {\"id\":1,\"vertices\": [ {\"x\":1,\"y\":2},{\"x\":1,\"y\":2}] } ]";
    output += "\"Homography\": [";
    for(int i=0; i<= 8; i++) {
	output += std::to_string(H.at<double>(i)) + ",";
    }
    output.pop_back();
    output += " ] , ";
    output += "\"Floor\": [";
    for(int i=1; i<= 9; i++) {
    	output += "{\"id\":"+ std::to_string(i) + ", \"vertices\": [ ";
        cv::Point2f sp = scenePoints[i-1];
        std::vector<cv::Point2i> square;
        float marker_size = 7.5;
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y+marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y+marker_size), H));
        objectiveImPos[i] = square;
        
        //std::cout<<square[0].x<<std::endl;
        output += "{\"x\":"+ std::to_string((square[0].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[0].y*real_h)/calib_h)+"},"+"{\"x\":"+ std::to_string((square[1].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[1].y*real_h)/calib_h)+"},"+"{\"x\":"+ std::to_string((square[2].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[2].y*real_h)/calib_h)+"},"+"{\"x\":"+ std::to_string((square[3].x*real_w)/calib_w)+", \"y\":"+ std::to_string((square[3].y*real_h)/calib_h)+"}";
        output += " ] },";
    }
    output.pop_back();
    output += " ] , ";

    output += "\"Marks\": [";
    std::map<int, cv::Point2i>::iterator mit2, mend2 = marker_position.end();
    for(mit2 = marker_position.begin(); mit2 != mend2; mit2++) {
        cv::Point2i &p = mit2->second;
        output += "{\"id\":"+ std::to_string(mit2->first) + ", \"vertices\": ";
        output += "{\"x\":"+ std::to_string((p.x*real_w)/calib_w)+", \"y\":"+ std::to_string((p.y*real_h)/calib_h)+"}},";
    }
    output.pop_back();
    output += " ] }";


    #ifdef SHOW_MAIN_RESULTS
        std::chrono::steady_clock::time_point eend1 = std::chrono::steady_clock::now();
        std::cout << "Calibrated Scene time = " << std::chrono::duration_cast<std::chrono::seconds>(eend1 - ebegin).count() << "[s]" << std::endl;
        //cv::resize(out, out, cv::Size(M1.cols*4, M1.rows*4));
        cv::imwrite("Full calibration.png", fout);
        std::chrono::steady_clock::time_point ebegin2 = std::chrono::steady_clock::now();
        
    #endif



    return std::make_pair(0,output);
}

