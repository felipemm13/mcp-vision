#include "Calibration_fixed.h"

Calibration_fixed::Calibration_fixed(){

};

//If only cut boxes remaining
bool Calibration_fixed::cutBBoxes(std::map<int, cv::Rect> &bboxes) {
    if(bboxes.size()==0)
        return true;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit)
       if(bit->second.y != 0)
           return false;
    return true;
}

//Suppress unviable bboxes based on normal shape features from markers
void Calibration_fixed::suppressUnviable(std::map<int, cv::Rect> &bboxes, int calibImgW, int calibImgH, cv::Mat &curImgMask) {
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
int Calibration_fixed::getLowestFromBBoxes(std::map<int, cv::Rect> &bboxes) {
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
int Calibration_fixed::getCorrectionToCutBoxes(std::map<int, cv::Rect> &bboxes) {
    int correction = 0, cur;
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
int Calibration_fixed::getViableCentralBBox(std::map<int, cv::Rect> &bboxes, int y_start,
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
int Calibration_fixed::getViableBBoxNearestToMarkerLeft(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
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

int Calibration_fixed::getViableBBoxHorizontalmostToMarkerRight(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
    int rindex = -1;
    float d, min_d = 360;
    std::map<int, cv::Rect>::iterator bit, bend = bboxes.end();
    for(bit=bboxes.begin(); bit!=bend; ++bit) {
        cv::Rect &bb = bit->second;
        if(bb.y == 0)
            continue;
        if(bb.x > marker.x) { //Viable
            d = fabs(atan2(fabs(marker.y - bb.y), bb.x - marker.x))*180/M_PI;
            //std::cout << "BB ID: " << bit->first << ": angle --> " << d << std::endl;
            if(d < min_d) {
                min_d = d;
                rindex = bit->first;
            }
        }
    }
    return rindex;
}

int Calibration_fixed::getViableBBoxHorizontalmostToMarkerLeft(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
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
int Calibration_fixed::getViableBBoxNearestToMarkerRight(std::map<int, cv::Rect> &bboxes, cv::Point2i &marker, cv::Mat &curImgMask) {
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
cv::Point2i Calibration_fixed::getMarker(cv::Rect &r, cv::Mat &curImgMask) {
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
int Calibration_fixed::lowestMarker(std::map<int, cv::Rect> &bboxes, int &yl_index) {
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
void Calibration_fixed::candidateStats(std::map<int, cv::Rect> &bboxes, int calibImgW, int calibImgH, cv::Mat &curImgMask) {
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
void Calibration_fixed::paintRectangles(cv::Mat &img, std::map<int, cv::Rect> &bboxes, cv::Scalar &color) {
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
void Calibration_fixed::getBlobs(cv::Mat &connected_components, std::map<int, cv::Rect> &bboxes) {
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

cv::Mat Calibration_fixed::BGMaskFromYUV(const cv::Mat &image, int Yl, int Yh, cv::Scalar &yuv_bin, int bins) {
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(0));
    uchar *data = image.data, *mdata = mask.data;
    int Y, U, V, i, k, bsize = 256/bins, bsize_2 = bsize/2, rows_cols=image.rows*image.cols;
    int Ul = yuv_bin.val[1] - bsize_2, Uh = yuv_bin.val[1] + bsize_2,
        Vl = yuv_bin.val[2] - bsize_2, Vh = yuv_bin.val[2] + bsize_2;

    for(i=0, k=0; i<rows_cols; ++i, k+=3) {
        Y = data[k];
        U = data[k+1];
        V = data[k+2];
        if(U>=Ul && U<=Uh && V>=Vl && V<=Vh && Y>=Yl && Y<=Yh)
            mdata[i] = 255;
    }
    return mask;

}


//Obtains foreground mask from initial fg mean and bg mean, according to proximity of each pixel.
//It also calculates the new mean fg and bg from actual fg and bg pixels, and returns pixel counts.
cv::Mat Calibration_fixed::FGMaskFromImage(const cv::Mat &image,
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
cv::Mat Calibration_fixed::FGMaskFromImage_masked(const cv::Mat &image,
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



cv::Mat Calibration_fixed::completeNearlyConnected(cv::Mat &img, cv::Mat initial_mask, int T) {
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

cv::Mat Calibration_fixed::setMaskFromColor(cv::Mat &img, cv::Scalar mask_color, double T) {
    cv::Mat result;
    double B = mask_color.val[0], G = mask_color.val[1], R = mask_color.val[2];
    int lB = B - T < 0 ? 0 : B - T,
          lG = G - T < 0 ? 0 : G - T,
          lR = R - T < 0 ? 0 : R - T,
          uB = B + T > 255 ? 255 : B + T,
          uG = G + T > 255 ? 255 : G + T,
          uR = R + T > 255 ? 255 : R + T;
    /*std::cout << (int)lB << "; " << (int)lG << "; " << (int)lR << std::endl;
    std::cout << (int)uB << "; " << (int)uG << "; " << (int)uR << std::endl;*/
    cv::inRange(img, cv::Scalar(lB, lG, lR), cv::Scalar(uB, uG, uR), result);
    return result;
}

//Get current interpolated max:
float Calibration_fixed::interpolatedMax(const std::vector<float> &v, std::vector<bool> &used, int &freq, int &i_low, int &i_up) {
    int i, bins = v.size(), i_max, max = -1;

    //Get initial max and init used vector
    for(i=0; i<bins; ++i)
        if(!used[i] && v[i] > max) {
            max = v[i];
            i_max = i;
        }

    //All used:
    if(max <= 0) {
        i_low = i_up = 0;
        freq = 0;
        return 0;
    }

    // Set interpolation interval:
    int weight, vprev;
    //Min:
    vprev = weight = max;
    used[i_max] = true;
    i_low = i_max;
    for(i=i_max-1; i>=0; --i) {
        if(v[i] == 0) {
            i_low = i;
            used[i] = true;
            weight += v[i];
            break;
        } else if(v[i]<=vprev) {
            i_low = i;
            used[i] = true;
            vprev = v[i];
            weight += vprev;
        } else
            break;
    }

    //Max:
    vprev = max;
    i_up = i_max;
    for(i=i_max+1; i<bins; ++i) {
        if(v[i] == 0) {
            i_up = i;
            used[i] = true;
            weight += v[i];
            break;
        } else if(v[i]<=vprev) {
            i_up = i;
            used[i] = true;
            vprev = v[i];
            weight += vprev;
        } else
            break;
    }

    //Weighted mean v:
    float vv = 0, bsize = 256/bins, b_size_2 = bsize/2;
    for(i=i_low; i<=i_up; ++i)
        vv += (b_size_2 + i*bsize)*v[i];
    vv /= weight;
    freq = weight;
    return vv;
}

//Evaluates two top most peaks and returns interpolated Y of max frequency:
float Calibration_fixed::getPeakYInterval(const std::vector<float> &Y, int &freq, int &iY_low, int &iY_high) {
    int i, bins = Y.size();

    std::vector<bool> used(bins);
    for(i=0; i<bins; ++i)
        used[i] = false;

    int f1, il1, ih1, f2, il2, ih2;
    float Y1 = interpolatedMax(Y, used, f1, il1, ih1),
          Y2 = interpolatedMax(Y, used, f2, il2, ih2);

    if(f1 > f2) {
        freq = f1;
        iY_low = il1;
        iY_high = ih1;
        return Y1;
    }
    freq = f2;
    iY_low = il2;
    iY_high = ih2;
    return Y2;
}


//Finds histogram UV peak and adjusts it interpolating on Y neighborhood (2*nsize)x(2*nsize)x(2*nsize)
double Calibration_fixed::histoFindYUVMax(const cv::Mat &histo, int bins, cv::Scalar &best_max, int &Y_low, int &Y_high) {
    int bsize = 256/bins, bb=bins*bins;
    cv::Mat UVscore = cv::Mat::zeros(bins, bins, CV_32SC1);
    cv::Mat Ypeak = cv::Mat::zeros(bins, bins, CV_32FC1);
    cv::Mat iYlow = cv::Mat::zeros(bins, bins, CV_32SC1);
    cv::Mat iYhigh = cv::Mat::zeros(bins, bins, CV_32SC1);

    //Get max frequency Y peak for each UV:
    int freq, il, ih;
    float vY;
    for(int i=0, mU=bsize/2; i<bins; ++i, mU+=bsize)
        for(int j=0, mV=bsize/2; j<bins; ++j, mV+=bsize) {
            std::vector<float> Y(bins);
            for(int k=0; k<bins; ++k)
                Y[k] = histo.at<float>(k*bb + i*bins + j);
            vY = getPeakYInterval(Y, freq, il, ih);
            Ypeak.at<float>(i,j) = vY;
            iYlow.at<int>(i,j) = il;
            iYhigh.at<int>(i,j) = ih;
            UVscore.at<int>(i,j) = freq;
            /*if(freq > 0)
                std::cout << "(" << mU << ", " << mV << "): "
                          << vY << " [" << il*bsize << ";" << (ih+1)*bsize << "] --> " << freq << std::endl;*/
        }

    float bY, bU, bV;
    int best = -1;
    il = ih = 0;
    for(int i=0, mU=bsize/2; i<bins; ++i, mU+=bsize)
        for(int j=0, mV=bsize/2; j<bins; ++j, mV+=bsize) {
            if(UVscore.at<int>(i,j) > best) {
                best = UVscore.at<int>(i,j);
                bY = Ypeak.at<float>(i,j);
                bU = mU;
                bV = mV;
                il = iYlow.at<int>(i,j);
                ih = iYhigh.at<int>(i,j);
            }
        }
    Y_low = il*bsize;
    Y_high = (ih+1)*bsize;
    best_max = cv::Scalar(bY, bU, bV);
    return best;
}


//Finds histogram peak and adjusts it interpolating on neighborhood (2*nsize)x(2*nsize)x(2*nsize)
double Calibration_fixed::histoFindAdjustedMax(const cv::Mat &histo, int bins, int nsize, cv::Scalar &best_max) {
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

void Calibration_fixed::histogram3D_masked(const cv::Mat &im, int bins, cv::Mat *histo, cv::Mat &mask) {
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
void Calibration_fixed::histogram3D(const cv::Mat &im, int bins, cv::Mat *histo, bool normalize) {
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
    if(normalize && count > 0)
        for(i=0; i<bbb; ++i)
             histo->at<float>(i) /= count;
}

//Set scene points as star configuration (5 at center, and 8 markers at 200cm distances, 45° distance each
void Calibration_fixed::setScenePoints(std::vector<cv::Point2f> &scenePoints) {
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
cv::Point2i Calibration_fixed::transform(cv::Point2f p, cv::Mat &H) {
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0,0) = p.x;
    pin.at<double>(1,0) = p.y;
    pin.at<double>(2,0) = 1;

    cv::Mat pout = H*pin;

    return cv::Point2i(rint(pout.at<double>(0,0)/pout.at<double>(2,0)),
                       rint(pout.at<double>(1,0)/pout.at<double>(2,0)));
}

//Draws a 15 cm square on image
std::vector<cv::Point> Calibration_fixed::drawRectangle(cv::Mat &img, cv::Point2f &p, cv::Mat &H) {
    float hside = 7.5;
    std::vector<cv::Point> squarePoints;

    cv::Point2f corners[4];
    corners[0] = transform(cv::Point2f(p.x-hside, p.y-hside), H);
    corners[1] = transform(cv::Point2f(p.x+hside, p.y-hside), H);
    corners[2] = transform(cv::Point2f(p.x+hside, p.y+hside), H);
    corners[3] = transform(cv::Point2f(p.x-hside, p.y+hside), H);

    for (int i = 0; i < 4; ++i) {
        cv::line(img, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 255));
        squarePoints.push_back(corners[i]);
    }

    return squarePoints;
}

//Get image point from scene coordinate
cv::Point2i Calibration_fixed::getPoint(cv::Point2f p, cv::Mat &H) {
    //Get center in image coordinates
    return transform(p, H);
}

//Draw circular marker and writes id on image
void Calibration_fixed::addMarker(cv::Mat &img, const cv::Point2i &p, int index) {
    cv::circle(img, p, 3, cv::Scalar(0,0,255));
    cv::putText(img, std::to_string(index),
                cv::Point(p.x+2,p.y), cv::FONT_HERSHEY_DUPLEX, 0.5,
                cv::Scalar(0,255,0));
}

//Adds marker with circle, 15 cm box and id, to image
void Calibration_fixed::addPoint(cv::Mat &img, cv::Point2f &p, cv::Mat &H, int index, std::vector<cv::Point3f> &drawnPoints) {
    cv::Point2i p_im = transform(p, H);
    cv::circle(img, p_im, 3, cv::Scalar(0,0,255));

    drawRectangle(img, p, H);
    drawnPoints.push_back(cv::Point3f(p_im.x, p_im.y, index));
    cv::putText(img, std::to_string(index),
                cv::Point(p_im.x+2,p_im.y), cv::FONT_HERSHEY_DUPLEX, 0.5,
                cv::Scalar(0,255,0));
}

//Adds marker with circle, 15 cm box and id, to image
std::vector<cv::Point> Calibration_fixed::addPoint_2(cv::Mat &img, cv::Point2f &p, cv::Mat &H, int id) {
    cv::Point2i p_im = transform(p, H);
    cv::circle(img, p_im, 3, cv::Scalar(0,0,255));

    std::vector<cv::Point> squarePoints;
    squarePoints = drawRectangle(img, p, H);

    cv::putText(img, std::to_string(id),
                cv::Point(p_im.x + 2, p_im.y), cv::FONT_HERSHEY_DUPLEX, 0.5,
                cv::Scalar(0,255,0));
    return squarePoints;
}


size_t Calibration_fixed::write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    vector<uchar> *stream = (vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);
    return count;
}

//function to retrieve the image as cv::Mat data type
cv::Mat Calibration_fixed::curlImg(const char *img_url, int timeout)
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


// Function that finds the closest black pixel to a given point (mask)
void Calibration_fixed::findClosestBlackPoints(const cv::Mat& mask, const std::vector<cv::Point3f>& points, std::vector<cv::Point>& closestBlackPoints, cv::Mat& visualizedMask, std::map<int, bool>& marker_availability){
    closestBlackPoints.clear();
    float maxDistance = 100;
    for (const auto& point : points) {
        float minDistance = std::numeric_limits<float>::max();
        cv::Point closestBlackPoint;
        cv::Point punto2D(static_cast<int>(point.x), static_cast<int>(point.y));
        bool foundBlackPixel = false;

        for (int y = 0; y < mask.rows; ++y) {
            for (int x = 0; x < mask.cols; ++x) {
                if (mask.at<uchar>(y, x) == 0) { // Píxel negro
                    cv::Point currentPoint(x, y);
                    float currentDistance = std::sqrt(std::pow(point.x - x, 2) + std::pow(point.y - y, 2));
                    if (currentDistance < minDistance && currentDistance < maxDistance) {
                        minDistance = currentDistance;
                        closestBlackPoint = currentPoint;
                        foundBlackPixel = true;
        }}}}

        if (foundBlackPixel) {
            marker_availability[point.z] = true;
            closestBlackPoints.push_back(closestBlackPoint);
            cv::circle(visualizedMask, punto2D, 3, cv::Scalar(0, 0, 255), -1); // Punto de interés en rojo
            cv::circle(visualizedMask, closestBlackPoint, 3, cv::Scalar(255, 0, 0), -1); // Punto negro más cercano en azul
            cv::line(visualizedMask, punto2D, closestBlackPoint, cv::Scalar(0, 255, 0)); // Línea horizontal en verde
        }
}}


// Functions to compare the color between two points
bool Calibration_fixed::areColorsSimilar(cv::Vec3b color1, cv::Vec3b color2, int Tolerance) {
    cv::Mat colorMat1, colorMat2;
    cv::cvtColor(cv::Mat(1, 1, CV_8UC3, color1), colorMat1, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(cv::Mat(1, 1, CV_8UC3, color2), colorMat2, cv::COLOR_BGR2YCrCb);
    cv::Vec3b ycrcb1 = colorMat1.at<cv::Vec3b>(0, 0);
    cv::Vec3b ycrcb2 = colorMat2.at<cv::Vec3b>(0, 0);

    return (std::abs(ycrcb1[0] - ycrcb2[0]) < Tolerance) &&
           (std::abs(ycrcb1[1] - ycrcb2[1]) < Tolerance) &&
           (std::abs(ycrcb1[2] - ycrcb2[2]) < Tolerance);
}

bool Calibration_fixed::compareColorsAt(cv::Mat &image, int x1, int y1, cv::Vec3b color_ref, int luminosityTolerance) {
    cv::Vec3b color1 = image.at<cv::Vec3b>(cv::Point(x1, y1));
    return areColorsSimilar(color1, color_ref, luminosityTolerance);
}

cv::Vec3b Calibration_fixed::findPredominantColor(const cv::Mat& imagen, int x, int y, int radio) {
    int yInicio = std::max(0, y - radio);
    int yFin = std::min(imagen.rows, y + radio + 1);
    int xInicio = std::max(0, x - radio);
    int xFin = std::min(imagen.cols, x + radio + 1);

    cv::Mat roi = imagen(cv::Rect(xInicio, yInicio, xFin - xInicio, yFin - yInicio));
    int totalPixeles = roi.rows * roi.cols;
    cv::Mat pixeles(totalPixeles, 1, CV_8UC3);
    int indice = 0;
    for (int y = 0; y < roi.rows; ++y) {
        for (int x = 0; x < roi.cols; ++x) {
            cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);
            pixeles.at<cv::Vec3b>(indice++, 0) = pixel;
        }
    }
    int maxCount = 0;
    cv::Vec3b colorPredominant;

    for (int i = 0; i < totalPixeles; ++i) {
        int count = 0;
        for (int j = 0; j < totalPixeles; ++j) {
            if (pixeles.at<cv::Vec3b>(i, 0) == pixeles.at<cv::Vec3b>(j, 0)) {++count;}
        }

        if (count > maxCount) {
            maxCount = count;
            colorPredominant = pixeles.at<cv::Vec3b>(i, 0);
    }}
    return colorPredominant;
}

// Here we calculate de distances between the 6 points calculated with the algotim vs the 6 first point after H
void Calibration_fixed::addFixed_marks(std::vector<cv::Point3f> imagePoints_aux, std::vector<cv::Point3f> drawnPoints, cv::Mat& fout_fixed, cv::Mat visualizedMask) {
    for (const auto& point : imagePoints_aux) {
        addMarker(fout_fixed, cv::Point2i(point.x, point.y), point.z+1);
        cv::Point drawnPoint2D(drawnPoints[point.z].x, drawnPoints[point.z].y); //Aux var to make 2D drawnpoints
        cv::Point imagePoints2D(point.x, point.y); //Aux var to make 2D drawnpoints
        cv::circle(visualizedMask, imagePoints2D, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(visualizedMask, drawnPoint2D, 3, cv::Scalar(255, 0, 0), -1);
        cv::line(visualizedMask, imagePoints2D, drawnPoint2D, cv::Scalar(0, 255, 0)); // Línea horizontal en verde
    }
}


std::tuple<int,std::string ,std::vector<PointWithContour>, std::string, int, int > Calibration_fixed::getMarksAutomatic(std::string screenshot){
    try{
    ///Process automatic calibration
    std::chrono::steady_clock::time_point ebegin = std::chrono::steady_clock::now();

    ImagemConverter test = ImagemConverter();
    std::string base64_imageFromApp = screenshot.erase(0,23);
    cv::Mat current, M, M1 = test.str2mat(base64_imageFromApp); //Carpeta parcial

    int real_w = M1.cols, real_h = M1.rows,
        calib_w = 1280, calib_h = (real_h * calib_w)/real_w; //Keeps proportion

    cv::resize(M1, M1, cv::Size(calib_w, calib_h));

#ifdef SHOW_MAIN_RESULTS
    cv::imshow("BG Image", M1);
    cv::waitKey(0);
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
    cv::imshow("Global Color Median Blur Image", M1m);
    cv::waitKey(0);
#endif

    //0.2 Get half image
    cv::Mat M1h(M1m, cv::Rect(0,M1.rows/2,M1.cols,M1.rows/2));
#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("Half Image", M1h);
    cv::waitKey(0);
#endif

    int gBGRbins = 16, gYUVbins = 16;
    cv::Mat YUV, ghistoBGR, ghistoYUV;
    histogram3D(M1h, gBGRbins, &ghistoBGR);
    cv::cvtColor(M1h, YUV, cv::COLOR_BGR2YCrCb);
    histogram3D(M1h, gBGRbins, &ghistoBGR);
    histogram3D(YUV, gYUVbins, &ghistoYUV, false);

    float gbest;
    cv::Scalar gbest_max;
    int Yl, Yh;
    gbest = histoFindYUVMax(ghistoYUV, gYUVbins, gbest_max, Yl, Yh);

#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "Best YUV: "
              << gbest_max.val[0] << " [" << Yl << ";" << Yh << "],"
              << gbest_max.val[1] << ","
              << gbest_max.val[2] << ": " << gbest << std::endl;
#endif

    //0.4 Alternative YUV: Get simple mask from Y interval, and U,V on their bin,
    //    assuming that color will not change significantly.
    cv::Mat ginitial_mask_YUV;
    ginitial_mask_YUV = BGMaskFromYUV(YUV, Yl, Yh, gbest_max, gYUVbins);
    //ginitial_mask = setMaskFromColor(M1m, bg_adj, 20);
#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("GInitial Mask YUV", ginitial_mask_YUV);
    cv::waitKey(0);
#endif

    cv::Mat global_mask = cv::Mat::zeros(M1m.size(), CV_8UC1), partial_gmask(global_mask, cv::Rect(0,M1.rows/2,M1.cols,M1.rows/2));
    ginitial_mask_YUV.copyTo(partial_gmask);

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("Global Mask", global_mask);
    cv::waitKey(0);
#endif

    cv::Mat ext_global_mask = completeNearlyConnected(M1m, global_mask, 20);

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
    /*std::cout << "First: " << first << std::endl;
    std::cout << "First black: " << first_black << std::endl;
    std::cout << "Last: " << last << std::endl;
    std::cout << "Non-convex: " << non_convex << std::endl;*/
    if(non_convex) {
        if(last < first_black) { //White till the end, after concavity
            memset(ext_global_mask.data + eind + first, 255, ecols - first);
        } else {
            memset(ext_global_mask.data + eind + first, 255, last - first + 1);
        }
    }

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("Extended Global Mask", ext_global_mask);
    cv::waitKey(0);
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
    cv::imshow("Carpet area", carpet_area0);
    cv::imshow("Carpet mask", carpet_mask0);
    cv::waitKey(0);
#endif



    //1. Cycle till first bboxes are found
    int it_count = 0;
    while(true) { //While no valid initial markers found yet
        ++it_count;
        cv::Mat M1r(M1m, cv::Rect(0, y_start, M1.cols, y_end-y_start+1));
        cv::Mat mask_r(carpet_mask0, cv::Rect(0, y_start, M1.cols, y_end-y_start+1));

#ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imshow("Image Section", M1r);
        cv::waitKey(0);
#endif

#ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imshow("Local Mask", mask_r);
        cv::waitKey(0);
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
        cv::imshow("Initial mask", initial_mask);
        cv::waitKey(0);
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
        std::cout << "ACA PRIMERA MASK" << std::endl;
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
        cv::imshow("Initial bboxes", im_bb);
        cv::waitKey(0);
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
            //std::cout << mit->first << ": (" << p.x << "; " << p.y << ")" << std::endl;
        }
    }

    cv::Mat carpet_area;
    M1.copyTo(carpet_area);

    //Print final markers:
    // MARCAS FINALES, marker position guarda todos las posiciones, cada marker_position tiene un first referente al id
    // y un second referente a la posicion de la marca
    cv::Mat mout;
    M1.copyTo(mout);
    /*std::cout << "Final markers: " << std::endl;*/
    std::map<int, cv::Point2i>::iterator mit, mend = marker_position.end();
    for(mit = marker_position.begin(); mit != mend; mit++) {
        cv::Point2i &p = mit->second;
        //std::cout << mit->first << ": (" << p.x << "; " << p.y << ")" << std::endl;
        addMarker(mout, mit->second, mit->first);
    }
#ifdef SHOW_MAIN_RESULTS
        cv::imshow("Available markers", mout);
        cv::waitKey(0);
#endif

    //Here we expect 4 points (for the moment...).
    //Calibrate scene
    std::vector<cv::Point2f> scenePoints;
    setScenePoints(scenePoints); // Scene points es la estrella hecha a mano

    std::vector<cv::Point2f> calibScenePoints, imagePoints;
    std::vector<cv::Point3f> imagePoints_aux;

    calibScenePoints.push_back(scenePoints[0]);
    imagePoints.push_back(cv::Point2f(marker_position[1].x, marker_position[1].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[1].x, marker_position[1].y, 0));
    
    calibScenePoints.push_back(scenePoints[1]);
    imagePoints.push_back(cv::Point2f(marker_position[2].x, marker_position[2].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[2].x, marker_position[2].y, 1));

    calibScenePoints.push_back(scenePoints[2]);
    imagePoints.push_back(cv::Point2f(marker_position[3].x, marker_position[3].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[3].x, marker_position[3].y, 2));

    if(marker_availability[4]) {
        calibScenePoints.push_back(scenePoints[3]);
        imagePoints.push_back(cv::Point2f(marker_position[4].x, marker_position[4].y));
        imagePoints_aux.push_back(cv::Point3f(marker_position[4].x, marker_position[4].y, 3));
    }

    calibScenePoints.push_back(scenePoints[4]);
    imagePoints.push_back(cv::Point2f(marker_position[5].x, marker_position[5].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[5].x, marker_position[5].y, 4));

    if(marker_availability[6]) {
        calibScenePoints.push_back(scenePoints[5]);
        imagePoints.push_back(cv::Point2f(marker_position[6].x, marker_position[6].y));
        imagePoints_aux.push_back(cv::Point3f(marker_position[6].x, marker_position[6].y, 5));
    }

    if(marker_availability[7]) {
        calibScenePoints.push_back(scenePoints[6]);
        imagePoints.push_back(cv::Point2f(marker_position[7].x, marker_position[7].y));
        imagePoints_aux.push_back(cv::Point3f(marker_position[7].x, marker_position[7].y, 6));
    }
    if(marker_availability[8]) {
        calibScenePoints.push_back(scenePoints[7]);
        imagePoints.push_back(cv::Point2f(marker_position[8].x, marker_position[8].y));
        imagePoints_aux.push_back(cv::Point3f(marker_position[8].x, marker_position[8].y, 7));
    }
    if(marker_availability[9]) {
        calibScenePoints.push_back(scenePoints[8]);
        imagePoints.push_back(cv::Point2f(marker_position[9].x, marker_position[9].y));
        imagePoints_aux.push_back(cv::Point3f(marker_position[9].x, marker_position[9].y, 8));
    }
    // Marker position [7,8,9] are (0,0)

    //===============================================================================================//
    cv::Mat H = cv::findHomography(calibScenePoints, imagePoints, cv::RANSAC, 5);
    cv::Mat fout;
    cv::Mat fout_fixed;
    std::vector<cv::Point3f> drawnPoints; // Here we save the final points with index
    M1.copyTo(fout);
    M1.copyTo(fout_fixed);

    std::map<int, std::vector<cv::Point2i> > objectiveImPos;
    for(int i=1; i<= 9; i++) {
        addPoint(fout, scenePoints[i-1], H, i, drawnPoints);
    }

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Homografia", fout);
        cv::waitKey(0);
    #endif

    for(int i=1; i<= 9; i++) {
        cv::Point2f sp = scenePoints[i-1];
        std::vector<cv::Point2i> square;
        float marker_size = 7.5;
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y+marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y+marker_size), H));
        objectiveImPos[i] = square;
    }

    //===============================================================================================//

    // We make a filter by color at the mask
    cv::Mat ext_global_mask_filtered = ext_global_mask.clone();
    cv::Vec3b color_ref = findPredominantColor(M1m, imagePoints[1].x, imagePoints[1].y, 15);

    #ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Predominant Color: " << color_ref << std::endl;
    #endif

    for (int y = ext_global_mask_filtered.rows/4; y < ext_global_mask_filtered.rows; ++y) {
        for (int x = 0; x < ext_global_mask_filtered.cols; ++x) {
            if (ext_global_mask_filtered.at<uchar>(y, x) == 0) { 
                if (!compareColorsAt(M1m, x, y, color_ref, 30)) {
                    ext_global_mask_filtered.at<uchar>(y, x) = 255;
    }}}}
    //===============================================================================================//

    // New mask with with lines between the real mark and the correction
    cv::Mat visualizedMask;
    cv::cvtColor(ext_global_mask_filtered, visualizedMask, cv::COLOR_GRAY2BGR);
    addFixed_marks(imagePoints_aux, drawnPoints, fout_fixed, visualizedMask);
    //===============================================================================================//

    // We are gonna find the positions of the 3 points who the algoritm doesnt calculate
    std::vector<cv::Point3f> selectedPoints;
    std::vector<cv::Point> closestBlackPoints;
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat maskClone = ext_global_mask_filtered.clone();
    cv::Mat drawing = cv::Mat::zeros(ext_global_mask_filtered.size(), CV_8UC3);

    std::vector<int> ids;
    if(!marker_availability[6]){ids.push_back(6);}
    ids.push_back(9);
    ids.push_back(8);
    ids.push_back(7);
    if(!marker_availability[4]){ids.push_back(4);}


    if(!marker_availability[4]){selectedPoints.push_back(drawnPoints[3]);}
    if(!marker_availability[6]){selectedPoints.push_back(drawnPoints[5]);}
    selectedPoints.push_back(drawnPoints[6]);
    selectedPoints.push_back(drawnPoints[7]);
    selectedPoints.push_back(drawnPoints[8]);

    findClosestBlackPoints(ext_global_mask_filtered, selectedPoints, closestBlackPoints, visualizedMask, marker_availability);
    //===============================================================================================//

    // We are gonna findContours on a defined rectangle in the image for efficiency
    cv::Point3f point0, point1, point2, point3, point4, point5, point6, point7, point8;
    for (const auto &point : imagePoints_aux) { // We need to identify each point for error management
        if (point.z == 0){point0 = point;}
        if (point.z == 1){point1 = point;}
        if (point.z == 2){point2 = point;}
        if (point.z == 3){point3 = point;}
        if (point.z == 4){point4 = point;}
        if (point.z == 5){point5 = point;}
        if (point.z == 6){point6 = point;}
        if (point.z == 7){point7 = point;}
        if (point.z == 8){point8 = point;}}   

    int verticalDistance = point1.y - point4.y;
    int height = (verticalDistance * 3) / 4;
    int x = point2.x;
    int y = point1.y;
    int x2 = point0.x;
    int y2 = point4.y - height;
    cv::Rect squareROI(x - 100, y2 , x2-x + 220, y-y2 + 50); //we add some slack
    cv::rectangle(drawing, squareROI, cv::Scalar(255));

    cv::Mat squareMask = maskClone(squareROI);
    cv::findContours(squareMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        for (size_t j = 0; j < contours[i].size(); j++) {
            contours[i][j].x += x - 100;
            contours[i][j].y += y2;
    }}

    for (size_t i = 0; i < contours.size(); i++) {
        double contourArea = cv::contourArea(contours[i]);
        if (contourArea > 50 && contourArea < 100000) {
            cv::Moments m = cv::moments(contours[i]);
            if (m.m00 > 0) {
                int cx = static_cast<int>(m.m10 / m.m00);
                int cy = static_cast<int>(m.m01 / m.m00);
                cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                cv::drawContours(drawing, contours, static_cast<int>(i), color, 2, cv::LINE_8);
                std::string areaText = std::to_string(contourArea);
                cv::putText(drawing, areaText, cv::Point(cx, cy), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                cv::circle(drawing, cv::Point(cx, cy), 2, color, -1);
    }}}

    #ifdef SHOW_TEST
        cv::imshow("Contornos", drawing);
        cv::waitKey(0);
    #endif
    //===============================================================================================//

    // We are gonna add fixed 7,8,9 points to the image
    std::vector<cv::Point2i> centroids;

    for (const auto& contour : contours) {
        double contourArea = cv::contourArea(contour);
        cv::Moments m = cv::moments(contour);
        if (contourArea > 50 && contourArea < 100000){
            for (const auto& blackpoint : closestBlackPoints) {
                if (cv::pointPolygonTest(contour, blackpoint, false) >= 0) {
                    double area = cv::contourArea(contour);
                    cv::drawContours(visualizedMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 255, 0), 2);
                    if (m.m00 > 0) {
                        int cx = static_cast<int>(m.m10 / m.m00);
                        int cy = static_cast<int>(m.m01 / m.m00);
                        centroids.push_back(cv::Point2i(cx, cy));}
                    break;
    }}}}

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Mascara con correciones", visualizedMask);
        cv::waitKey(0);
    #endif

    // We will order the points from left to right
    std::sort(centroids.begin(), centroids.end(), [](const cv::Point2i& a, const cv::Point2i& b) {
        return a.x < b.x;
    });

    std::vector<cv::Point3f> fixed_final_marks;
    for (size_t i = 0; i < centroids.size(); ++i) {
        addMarker(fout_fixed, centroids[i], ids[i]);
        fixed_final_marks.push_back(cv::Point3f(centroids[i].x, centroids[i].y, static_cast<float>(ids[i])));
    }

    for (const auto& punto : imagePoints_aux) {fixed_final_marks.push_back(cv::Point3f(punto.x, punto.y, static_cast<float>(punto.z+1)));}

    std::sort(fixed_final_marks.begin(), fixed_final_marks.end(), 
              [](const cv::Point3f& a, const cv::Point3f& b) {
                  return a.z < b.z;
              });


    // Lets save the final points with contour
    std::vector<PointWithContour> final_points_contour;

    for (size_t i = 0; i < fixed_final_marks.size(); ++i) {
        cv::Point punto_aux(fixed_final_marks[i].x, fixed_final_marks[i].y);
        for (size_t j = 0; j < contours.size(); ++j) {
            double distancia = cv::pointPolygonTest(contours[j], punto_aux, true);
            if (distancia >= 0) {
                PointWithContour puntoConContorno;
                puntoConContorno.punto = fixed_final_marks[i];
                puntoConContorno.indiceContorno = j;
                puntoConContorno.contorno = contours[j];
                final_points_contour.push_back(puntoConContorno);
                break;
            }
        }
    }

    // if we not find any mark we just return the mark created by the homography
    std::vector<int> markerIndices = {4, 6, 7, 8, 9};

    for (int index : markerIndices) {
        if (!marker_availability[index]) {
            std::vector<cv::Point> rectangle = addPoint_2(fout_fixed, scenePoints[index - 1], H, index);
            PointWithContour puntoHomografia;
            puntoHomografia.punto.x = drawnPoints[index - 1].x;
            puntoHomografia.punto.y = drawnPoints[index - 1].y;
            puntoHomografia.punto.z = index;
            puntoHomografia.indiceContorno = 20 + index;
            puntoHomografia.contorno = rectangle;
            final_points_contour.push_back(puntoHomografia);
    }}


    // Lets draw the final image
    cv::Mat fout_final;
    M1.copyTo(fout_final);
    for (const PointWithContour& point : final_points_contour) {
        cv::drawContours(fout_final, std::vector<std::vector<cv::Point>>(1, point.contorno), 0, cv::Scalar(0, 255, 255), 2);
        cv::circle(fout_final, cv::Point(point.punto.x, point.punto.y), 3, cv::Scalar(0, 0, 255));
        std::string idText = std::to_string(static_cast<int>(point.punto.z));
        cv::putText(fout_final, idText,
                    cv::Point(point.punto.x + 5, point.punto.y - 5), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    cv::Scalar(0, 255, 0));
    }

    #ifdef SHOW_TEST
        for (const auto& punto : final_points_contour) {
            std::cout << "Punto ["<< punto.punto.z << "]: ubicado en (" << punto.punto.x << ", " << punto.punto.y << ")" << " pertenece al contorno " << punto.indiceContorno << std::endl;
        }
    #endif

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Fixed sin contornos", fout_fixed);
        cv::waitKey(0);
    #endif

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Fixed con contornos", fout_final);
        std::chrono::steady_clock::time_point eend1 = std::chrono::steady_clock::now();
        std::cout << "Calibrated Scene time = " << std::chrono::duration_cast<std::chrono::seconds>(eend1 - ebegin).count() << "[s]" << std::endl;
        cv::waitKey(0);
    #endif

    ImagemConverter Converter = ImagemConverter();
    std::string H_base64 = Converter.mat2str(H);

    return std::tuple<int, std::string, std::vector<PointWithContour>, std::string, int, int> (0, "{\"state\":\"success\"}", final_points_contour, H_base64, calib_w, calib_h);
    
    }catch(std::string JSON){
        std::string H_base64;
        std::vector<PointWithContour> final_points_contour;
        int calib_w; int calib_h;
        return std::tuple<int, std::string, std::vector<PointWithContour>, std::string, int, int> (-1, "{\"state\":\"error\"}", final_points_contour, H_base64, calib_w, calib_h);
    }
}


std::tuple<int,std::string ,std::vector<PointWithContour>, std::string, int, int > Calibration_fixed::getMarksSemiAutomatic(std::string screenshot,
                                                        double mark1_x, double mark1_y,
                                                        double mark2_x, double mark2_y,
                                                        double mark3_x, double mark3_y,
                                                        double mark4_x, double mark4_y,
                                                        double mark5_x, double mark5_y,
                                                        double mark6_x, double mark6_y) {

    try{
    ///Process automatic calibration
    std::chrono::steady_clock::time_point ebegin = std::chrono::steady_clock::now();

    std::cout << "Marca 1: X = " << mark1_x << ", Y = " << mark1_y << std::endl;
    std::cout << "Marca 2: X = " << mark2_x << ", Y = " << mark2_y << std::endl;
    std::cout << "Marca 3: X = " << mark3_x << ", Y = " << mark3_y << std::endl;
    std::cout << "Marca 4: X = " << mark4_x << ", Y = " << mark4_y << std::endl;
    std::cout << "Marca 5: X = " << mark5_x << ", Y = " << mark5_y << std::endl;
    std::cout << "Marca 6: X = " << mark6_x << ", Y = " << mark6_y << std::endl;

    ImagemConverter test = ImagemConverter();
    std::string base64_imageFromApp = screenshot.erase(0,23);
    cv::Mat current, M, M1 = test.str2mat(base64_imageFromApp); //Carpeta parcial
    int real_w = M1.cols, real_h = M1.rows,
        calib_w = 1280, calib_h = (real_h * calib_w)/real_w; //Keeps proportion

    cv::resize(M1, M1, cv::Size(calib_w, calib_h));

#ifdef SHOW_MAIN_RESULTS
    cv::imwrite("BG Image.png", M1);
    
#endif

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
        cv::imshow("Global Color Median Blur Image", M1m);
        cv::waitKey(0);
    #endif

    //0.2 Get half image
    cv::Mat M1h(M1m, cv::Rect(0,M1.rows/2,M1.cols,M1.rows/2));
    #ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imshow("Half Image", M1h);
        cv::waitKey(0);
    #endif

    int gBGRbins = 16, gYUVbins = 16;
    cv::Mat YUV, ghistoBGR, ghistoYUV;
    histogram3D(M1h, gBGRbins, &ghistoBGR);
    cv::cvtColor(M1h, YUV, cv::COLOR_BGR2YCrCb);
    histogram3D(M1h, gBGRbins, &ghistoBGR);
    histogram3D(YUV, gYUVbins, &ghistoYUV, false);

    float gbest;
    cv::Scalar gbest_max;
    int Yl, Yh;
    gbest = histoFindYUVMax(ghistoYUV, gYUVbins, gbest_max, Yl, Yh);

    #ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Best YUV: "
                << gbest_max.val[0] << " [" << Yl << ";" << Yh << "],"
                << gbest_max.val[1] << ","
                << gbest_max.val[2] << ": " << gbest << std::endl;
    #endif

    //0.4 Alternative YUV: Get simple mask from Y interval, and U,V on their bin,
    //    assuming that color will not change significantly.
    cv::Mat ginitial_mask_YUV;
    ginitial_mask_YUV = BGMaskFromYUV(YUV, Yl, Yh, gbest_max, gYUVbins);

    #ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imshow("GInitial Mask YUV", ginitial_mask_YUV);
        cv::waitKey(0);
    #endif

    cv::Mat global_mask = cv::Mat::zeros(M1m.size(), CV_8UC1), partial_gmask(global_mask, cv::Rect(0,M1.rows/2,M1.cols,M1.rows/2));
    ginitial_mask_YUV.copyTo(partial_gmask);

    #ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imshow("Global Mask", global_mask);
        cv::waitKey(0);
    #endif

    cv::Mat ext_global_mask = completeNearlyConnected(M1m, global_mask, 20);

    cv::Mat carpet_area;
    M1.copyTo(carpet_area);
    
    cv::Mat mout;
    M1.copyTo(mout);
    std::map<int, cv::Point2i> marker_position;
    marker_position[1] = cv::Point2i(mark1_x, mark1_y);
    marker_position[2] = cv::Point2i(mark2_x, mark2_y);
    marker_position[3] = cv::Point2i(mark3_x, mark3_y);
    marker_position[4] = cv::Point2i(mark4_x, mark4_y);
    marker_position[5] = cv::Point2i(mark5_x, mark5_y);
    marker_position[6] = cv::Point2i(mark6_x, mark6_y);

    std::map<int, bool> marker_availability;
    marker_availability[1] = true;
    marker_availability[2] = true;
    marker_availability[3] = true;
    marker_availability[4] = true;
    marker_availability[5] = true;
    marker_availability[6] = true;

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
    setScenePoints(scenePoints); // Scene points es la estrella hecha a mano

    std::vector<cv::Point2f> calibScenePoints, imagePoints;
    std::vector<cv::Point3f> imagePoints_aux;

    calibScenePoints.push_back(scenePoints[0]);
    imagePoints.push_back(cv::Point2f(marker_position[1].x, marker_position[1].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[1].x, marker_position[1].y, 0));
    
    calibScenePoints.push_back(scenePoints[1]);
    imagePoints.push_back(cv::Point2f(marker_position[2].x, marker_position[2].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[2].x, marker_position[2].y, 1));

    calibScenePoints.push_back(scenePoints[2]);
    imagePoints.push_back(cv::Point2f(marker_position[3].x, marker_position[3].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[3].x, marker_position[3].y, 2));

    calibScenePoints.push_back(scenePoints[3]);
    imagePoints.push_back(cv::Point2f(marker_position[4].x, marker_position[4].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[4].x, marker_position[4].y, 3));

    calibScenePoints.push_back(scenePoints[4]);
    imagePoints.push_back(cv::Point2f(marker_position[5].x, marker_position[5].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[5].x, marker_position[5].y, 4));

    calibScenePoints.push_back(scenePoints[5]);
    imagePoints.push_back(cv::Point2f(marker_position[6].x, marker_position[6].y));
    imagePoints_aux.push_back(cv::Point3f(marker_position[6].x, marker_position[6].y, 5));


    cv::Mat H = cv::findHomography(calibScenePoints, imagePoints, cv::RANSAC, 5);
    cv::Mat fout;
    cv::Mat fout_fixed;
    std::vector<cv::Point3f> drawnPoints; // Here we save the final points with index
    M1.copyTo(fout);
    M1.copyTo(fout_fixed);

    std::map<int, std::vector<cv::Point2i> > objectiveImPos;
    for(int i=1; i<= 9; i++) {
        addPoint(fout, scenePoints[i-1], H, i, drawnPoints);
    }

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Homografia", fout);
        cv::waitKey(0);
    #endif

    for(int i=1; i<= 9; i++) {
        cv::Point2f sp = scenePoints[i-1];
        std::vector<cv::Point2i> square;
        float marker_size = 7.5;
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y-marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x+marker_size, sp.y+marker_size), H));
        square.push_back(getPoint(cv::Point2f(sp.x-marker_size, sp.y+marker_size), H));
        objectiveImPos[i] = square;
    }

    //===============================================================================================//

    // We make a filter by color at the mask
    cv::Mat ext_global_mask_filtered = ext_global_mask.clone();
    cv::Vec3b color_ref = findPredominantColor(M1m, imagePoints[1].x, imagePoints[1].y, 15);

    #ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Predominant Color: " << color_ref << std::endl;
    #endif

    for (int y = ext_global_mask_filtered.rows/4; y < ext_global_mask_filtered.rows; ++y) {
        for (int x = 0; x < ext_global_mask_filtered.cols; ++x) {
            if (ext_global_mask_filtered.at<uchar>(y, x) == 0) { 
                if (!compareColorsAt(M1m, x, y, color_ref, 30)) {
                    ext_global_mask_filtered.at<uchar>(y, x) = 255;
    }}}}
    //===============================================================================================//

    // New mask with with lines between the real mark and the correction
    cv::Mat visualizedMask;
    cv::cvtColor(ext_global_mask_filtered, visualizedMask, cv::COLOR_GRAY2BGR);
    addFixed_marks(imagePoints_aux, drawnPoints, fout_fixed, visualizedMask);
    //===============================================================================================//

    // We are gonna find the positions of the 3 points who the algoritm doesnt calculate
    std::vector<cv::Point3f> selectedPoints;
    std::vector<cv::Point> closestBlackPoints;
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat maskClone = ext_global_mask_filtered.clone();
    cv::Mat drawing = cv::Mat::zeros(ext_global_mask_filtered.size(), CV_8UC3);

    std::vector<int> ids;
    ids.push_back(9);
    ids.push_back(8);
    ids.push_back(7);

    selectedPoints.push_back(drawnPoints[6]);
    selectedPoints.push_back(drawnPoints[7]);
    selectedPoints.push_back(drawnPoints[8]);

    findClosestBlackPoints(ext_global_mask_filtered, selectedPoints, closestBlackPoints, visualizedMask, marker_availability);
    //===============================================================================================//

    // We are gonna findContours on a defined rectangle in the image for efficiency
    cv::Point3f point0, point1, point2, point3, point4, point5, point6, point7, point8;
    for (const auto &point : imagePoints_aux) { // We need to identify each point for error management
        if (point.z == 0){point0 = point;}
        if (point.z == 1){point1 = point;}
        if (point.z == 2){point2 = point;}
        if (point.z == 3){point3 = point;}
        if (point.z == 4){point4 = point;}
        if (point.z == 5){point5 = point;}
        if (point.z == 6){point6 = point;}
        if (point.z == 7){point7 = point;}
        if (point.z == 8){point8 = point;}}   

    int verticalDistance = point1.y - point4.y;
    int height = (verticalDistance * 3) / 4;
    int x = point2.x;
    int y = point1.y;
    int x2 = point0.x;
    int y2 = point4.y - height;
    cv::Rect squareROI(x - 100, y2 , x2-x + 220, y-y2 + 50); //we add some slack
    cv::rectangle(drawing, squareROI, cv::Scalar(255));

    cv::Mat squareMask = maskClone(squareROI);
    cv::findContours(squareMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        for (size_t j = 0; j < contours[i].size(); j++) {
            contours[i][j].x += x - 100;
            contours[i][j].y += y2;
    }}

    for (size_t i = 0; i < contours.size(); i++) {
        double contourArea = cv::contourArea(contours[i]);
        if (contourArea > 50 && contourArea < 100000) {
            cv::Moments m = cv::moments(contours[i]);
            if (m.m00 > 0) {
                int cx = static_cast<int>(m.m10 / m.m00);
                int cy = static_cast<int>(m.m01 / m.m00);
                cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                cv::drawContours(drawing, contours, static_cast<int>(i), color, 2, cv::LINE_8);
                std::string areaText = std::to_string(contourArea);
                cv::putText(drawing, areaText, cv::Point(cx, cy), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                cv::circle(drawing, cv::Point(cx, cy), 2, color, -1);
    }}}

    #ifdef SHOW_TEST
        cv::imshow("Contornos", drawing);
        cv::waitKey(0);
    #endif
    //===============================================================================================//

    // We are gonna add fixed 7,8,9 points to the image
    std::vector<cv::Point2i> centroids;

    for (const auto& contour : contours) {
        double contourArea = cv::contourArea(contour);
        cv::Moments m = cv::moments(contour);
        if (contourArea > 50 && contourArea < 100000){
            for (const auto& blackpoint : closestBlackPoints) {
                if (cv::pointPolygonTest(contour, blackpoint, false) >= 0) {
                    double area = cv::contourArea(contour);
                    cv::drawContours(visualizedMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 255, 0), 2);
                    if (m.m00 > 0) {
                        int cx = static_cast<int>(m.m10 / m.m00);
                        int cy = static_cast<int>(m.m01 / m.m00);
                        centroids.push_back(cv::Point2i(cx, cy));}
                    break;
    }}}}

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Mascara con correciones", visualizedMask);
        cv::waitKey(0);
    #endif

    // We will order the points from left to right
    std::sort(centroids.begin(), centroids.end(), [](const cv::Point2i& a, const cv::Point2i& b) {
        return a.x < b.x;
    });

    std::vector<cv::Point3f> fixed_final_marks;
    for (size_t i = 0; i < centroids.size(); ++i) {
        addMarker(fout_fixed, centroids[i], ids[i]);
        fixed_final_marks.push_back(cv::Point3f(centroids[i].x, centroids[i].y, static_cast<float>(ids[i])));
    }

    for (const auto& punto : imagePoints_aux) {fixed_final_marks.push_back(cv::Point3f(punto.x, punto.y, static_cast<float>(punto.z+1)));}

    std::sort(fixed_final_marks.begin(), fixed_final_marks.end(), 
              [](const cv::Point3f& a, const cv::Point3f& b) {
                  return a.z < b.z;
              });


    // Lets save the final points with contour
    std::vector<PointWithContour> final_points_contour;

    for (size_t i = 0; i < fixed_final_marks.size(); ++i) {
        cv::Point punto_aux(fixed_final_marks[i].x, fixed_final_marks[i].y);
        for (size_t j = 0; j < contours.size(); ++j) {
            double distancia = cv::pointPolygonTest(contours[j], punto_aux, true);
            if (distancia >= 0) {
                PointWithContour puntoConContorno;
                puntoConContorno.punto = fixed_final_marks[i];
                puntoConContorno.indiceContorno = j;
                puntoConContorno.contorno = contours[j];
                final_points_contour.push_back(puntoConContorno);
                break;
            }
        }
    }

    // if we not find any mark we just return the mark created by the homography
    std::vector<int> markerIndices = {7, 8, 9};

    for (int index : markerIndices) {
        if (!marker_availability[index]) {
            std::vector<cv::Point> rectangle = addPoint_2(fout_fixed, scenePoints[index - 1], H, index);
            PointWithContour puntoHomografia;
            puntoHomografia.punto.x = drawnPoints[index - 1].x;
            puntoHomografia.punto.y = drawnPoints[index - 1].y;
            puntoHomografia.punto.z = index;
            puntoHomografia.indiceContorno = 20 + index;
            puntoHomografia.contorno = rectangle;
            final_points_contour.push_back(puntoHomografia);
    }}

    // Lets draw the final image
    cv::Mat fout_final;
    M1.copyTo(fout_final);
    for (const PointWithContour& point : final_points_contour) {
        cv::drawContours(fout_final, std::vector<std::vector<cv::Point>>(1, point.contorno), 0, cv::Scalar(0, 255, 255), 2);
        cv::circle(fout_final, cv::Point(point.punto.x, point.punto.y), 3, cv::Scalar(0, 0, 255));
        std::string idText = std::to_string(static_cast<int>(point.punto.z));
        cv::putText(fout_final, idText,
                    cv::Point(point.punto.x + 5, point.punto.y - 5), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    cv::Scalar(0, 255, 0));
    }

    #ifdef SHOW_TEST
        for (const auto& punto : final_points_contour) {
            std::cout << "Punto ["<< punto.punto.z << "]: ubicado en (" << punto.punto.x << ", " << punto.punto.y << ")" << " pertenece al contorno " << punto.indiceContorno << std::endl;
        }
    #endif

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Fixed sin contornos", fout_fixed);
        cv::waitKey(0);
    #endif

    #ifdef SHOW_MAIN_RESULTS
        cv::imshow("Fixed con contornos", fout_final);
        cv::waitKey(0);
    #endif

    #ifdef SHOW_MAIN_RESULTS
        std::chrono::steady_clock::time_point eend1 = std::chrono::steady_clock::now();
        std::cout << "Calibrated Scene time = " << std::chrono::duration_cast<std::chrono::seconds>(eend1 - ebegin).count() << "[s]" << std::endl;
        cv::imwrite("Full calibration.png", fout);
    #endif

    ImagemConverter Converter = ImagemConverter();
    std::string H_base64 = Converter.mat2str(H);

    return std::tuple<int, std::string, std::vector<PointWithContour>, std::string, int, int> (0, "{\"state\":\"success\"}", final_points_contour, H_base64, calib_w, calib_h);

    }catch(std::string JSON){
        std::string H_base64;
        std::vector<PointWithContour> final_points_contour;
        int calib_w; int calib_h;
        return std::tuple<int, std::string, std::vector<PointWithContour>, std::string, int, int> (-1, "{\"state\":\"error\"}", final_points_contour, H_base64, calib_w, calib_h);
    }
}