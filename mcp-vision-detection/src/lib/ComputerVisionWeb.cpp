#include "ComputerVisionWeb.h"

ComputerVisionWeb::ComputerVisionWeb()
{
}

void ComputerVisionWeb::setScenePoints(std::vector<cv::Point2f> &scenePoints)
{
    // The nine scene points
    scenePoints.resize(9);
    cv::Point2f p;
    p.x = 141.421356237;
    p.y = 141.421356237;
    scenePoints[0] = p; // Position 1
    p.x = 0;
    p.y = 200;
    scenePoints[1] = p; // Position 2
    p.x = -141.421356237;
    p.y = 141.421356237;
    scenePoints[2] = p; // Position 3
    p.x = 200;
    p.y = 0;
    scenePoints[3] = p; // Position 4
    p.x = 0;
    p.y = 0;
    scenePoints[4] = p; // Position 5
    p.x = -200;
    p.y = 0;
    scenePoints[5] = p; // Position 6
    p.x = 141.421356237;
    p.y = -141.421356237;
    scenePoints[6] = p; // Position 7
    p.x = 0;
    p.y = -200;
    scenePoints[7] = p; // Position 8
    p.x = -141.421356237;
    p.y = -141.421356237;
    scenePoints[8] = p; // Position 9
}

cv::Point2i ComputerVisionWeb::transform(cv::Point2f p)
{
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0, 0) = p.x;
    pin.at<double>(1, 0) = p.y;
    pin.at<double>(2, 0) = 1;

    cv::Mat pout = pin;

    return cv::Point2i(rint(pout.at<double>(0, 0) / pout.at<double>(2, 0)),
                       rint(pout.at<double>(1, 0) / pout.at<double>(2, 0)));
}

cv::Point2i ComputerVisionWeb::getPoint(cv::Point2f p)
{
    // Get center in image coordinates
    return transform(p);
}

namespace
{
    std::size_t callback(const char *in, std::size_t size, std::size_t num, std::string *out)
    {
        const std::size_t totalBytes(size * num);
        out->append(in, totalBytes);
        return totalBytes;
    }
}

size_t writeData(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

bool downloadFile(const std::string &url, const std::string &outFilename)
{
    CURL *curl;
    FILE *fp;
    CURLcode res;
    curl = curl_easy_init();
    if (curl)
    {
        fp = fopen(outFilename.c_str(), "wb");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeData);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
        return (res == CURLE_OK);
    }
    return false;
}

void downloadMedia(const std::string &videoUrl, const std::string &imageUrl)
{
    std::string videoFilename = "/usr/src/app/mcp-vision-detection/video.mp4";
    std::string imageFilename = "/usr/src/app/mcp-vision-detection/bg.jpg";

    if (downloadFile(videoUrl, videoFilename))
    {
        std::cout << "Video downloaded successfully." << std::endl;
    }
    else
    {
        std::cout << "Failed to download video." << std::endl;
    }

    if (downloadFile(imageUrl, imageFilename))
    {
        std::cout << "Image downloaded successfully." << std::endl;
    }
    else
    {
        std::cout << "Failed to download image." << std::endl;
    }
}

std::vector<MarkAndTime> parseSimpleJson(const std::string &jsonString) {
    std::vector<MarkAndTime> marks;
    auto json = nlohmann::json::parse(jsonString);

    for (const auto& item : json) {
        MarkAndTime mark;
        mark.mark_correct = item["mark_correct"].get<int>();
        mark.frame = item["frame"].get<int>();
        marks.push_back(mark);
    }

    return marks;
}

std::string toJSON(const std::vector<Section>& sections) {
    std::string json = "[\n";
    for(size_t i = 0; i < sections.size(); ++i) {
        const Section& sec = sections[i];
        json += "    {\n";
        json += "    \"id_sequence\": " + std::to_string(i) + ",\n";
        json += "    \"takeoff_frame\": " + std::to_string(sec.takeoff_frame) + ",\n";
        json += "    \"arrival_frame\": " + std::to_string(sec.arrival_frame) + ",\n";
        json += "    \"error\": " + std::string(sec.error ? "true" : "false") + "\n";
        json += "    }";
        if (i < sections.size() - 1) json += ",";
        json += "\n";
    }
    json += "]";

    std::string json2 = "[\n";
    for(size_t i = 0; i < sections.size(); ++i) {
        const Section& sec = sections[i];
        json2 += "    {\n";
        json2 += "    \"id_sequence\": " + std::to_string(i) + ",\n";
        json2 += "    \"takeoff_frame\": " + std::to_string(sec.takeoff_frame) + ",\n";
        json2 += "    \"arrival_frame\": " + std::to_string(sec.arrival_frame) + ",\n";
        json2 += "    \"error\": " + std::string(sec.error ? "true" : "false") + ",\n";
        json2 += "    \"items\": [\n";
        for (size_t j = 0; j < sec.items.size(); ++j) {
            const item& it = sec.items[j];
            json2 += "        {\n";
            json2 += "        \"code\": " + std::to_string(it.code) + ",\n";
            json2 += "        \"intersects\": " + std::to_string(it.intersects) + ",\n";
            json2 += "        \"frame\": " + std::to_string(it.frame) + ",\n";
            json2 += "        \"d_l\": " + std::to_string(it.d_l) + ",\n";
            json2 += "        \"d_r\": " + std::to_string(it.d_r) + ",\n";
            json2 += "        \"step_l\": " + std::string(it.step_l ? "true" : "false") + ",\n";
            json2 += "        \"step_r\": " + std::string(it.step_r ? "true" : "false") + "\n";
            json2 += "        }";
            if (j < sec.items.size() - 1) json2 += ",";
            json2 += "\n";
        }
        json2 += "    ]\n";
        json2 += "    }";
        if (i < sections.size() - 1) json2 += ",";
        json2 += "\n";
    }
    json2 += "]";

    std::cout << "JSON NUEVO \n\n" << json2 << std::endl;

    return json;
}

// Per frame: Two feet. By foot: (x y w h code xp yp d)
//  (x,y,w,h): foot rect                (left_step, right_step)
//  code:                               (in_objective1, in_objective2)
//      0: No step
//    1-9: Step to nearest objective
//  (xp,yp): Feet contact point         (left_foot, right_foot)
//  d: distance to nearest center       (odist1, odist2)
std::string ComputerVisionWeb::buildFinalOutputFinal(FeetTracker &ft, std::vector<MarkAndTime> sequence, int maxFrame) {
    //in_objective1, in_objective2
    //odist1, odist2
    
    // for (int i = 0; i < maxFrame; ++i) {
    //     std::cout << "Frame Index: " << i << "\n\tLeft: " << ft.in_objective1[i] << "\n\tRight: " << ft.in_objective2[i] << std::endl;
    // }
    
    //Get central stimulus central position
    cv::Point2f pcentral = ft.contourCentersScene[4];
    
    
    const int relevant_change = 10; //Number of centimeters for considering relevant change in position
    const int static_step = 15; //Number of frames for considering that step is not displacing
    
    int current_seq = 0, //index for starting current sequence
        current_center_exit1 = 0, current_center_exit2 = 0, //index for exiting center on current sequence for each foot
        next_seq = 0;  //index for starting next sequence
    cv::Point p_out1, p_out2;
    
    //Variables for stimuli sequence:
    uint n_objectives = sequence.size();
    int i, cur_objective, cur_frame;

    //Divide items in stimuli sequence data and calculate frames de despegue y llegada
    // Lista para almacenar los resultados
    std::vector<Section> sequences;
    bool first_stimulus = true;
    
    //Get intervals per objective:
    for (int j = 0; j < n_objectives; ++j) {
        Section cur_section;
        item cur_item;
        
        cur_objective = sequence[j].mark_correct; 
        cur_frame = sequence[j].frame;

#ifdef SHOW_DEBUG_TEXT
        std::cout << "Marking.\n\tCurrent stimulus: " << j << std::endl;
        std::cout << "\tCurrent stimulus objective: " << cur_objective << std::endl;
        std::cout << "\tCurrent stimulus index: " << cur_frame << std::endl;
#endif        
        //Booleans for marking errors (assume right first):
        bool step_center = true, step_objective = true, right_objective = true; 
        
        //Advance until both are near the 5 zone (assume that the player can be late):


        for (i = current_seq; i < maxFrame; ++i)
            if(ft.in_objective1[i] == 5 || ft.in_objective2[i] == 5)
                break;
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tCurrent sequence start - prev cur_frame: " << current_seq << std::endl;
#endif        
        
        //Update start of current seq: if stimulus presentation is higher than presence in zone 5, start from stimulus presentation frame
        current_seq = (cur_frame >= i)? cur_frame : i;

#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tCurrent sequence start - after cur_frame: " << current_seq << std::endl;
#endif        

        
        //Search for center position exit, considered as the first step out of center zone:
        bool step_out_detected1 = false, step_out_detected2 = false;
        int sure_frame1, sure_frame2;

        //Search for left exit:
        for (i = current_seq; i < maxFrame; ++i) {
            if(ft.left_step[i] == 0) //Continue until a step is detected
                continue;
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tCurrent frame - search left position exit: " << i << std::endl;
#endif        
            //Left foot steps out:
            if(ft.in_objective1[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tExit frame left - position exit: " << i << std::endl;
        std::cout << "\tExit frame left - code found: " << ft.in_objective1[i] << std::endl;
#endif        
                current_center_exit1 = i;
                p_out1 = ft.left_foot[i];
                sure_frame1 = i;
                step_out_detected1 = true;
                break;
            }            
        }

        //Search for right exit:
        for (i = current_seq; i < maxFrame; ++i) {
            if(ft.right_step[i] == 0) //Continue until a step is detected
                continue;
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tCurrent frame - search right position exit: " << i << std::endl;
#endif        
            //Right foot steps out:
            if(ft.in_objective2[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tExit frame right - position exit: " << i << std::endl;
        std::cout << "\tExit frame right - code found: " << ft.in_objective2[i] << std::endl;
#endif        
                current_center_exit2 = i;
                p_out2 = ft.right_foot[i];
                sure_frame2 = i;
                step_out_detected2 = true;
                break;
            }            
        }

        
        
        if(step_out_detected1 || step_out_detected2) { //It shall be detected... if not maybe end of stimuli sequence or player skip some stimuli
            //Check which is the first leg going on the objective direction
            if(!step_out_detected1) { //Give a reference the second sure step if first not found
                p_out1 = p_out2;
                current_center_exit1 = current_center_exit2;
                sure_frame1 = sure_frame2;
            }
            if(!step_out_detected2) { //Give a reference the second sure step if first not found
                p_out2 = p_out1;
                current_center_exit2 = current_center_exit1;
                sure_frame2 = sure_frame1;
            }
            
            int lindex_1, lindex_2, rindex_1, rindex_2, il_found, ir_found, il_last = current_seq+1, ir_last = current_seq+1, il_initial, il_last_stepping, ir_initial, ir_last_stepping;
            bool stepping = false, first = true, pl_found = false, pr_found = false;
            cv::Point2f p1, p2, p_out_s = ft.imageToScene(p_out1), p_center = ft.contourCentersScene[4]; //Take central point as reference
            float d1, d2, d, 
                  d_center_obj = sqrt((p_out_s.x - p_center.x)*(p_out_s.x - p_center.x) + (p_out_s.y - p_center.y)*(p_out_s.y - p_center.y));
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tProcessing take-off left..." << std::endl;
                std::cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << std::endl;
                std::cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << std::endl;
#endif

            //Check backwards first relevant change in left step keeping going far sure feet and near center
            for (i = sure_frame1; i >= current_seq; --i) {
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off - left - frame: " << i << std::endl;
#endif                        
                if(first) { //Search for end of first stepping
                    if(!stepping && ft.left_step[i] == 1) { //A step
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - left - start stepping... " << std::endl;
#endif                        
                        stepping = true;
                        il_last = i;
                        il_initial = i;
                        lindex_1 = i;
                    } else if(stepping && ft.left_step[i] == 0) { //Is stepping, so check if it stops doing so
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - left - stop stepping... " << std::endl;
#endif                  
                        il_last_stepping = i-1;      
                        stepping = false;
                        first = false; //First index ready
                    }
                } else { //Search for first of following step
                    if(ft.left_step[i] == 1) { //First position of next step
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - left - left following found... " << std::endl;
#endif        
                        il_last = i;
                        lindex_2 = i;
                        //Significant displacement criterion
                        p1 = ft.imageToScene(ft.left_foot[lindex_1]);
                        p2 = ft.imageToScene(ft.left_foot[lindex_2]);
                        d  = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
                        //d1 = sqrt((p_out_s.x - p1.x)*(p_out_s.x - p1.x) + (p_out_s.y - p1.y)*(p_out_s.y - p1.y));//L2 norm
                        //d2 = sqrt((p_out_s.x - p2.x)*(p_out_s.x - p2.x) + (p_out_s.y - p2.y)*(p_out_s.y - p2.y));//L2 norm
                        d1 = magnitude(projectVector(cv::Point2f(p1.x - p_out_s.x, p1.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                        d2 = magnitude(projectVector(cv::Point2f(p2.x - p_out_s.x, p2.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                        
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << std::endl;
                        std::cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << std::endl;
                        std::cout << "\tFoot 1 (x,y): " << p1.x << ", " << p1.y << " at index " << lindex_1 << std::endl;
                        std::cout << "\tFoot 2 (x,y): " << p2.x << ", " << p2.y << " at index " << lindex_2 << std::endl;                        
                        std::cout << "\tTake-off - left - distance between steps: " << d << std::endl;
                        std::cout << "\tTake-off - left - projected distance between 1st step and objective: " << d1 << std::endl;
                        std::cout << "\tTake-off - left - projected distance between 2nd step and objective: " << d2 << std::endl;
                        std::cout << "\tTake-off - left - distance between center and objective: " << d_center_obj << std::endl;
#endif
                        if(d1 < d2 && d > relevant_change && il_initial - il_last_stepping < static_step && d2 < d_center_obj && ft.odist1[i] != 0) { //While the step is approaching to the objective and not farther than center, keep searching...
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - left - keeps approaching objective... " << std::endl;
#endif
                            stepping = true;                            
                            first = true; //Consider as first again
                            lindex_1 = lindex_2;
                            il_initial = lindex_1;
                            continue;
                        }
                                
                        //If we reach here, it means that p1 is the take-off step
                        pl_found = true;
          
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - left - il initial:" << il_initial << std::endl;
                        std::cout << "\tTake-off - left - il last stepping:" << il_last_stepping << std::endl;
                        std::cout << "\tTake-off - left - stepping diff:" << il_initial - il_last_stepping << std::endl;
#endif                        
                        
                        if(il_initial - il_last_stepping >= static_step) { //Static step, so this is considered the take-off step
                            il_found = il_initial + 1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - left - static step: Found at " << il_found << "." << std::endl;
#endif
                            break;
                        }
                        
                        if(d <= relevant_change) { // If two little steps, assume second is the take_off
                            il_found = lindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - left - irrelevant step distance: Found at " << il_found << "." << std::endl;
#endif        

                            break;
                        }
                        
                        if(d1 > d2) { //If p1 is farther than p2, it is the take-off step
                            il_found = il_initial + 1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - left - first step farther than second: Found at " << il_found << ". " << std::endl;
#endif
                        } else { //Else, p2 is.                            
                            il_found = lindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - left - second step farther than first: Found at " << il_found << ". " << std::endl;
#endif
                        }
                        break;
                    }
                }
            }
            //If left not found, take last frame stepping as take-off:
            if(!pl_found) {
                il_found = il_last + 1;
                pl_found = true;
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off - left - not found so take last step: Found at " << il_found << ". " << std::endl;
#endif
            }

            //Now check first relevant change in right step
            stepping = false; 
            first = true;
            p_out_s = ft.imageToScene(p_out2);
            d_center_obj = sqrt((p_out_s.x - p_center.x)*(p_out_s.x - p_center.x) + (p_out_s.y - p_center.y)*(p_out_s.y - p_center.y));
#ifdef SHOW_DEBUG_TEXT
            std::cout << "\tProcessing take-off right..." << std::endl;
            std::cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << std::endl;
            std::cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << std::endl;
#endif
            for (i = sure_frame2; i >= current_seq; --i) {
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off - right - frame: " << i << std::endl;
#endif                        
                if(first) { //Search for end of first stepping
                    if(!stepping && ft.right_step[i] == 1) { //A step
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - right - start stepping... " << std::endl;
#endif                        
                        stepping = true;
                        ir_last = i;
                        ir_initial = i;
                        rindex_1 = i;
                    } else if(stepping && ft.right_step[i] == 0) { //Is stepping, so check if it stops doing so
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - right - stop stepping... " << std::endl;
#endif                  
                        ir_last_stepping = i+1;
                        stepping = false;
                        first = false; //First index ready
                    }
                } else { //Search for first of following step
                    if(ft.right_step[i] == 1) { //First position of next step
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - right - right following found... " << std::endl;
#endif        
                        ir_last = i;
                        rindex_2 = i;
                        //Significant displacement criterion
                        p1 = ft.imageToScene(ft.right_foot[rindex_1]);
                        p2 = ft.imageToScene(ft.right_foot[rindex_2]);
                        d  = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
//                        d1 = sqrt((p_out_s.x - p1.x)*(p_out_s.x - p1.x) + (p_out_s.y - p1.y)*(p_out_s.y - p1.y));//L2 norm
//                        d2 = sqrt((p_out_s.x - p2.x)*(p_out_s.x - p2.x) + (p_out_s.y - p2.y)*(p_out_s.y - p2.y));//L2 norm
                        d1 = magnitude(projectVector(cv::Point2f(p1.x - p_out_s.x, p1.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                        d2 = magnitude(projectVector(cv::Point2f(p2.x - p_out_s.x, p2.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));

#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << std::endl;
                        std::cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << std::endl;
                        std::cout << "\tFoot 1 (x,y): " << p1.x << ", " << p1.y << " at index " << rindex_1 << std::endl;
                        std::cout << "\tFoot 2 (x,y): " << p2.x << ", " << p2.y << " at index " << rindex_2 << std::endl;   
                        std::cout << "\tTake-off - right - distance between steps: " << d << std::endl;
                        std::cout << "\tTake-off - right - projected distance between 1st step and objective: " << d1 << std::endl;
                        std::cout << "\tTake-off - right - projected distance between 2nd step and objective: " << d2 << std::endl;
                        std::cout << "\tTake-off - right - distance between center and objective: " << d_center_obj << std::endl;
#endif
                        if(d1 < d2 && d > relevant_change && ir_initial - ir_last_stepping < static_step && d2 < d_center_obj  && ft.odist2[i] != 0) { //While the step is approaching to the objective and not farther than center, keep searching...
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - right - keeps approaching objective... " << std::endl;
#endif
                            stepping = true;                            
                            first = true; //Consider as first again
                            rindex_1 = rindex_2;
                            ir_initial = rindex_1;
                            continue;
                        }
                                
                        //If we reach here, it means that p1 is the take-off step
                        pr_found = true;
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - right - ir initial:" << ir_initial << std::endl;
                        std::cout << "\tTake-off - right - ir last stepping:" << ir_last_stepping << std::endl;
                        std::cout << "\tTake-off - right - stepping diff:" << ir_initial - ir_last_stepping << std::endl;
#endif        
                        
                        if(ir_initial - ir_last_stepping >= static_step) { //Static step, so this is considered the take-off step
                            ir_found = ir_initial + 1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - right - static step: Found at " << ir_found << "." << std::endl;
#endif        
                            break;
                        }
                        
                        if(d <= relevant_change) { // If two little steps, assume second is the take_off
                            ir_found = rindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - right - irrelevant step distance: Found at " << ir_found << "." << std::endl;
#endif        
                            break;
                        }
                        
                        if(d1 > d2) { //If p1 is farther than p2, it is the take-off step
                            ir_found = ir_initial+1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - right - first step farther than second: Found at " << ir_found << "." << std::endl;
#endif
                        } else { //Else, p2 is.                            
                            ir_found = rindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - right - second step farther than first: Found at " << ir_found << "." << std::endl;
#endif
                        }
                        break;
                    }
                }
            }
            //If left not found, take last frame stepping as take-off:
            if(!pr_found) {
                ir_found = ir_last + 1;
                pr_found = true;
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off - right - not found so take last step: Found at " << ir_found << ". " << std::endl;
#endif
            }
            
            //Set take-off frame:
            if(pl_found && pr_found) {
                if(il_found < ir_found) { //Take-off is from left
                    cur_item.code = 5;
                    cur_item.d_l = ft.odist1[il_found];
                    cur_item.d_r = ft.odist2[ir_found];
                    cur_item.step_l = true;
                    cur_item.step_r = false;
                    cur_item.frame = il_found;
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = il_found;
#ifdef SHOW_DEBUG_TEXT
                    std::cout << "\tTake-off: Both found - decided left" << std::endl;
#endif
                } else {
                    cur_item.code = 5;
                    cur_item.d_l = ft.odist1[il_found];
                    cur_item.d_r = ft.odist2[ir_found];
                    cur_item.step_l = false;
                    cur_item.step_r = true;
                    cur_item.frame = ir_found;                    
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = ir_found;
#ifdef SHOW_DEBUG_TEXT
                    std::cout << "\tTake-off: Both found - decided right" << std::endl;
#endif
                }
            } else if(pl_found) {
                cur_item.code = 5;
                cur_item.d_l = ft.odist1[il_found];
                cur_item.d_r = ft.odist2[il_last];
                cur_item.step_l = true;
                cur_item.step_r = false;
                cur_item.frame = il_found;
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = il_found;
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off: One found - left" << std::endl;
#endif
            } else if (pr_found) {
                cur_item.code = 5;
                cur_item.d_l = ft.odist1[il_found];
                cur_item.d_r = ft.odist2[ir_found];
                cur_item.step_l = false;
                cur_item.step_r = true;
                cur_item.frame = ir_found;                                    
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = ir_found;
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off: One found - right" << std::endl;
#endif
            } else { //if none, use max
                cur_item.code = 5;
                cur_item.d_l = ft.odist1[il_found];
                cur_item.d_r = ft.odist2[ir_found];
                cur_item.step_l = il_found;
                cur_item.step_r = ir_found;
                cur_item.frame = il_found<ir_found? il_found : ir_found;                                    
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = il_found<ir_found? il_found : ir_found;
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off: None found" << std::endl;
#endif
            }
#ifdef SHOW_DEBUG_TEXT
            std::cout << "\tTake-off: " << cur_item.frame << std::endl;
            std::cout << "\tTake-off - left?: " << cur_item.step_l << std::endl;
#endif
        //end if step_out_detected
        } else { 
#ifdef SHOW_DEBUG_TEXT
            std::cout << "\tNo step out found. " << std::endl;
#endif

            break; //No step out, means irrelevant rest of video
        }
        
        //Search for arrival info
        int last_arrival_frame = -1; //Value is -1 if no one steps, and the corresponding frame if it steps
        int nearest_arrival_code; //Stepping or not, it is the nearest arrival code
        int last_real = -1;
        float nearest_arrival_distance = FLT_MAX;
        bool still_near_center_l = true, still_near_center_r = true, real_arrival = false, is_left = true;
        for (int i = current_center_exit1<current_center_exit2?current_center_exit1:current_center_exit2; i < maxFrame; ++i) {
#ifdef SHOW_DEBUG_TEXT
            std::cout << "\tArrival - frame: " << i << std::endl;
#endif            
            //There might still be one of the feet near center
            if(ft.in_objective1[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tArrival - check left far center... " << std::endl;
#endif            
                still_near_center_l = false;
                if(ft.left_step[i] == 1) { 
#ifdef SHOW_DEBUG_TEXT
                    std::cout << "\tArrival - check left far center step - code: " << ft.in_objective1[i] << std::endl;
                    std::cout << "\tArrival - check left far center step - distance: " << ft.odist1[i] << std::endl;
                    std::cout << "\tArrival - check left far center step - last_real: " << last_real << std::endl;
#endif            
                    if(ft.odist1[i]==0 && last_real != ft.in_objective1[i]) { //Registers the first arrival with this code
                        last_arrival_frame = i;
                        last_real = nearest_arrival_code = ft.in_objective1[i];
                        nearest_arrival_distance = 0;
                        real_arrival = true;
                        is_left = true;
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tArrival - left real arrival found at: " << i << std::endl;
#endif            
                    } 
                    if(!real_arrival && ft.odist1[i] < nearest_arrival_distance) {
                        nearest_arrival_distance = ft.odist1[i];
                        last_arrival_frame = i;
                        nearest_arrival_code = ft.in_objective1[i];
                        is_left = true;
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tArrival - left near arrival found at: " << i << std::endl;
                std::cout << "\tArrival - left near arrival - nearest arrival distance: " << nearest_arrival_distance << std::endl;
#endif            
                    }
                }
            }
            if(ft.in_objective2[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tArrival - check right far center... " << std::endl;
#endif            
                still_near_center_r = false;
                if(ft.right_step[i] == 1) { 
#ifdef SHOW_DEBUG_TEXT
                    std::cout << "\tArrival - check right far center step - code: " << ft.in_objective2[i] << std::endl;
                    std::cout << "\tArrival - check right far center step - distance: " << ft.odist2[i] << std::endl;
                    std::cout << "\tArrival - check right far center step - last_real: " << last_real << std::endl;
#endif            
                    if(ft.odist2[i]==0 && last_real != ft.in_objective2[i]) { //Registers the first arrival with this code
                        last_arrival_frame = i;
                        last_real = nearest_arrival_code = ft.in_objective2[i];
                        nearest_arrival_distance = 0;
                        real_arrival = true;
                        is_left = false;
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tArrival - right real arrival found at: " << i << std::endl;
#endif            
                    } 
                    if(!real_arrival && ft.odist2[i] < nearest_arrival_distance) {
                        nearest_arrival_distance = ft.odist2[i];
                        last_arrival_frame = i;
                        nearest_arrival_code = ft.in_objective2[i];
                        is_left = false;
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tArrival - right near arrival found at: " << i << std::endl;
                        std::cout << "\tArrival - right near arrival - nearest arrival distance: " << nearest_arrival_distance << std::endl;
#endif            
                    }
                }

            }
            
            if(!still_near_center_l && ft.in_objective1[i] == 5 && ft.left_step[i] == 1) { //First step returning to center
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tArrival - left step returning to center found at: " << i << std::endl;
#endif            

                next_seq = i;
                break;
            }
            if(!still_near_center_r && ft.in_objective2[i] == 5 && ft.right_step[i] == 1) { //First step returning to center
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tArrival - right step returning to center found at: " << i << std::endl;
#endif            
                next_seq = i;
                break;
            }
            
        } 
        
        //Set arrival errors and arrival time
        if(!real_arrival || nearest_arrival_code != cur_objective) {
            if(real_arrival && nearest_arrival_code != cur_objective)
                right_objective = false;
            if(!real_arrival)
                step_objective = false; 
        }
        cur_item.code = nearest_arrival_code;
        cur_item.d_l = ft.odist1[last_arrival_frame];
        cur_item.d_r = ft.odist2[last_arrival_frame];
        cur_item.step_l = is_left? 1:0;
        cur_item.step_r = is_left? 0:1;
        cur_item.frame = last_arrival_frame;
        cur_section.items.push_back(cur_item);
        cur_section.arrival_frame = last_arrival_frame;
        cur_section.arrival_code = nearest_arrival_code;

#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tArrival - arrival frame: " << cur_section.arrival_frame << std::endl;
        std::cout << "\tArrival - arrival code: " << cur_section.arrival_code << std::endl;
        if(is_left)
            std::cout << "\tArrival - arrival distance: " << ft.odist1[last_arrival_frame] << std::endl;
        else
            std::cout << "\tArrival - arrival distance: " << ft.odist2[last_arrival_frame] << std::endl;
#endif            

        
        //Check if steps in center on return
        bool still_far_center_l = true, still_far_center_r = true, ready_l = false, ready_r = false;
        step_center = false;
        for (int i = next_seq; i < maxFrame; ++i) {
#ifdef SHOW_DEBUG_TEXT
            std::cout << "\tReturn to center - frame: " << i << std::endl;
#endif            
            //There might still be one of the feet near center
            if(ft.in_objective1[i] == 5) {
                still_far_center_l = false;
                if(ft.left_step[i] == 1) {
#ifdef SHOW_DEBUG_TEXT
            std::cout << "\tReturn to center - found left at: " << i << std::endl;
#endif            
                    step_center = true;                    
                    break;
                }
            }
            if(ft.in_objective2[i] == 5) {
                still_far_center_r = false;
                if(ft.right_step[i] == 1) {
#ifdef SHOW_DEBUG_TEXT
                    std::cout << "\tReturn to center - found right at: " << i << std::endl;
#endif            
                    step_center = true;
                    break;
                }
            }
            
            if(!still_far_center_l && ft.in_objective1[i] != 5)
                ready_l = true;
                
            if(!still_far_center_r && ft.in_objective2[i] != 5)
                ready_r = true;
            
            if(ready_l && ready_r) {
#ifdef SHOW_DEBUG_TEXT
            std::cout << "\tReturn to center - both ready at: " << i << std::endl;
#endif            
                break;
            }
        } 
        
        //Set error and store in sequences
        cur_section.error = (right_objective && step_objective && step_center) ? false : true;
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tFinal stimulus error: " << cur_section.error << std::endl;
#endif            

        sequences.push_back(cur_section);
        
        //Set next beggining to next sequence of centers
        current_seq = next_seq;
        first_stimulus = false;
    } //end stimuli sequence
    
    return toJSON(sequences);
}

size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    s->append((char*)contents, newLength);
    return newLength;
}

bool ComputerVisionWeb::callApi(const std::string& videoUrl) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    
    curl = curl_easy_init();
    if(curl) {
        std::string api_url = "http://blazepose-api-local:5000/process-video";
        std::string json_payload = "{\"video_url\": \"" + videoUrl + "\"}";

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            try {
                json responseJson = json::parse(readBuffer);
                json frames = responseJson["frames_info"];

                // Recorrer cada frame y almacenarlo en frames_info
                for (auto& frame : frames) {
                    FrameInfo frameInfo;
                    frameInfo.frame_index = frame["frame_index"];
                    frameInfo.stepDetection = frame["stepDetection"];
                    frameInfo.stepSide = frame["stepSide"];
                    
                    // Parsear la posición izquierda como cv::Point2i
                    frameInfo.left_position.heel = cv::Point2i(frame["left_position"]["heel"][0], frame["left_position"]["heel"][1]);
                    frameInfo.left_position.foot_index = cv::Point2i(frame["left_position"]["foot_index"][0], frame["left_position"]["foot_index"][1]);
                    frameInfo.left_position.ankle = cv::Point2i(frame["left_position"]["ankle"][0], frame["left_position"]["ankle"][1]);
                    frameInfo.left_position.center = cv::Point2i(frame["left_position"]["center"][0], frame["left_position"]["center"][1]);

                    // Parsear la posición derecha como cv::Point2i
                    frameInfo.right_position.heel = cv::Point2i(frame["right_position"]["heel"][0], frame["right_position"]["heel"][1]);
                    frameInfo.right_position.foot_index = cv::Point2i(frame["right_position"]["foot_index"][0], frame["right_position"]["foot_index"][1]);
                    frameInfo.right_position.ankle = cv::Point2i(frame["right_position"]["ankle"][0], frame["right_position"]["ankle"][1]);
                    frameInfo.right_position.center = cv::Point2i(frame["right_position"]["center"][0], frame["right_position"]["center"][1]);

                    // Almacenar en la lista frames_info
                    frames_info.push_back(frameInfo);
                }

            } catch (const std::exception& e) {
                std::cerr << "Error al parsear el JSON: " << e.what() << std::endl;
            }
        }
        curl_easy_cleanup(curl);
        return res == CURLE_OK;
    }
    return false;
}

std::string ComputerVisionWeb::mainFunction(std::string contourjson, std::string videoUrl, std::string imageUrl, std::string jsonString, std::string frameRate) {
    if (callApi(videoUrl)) {
        std::cout << "Procesamiento exitoso, datos recibidos desde la API de pose." << std::endl;
        int frame_count = 0;

        std::cout << "Muestra de los dos primero frames: \n"<< std::endl;
        for (const auto& frame : frames_info) {
            if (frame_count >= 2) {
                break;
            }

            std::cout << "Frame Index: " << frame.frame_index << std::endl;
            std::cout << "Step Detection: " << (frame.stepDetection ? "True" : "False") << std::endl;
            std::cout << "Step Side: " << frame.stepSide << std::endl;
            std::cout << "Left Heel Position: (" << frame.left_position.heel.x << ", " << frame.left_position.heel.y << ")" << std::endl;
            std::cout << "Left Foot Index Position: (" << frame.left_position.foot_index.x << ", " << frame.left_position.foot_index.y << ")" << std::endl;
            std::cout << "Left Ankle Position: (" << frame.left_position.ankle.x << ", " << frame.left_position.ankle.y << ")" << std::endl;
            std::cout << "Right Heel Position: (" << frame.right_position.heel.x << ", " << frame.right_position.heel.y << ")" << std::endl;
            std::cout << "Right Foot Index Position: (" << frame.right_position.foot_index.x << ", " << frame.right_position.foot_index.y << ")" << std::endl;
            std::cout << "Right Ankle Position: (" << frame.right_position.ankle.x << ", " << frame.right_position.ankle.y << ")\n" << std::endl;

            frame_count++;
        }

    } else {
        std::cout << "Error al llamar a la API pose-IA." << std::endl;
    }
    
    // String contornos se debe pasar a std::vector<Contour>
    std::istringstream iss(contourjson);

    Json::Value root;
    iss >> root;

    std::string string_calib_w = std::to_string(root["response"]["calib_w"].asInt());
    std::string string_calib_h = std::to_string(root["response"]["calib_h"].asInt());
    
    std::vector<Contour> contornos;

    for (const auto &item : root["response"]["points"])
    {
        Contour contorno;
        contorno.x = item["x"].asInt();
        contorno.y = item["y"].asInt();
        contorno.z = item["z"].asInt();
        contorno.indiceContorno = item["indiceContorno"].asInt();
        for (const auto &punto : item["contorno"])
        {
            cv::Point2f p{punto["x"].asInt(), punto["y"].asInt()};
            contorno.points.push_back(p);
            contorno.ipoints.push_back(p);
        }
        contornos.push_back(contorno);
    }

    // String sequence se debe pasar a std::vector<MarkAndTime>
    std::vector<MarkAndTime> sequence = parseSimpleJson(jsonString);

    // Video e imagen
    downloadMedia(videoUrl, imageUrl);

    std::string urlVideo = "/usr/src/app/mcp-vision-detection/video.mp4";
    std::string urlBG = "/usr/src/app/mcp-vision-detection/bg.jpg";

    int real_w, real_h;
    int calib_w = std::stoi(string_calib_w);
    int calib_h = std::stoi(string_calib_h);


    cv::VideoCapture vtest;
    vtest.open(urlVideo);

    float frame_rate = 0.0f;
    if (vtest.isOpened())
    {
        frame_rate = std::stof(frameRate);
    }
    else
    {
        std::cout << "El video no abrio!!" << std::endl;
        return "Error al abrir video";
    }

    // std::cout << "Contornos:\n";
    // for (const auto& contorno : contornos) {
    //     std::cout << "Contorno - X: " << contorno.x << ", Y: " << contorno.y << ", Z: " << contorno.z << ", indiceContorno: " << contorno.indiceContorno << "\n";
    //     std::cout << "Puntos:";
    //     for (const auto& punto : contorno.points) {
    //         std::cout << " (" << punto.x << ", " << punto.y << ")";
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << "Sequence:\n";
    for (const auto& markTime : sequence) {
        std::cout << "Mark: " << markTime.mark_correct << ", Time: " << markTime.frame << std::endl;
    }


    std::cout << "URL del video procesado: " << videoUrl << std::endl;
    std::cout << "URL de la imagen de fondo procesada: " << imageUrl << std::endl;
    std::cout << "Frame rate del video: " << frame_rate << std::endl;
    std::cout << "calib_w: " << calib_w << std::endl;
    std::cout << "calib_h: " << calib_h << std::endl;

    cv::Mat current, result, result_big;

    bool first = true;
    uint frame = 0, maxFrame, time = 0, msec_per_frame = 1000 / frame_rate,
        initial_msec = 0, // final_msec = 10000;
        // initial_msec = 0,
        final_msec = INT_MAX;
    cv::Mat fg;

    // NEW: Insert background calibration image to reinforce background
    cv::Mat bg = cv::imread(urlBG);

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("Background", bg);
#endif


    std::map<int, int> msecs;
    while (1)
    {
        vtest >> current;

        if (current.empty())
        {
            break;
        }
        ++frame;


        msecs[frame] = time;
        if (first)
        {
            first = false;
            real_w = current.cols;
            real_h = current.rows;

            if (real_w != bg.cols || real_h != bg.rows)
                cv::resize(bg, bg, current.size());
            std::cout << "Dimensions: " << real_w << "x" << real_h << std::endl;
            std::cout << "Dimensions Calib: " << calib_w << "x" << calib_h << std::endl;

            // We need to scale de points of each contour
            float scaleX = static_cast<float>(real_w) / static_cast<float>(calib_w);
            float scaleY = static_cast<float>(real_h) / static_cast<float>(calib_h);
            for (auto &contorno : contornos)
            {
                for (auto &punto : contorno.points)
                {
                    punto.x = static_cast<int>(punto.x * scaleX);
                    punto.y = static_cast<int>(punto.y * scaleY);
                }
            }
        }
    }
    vtest.release();
    maxFrame = frame;


    // Calibrate scene
    std::vector<cv::Point2f> scenePoints;
    std::map<int, std::vector<cv::Point2i>> objectiveImPos;

    setScenePoints(scenePoints);

    std::map<int, int>::iterator frame_it = msecs.begin();

    vtest.open(urlVideo);
    if (!vtest.isOpened())
    {
        std::cout << "El video no abrio la segunda vez!!" << std::endl;
        return "El video no abrio la segunda vez!!";
    }
    
    std::vector<cv::Point2f> contourCenters;
    float xx,yy;
    int n;
  
    //Get contour centers:
    for (const auto &c : contornos) {
        xx = yy = 0;
        n = 0;
        for (const auto &p : c.points) {
            xx += p.x;
            yy += p.y;
            ++n;
        }
        contourCenters.push_back(cv::Point2f(xx/n, yy/n));
    }

    for (int i = 1; i <= 9; i++)
    {
        cv::Point2f sp = scenePoints[i - 1];
        std::vector<cv::Point2i> square;
        square.push_back(getPoint(cv::Point2f(sp.x - 7.5, sp.y - 7.5)));
        square.push_back(getPoint(cv::Point2f(sp.x + 7.5, sp.y - 7.5)));
        square.push_back(getPoint(cv::Point2f(sp.x + 7.5, sp.y + 7.5)));
        square.push_back(getPoint(cv::Point2f(sp.x - 7.5, sp.y + 7.5)));
        objectiveImPos[i] = square;
    }

#ifdef MEMORY_DEBUG
    std::cerr << "End calibration init...\n\nStart step processing..." << std::endl;
#endif

    uint j_cur = 0, n_objectives = sequence.size();
    int cur_objective = sequence[0].mark_correct;

    for (uint i = 1; i <= maxFrame; ++i)
    {
        frame = frame_it->first;
#ifdef MEMORY_DEBUG
        std::cerr << "\tStep processing - Frame: " << frame << std::endl;
#endif

#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Frame: " << frame << std::endl;
        std::cout << "Time: " << frame_it->second << " [msecs]" << std::endl;
#endif
        vtest >> current;
#ifdef SHOW_INTERMEDIATE_RESULTS
        current.copyTo(cur_copy);
        cv::rectangle(cur_copy, gt_bboxes[frame], cv::Scalar(0, 255, 255), 1);
        // cv::resize(cur_copy, cur_copy, cv::Size(3*current.cols,3*current.rows));
        cv::imshow("Current Image", cur_copy);
#endif
        //Get currently active objective
        for (uint j = j_cur; j < n_objectives; ++j) {
            MarkAndTime &m = sequence[j];
            uint oframe = m.frame;
            // std::cout << "Frame: " << frame << "; OFrame: " << oframe << std::endl;
            if(frame < oframe) 
                break;
            int cobjective = m.mark_correct;
            cur_objective = cobjective;
            j_cur = j;
        
        }
        
#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Current Objective: " << cur_objective << std::endl;
#endif        
        frame_it++;
    }

    vtest.release();

#ifdef MEMORY_DEBUG
    std::cerr << "End step processing...\n\nStart step completion..." << std::endl;
#endif

#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "Last processed frame: " << frame << std::endl;
    std::cout << "Max frame: " << frame << std::endl;
#endif

    std::string out = buildFinalOutputFinal(ft, sequence, maxFrame);

#ifdef SHOW_FINAL_RESULTS
    std::cout << "============ OUT ============ \n" << out << std::endl;
#endif

#ifdef MEMORY_DEBUG
    std::cerr << "End step completion..." << std::endl;
#endif

    return out;
}