#include "ComputerVisionWeb.h"

ComputerVisionWeb::ComputerVisionWeb()
{
}
namespace
{
    size_t callback(const char *in, size_t size, size_t num, string *out)
    {
        const size_t totalBytes(size * num);
        out->append(in, totalBytes);
        return totalBytes;
    }
}

cv::Point2f ComputerVisionWeb::imageToScene(cv::Point2i p) {
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0,0) = p.x*this->calib_w/this->real_w;
    pin.at<double>(1,0) = p.y*this->calib_h/this->real_h;
    pin.at<double>(2,0) = 1;
    cv::Mat pout = this->H*pin;
    return cv::Point2f(pout.at<double>(0,0)/pout.at<double>(2,0),
                       pout.at<double>(1,0)/pout.at<double>(2,0));
}

size_t writeData(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

bool downloadFile(const string &url, const string &outFilename)
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

void downloadMedia(const string &videoUrl, const string &imageUrl)
{
    string videoFilename = "/usr/src/app/mcp-vision-detection/video.mp4";
    string imageFilename = "/usr/src/app/mcp-vision-detection/bg.jpg";

    if (downloadFile(videoUrl, videoFilename))
    {
        cout << "Video downloaded successfully." << endl;
    }
    else
    {
        cout << "Failed to download video." << endl;
    }

    if (downloadFile(imageUrl, imageFilename))
    {
        cout << "Image downloaded successfully." << endl;
    }
    else
    {
        cout << "Failed to download image." << endl;
    }
}

vector<MarkAndTime> parseSimpleJson(const string &jsonString) {
    vector<MarkAndTime> marks;
    auto json = nlohmann::json::parse(jsonString);

    for (const auto& item : json) {
        MarkAndTime mark;
        mark.mark_correct = item["mark_correct"].get<int>();
        mark.frame = item["frame"].get<int>();
        marks.push_back(mark);
    }

    return marks;
}

string toJSON(const vector<Section>& sections) {
    string json = "[\n";
    for(size_t i = 0; i < sections.size(); ++i) {
        const Section& sec = sections[i];
        json += "    {\n";
        json += "    \"id_sequence\": " + to_string(i) + ",\n";
        json += "    \"takeoff_frame\": " + to_string(sec.takeoff_frame) + ",\n";
        json += "    \"arrival_frame\": " + to_string(sec.arrival_frame) + ",\n";
        json += "    \"error\": " + string(sec.error ? "true" : "false") + "\n";
        json += "    }";
        if (i < sections.size() - 1) json += ",";
        json += "\n";
    }
    json += "]";

    string json2 = "[\n";
    for(size_t i = 0; i < sections.size(); ++i) {
        const Section& sec = sections[i];
        json2 += "    {\n";
        json2 += "    \"id_sequence\": " + to_string(i) + ",\n";
        json2 += "    \"takeoff_frame\": " + to_string(sec.takeoff_frame) + ",\n";
        json2 += "    \"arrival_frame\": " + to_string(sec.arrival_frame) + ",\n";
        json2 += "    \"error\": " + string(sec.error ? "true" : "false") + ",\n";
        json2 += "    \"items\": [\n";
        for (size_t j = 0; j < sec.items.size(); ++j) {
            const item& it = sec.items[j];
            json2 += "        {\n";
            json2 += "        \"code\": " + to_string(it.code) + ",\n";
            json2 += "        \"intersects\": " + to_string(it.intersects) + ",\n";
            json2 += "        \"frame\": " + to_string(it.frame) + ",\n";
            json2 += "        \"d_l\": " + to_string(it.d_l) + ",\n";
            json2 += "        \"d_r\": " + to_string(it.d_r) + ",\n";
            json2 += "        \"step_l\": " + string(it.step_l ? "true" : "false") + ",\n";
            json2 += "        \"step_r\": " + string(it.step_r ? "true" : "false") + "\n";
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

    cout << "JSON NUEVO \n\n" << json2 << endl;

    return json;
}

cv::Point2f ComputerVisionWeb::transformInv(cv::Point2f p)
{
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0, 0) = (p.x * calib_w) / real_w;
    pin.at<double>(1, 0) = (p.y * calib_h) / real_h;
    pin.at<double>(2, 0) = 1;
    cv::Mat pout = pin;
    return cv::Point2f(pout.at<double>(0, 0) / pout.at<double>(2, 0),
                       pout.at<double>(1, 0) / pout.at<double>(2, 0));
}

cv::Mat ComputerVisionWeb::recalibrateHomography() {
    vector<cv::Point2f> scenePoints;
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
    
    //Using same parameters as in calibration phase:
    return cv::findHomography(contourCenters, scenePoints, cv::RANSAC, 5);
}

float ComputerVisionWeb::distance(cv::Point2f &p1, cv::Point2f &p2)
{
    float dx = p1.x - p2.x, dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

void ComputerVisionWeb::processAvailableStepsWithCoverageArea(int index, int cur_objective)
{
    int pos_correction = frames_to_store / 2 + 3;
    if (index >= pos_correction)
    {
        int sframe = index - pos_correction + 1;
        processStepsWithCoverageArea(index - pos_correction, sframe, cur_objective);
        // smasks.erase(sframe);
        // spos.erase(sframe);
    }
}

void ComputerVisionWeb::processStepsWithCoverageArea(int index, int frame, int cur_objective){
    // smoothDisplacement(index); Esto ya esta considerado en el algoritmo de pasos
    // smoothBBoxes(index); Ya no hay BBoxes
    // TODO , left y right son los rectangulos de los pies, hay que generar el poligono de los pies y empezar
    // a usar estos en vez de las cajas, una vez con esto podemos ya seguir a las intersecciones
    // cv::Rect &left = left_rects_s[index], &right = right_rects_s[index];

    // Aca determinamos si el paso es valido o no, esto ya esta en la información del nuevo alg
    if (frames_info[index].stepDetection){
        if (frames_info[index].stepSide == "Both") {
            right_step[index] = 1;
            left_step[index] = 1;

        } else if (frames_info[index].stepSide == "Right") {
            right_step[index] = 1;

        } else if (frames_info[index].stepSide == "Left") {
            left_step[index] = 1;
        }
    }
    // if (leftStepCriteria(index)) { left_step[index] = 1; }
    // if (rightStepCriteria(index)){ right_step[index] = 1; }

    cv::Mat cur_copy2;
    int index_contour = intersectsObjective(cur_copy2, index, frame, left, left_step[index], right, right_step[index]);

#ifdef SHOW_FINAL_RESULTS
    cout << "Processed step index: " << index << endl;
    cout << "Processed step frame: " << frame << endl;

    if (left_step[index])
    {
        cout << "Step on left foot:" << this->odist1[index] << " to " << in_objective1[index] << " objective." << endl;
        cv::Point2f p = transformInv(left_foot[index]);
        cout << "Left foot position:" << p.x << ", " << p.y << endl;
    }

    if (right_step[index])
    {
        cout << "Step on right foot:" << this->odist2[index] << " to " << in_objective2[index] << " objective." << endl;
        cv::Point2f p = transformInv(right_foot[index]);
        cout << "Right foot position:" << p.x << ", " << p.y << endl;
    }

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;
    int thickness = 2;

    sframes[frame].copyTo(cur_copy2);
    if (left_step[index]) {
        cv::rectangle(cur_copy2, left, cv::Scalar(0, 255, 0));
        string text = "L";
        cv::putText(cur_copy2, text, cv::Point(left.x+1,left.y + left.height - 3), fontFace, fontScale, cv::Scalar(255, 255, 255),              thickness);
    } else
        cv::rectangle(cur_copy2, left, cv::Scalar(0, 0, 255)); // Left
    if (right_step[index]) {
        cv::rectangle(cur_copy2, right, cv::Scalar(0, 255, 0));
        string text = "R";
        cv::putText(cur_copy2, text, cv::Point(right.x+1,right.y + right.height - 3), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
    } else
        cv::rectangle(cur_copy2, right, cv::Scalar(0, 0, 255)); // Right

    drawObjectives(cur_copy2 , index_contour+1, cur_objective);

    fontScale = 0.5;
    for(int i=0; i<contourCenters.size(); ++i) {
        string text = to_string(i+1);
        cv::Point2f &p = contourCenters[i];
        //cout << "Objective " << i+1 << ": " << p.x << ", " << p.y << endl; 
        cv::circle(cur_copy2, cv::Point(rint(p.x), rint(p.y)), 2, cv::Scalar(255, 255, 0));
        cv::putText(cur_copy2, text, cv::Point(rint(p.x)+5, rint(p.y)), fontFace, fontScale, cv::Scalar(255, 255, 0),              thickness);
    }
    
    if (left_step[index])
        cv::circle(cur_copy2, left_foot[index], 3, cv::Scalar(0, 255, 255));

    if (right_step[index])
        cv::circle(cur_copy2, right_foot[index], 3, cv::Scalar(0, 255, 255));
    // cv::resize(cur_copy2, cur_copy2, cv::Size(4*cur_copy2.cols, 4*cur_copy2.rows));
    fontScale = 1.0;
    string text = "Frame: " + to_string(frame) + "  Index: " + to_string(index);
    cv::putText(cur_copy2, text, cv::Point(10,30), fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);

    cv::namedWindow("Everything", cv::WINDOW_NORMAL);
    cv::resizeWindow("Everything", 1920, 1000);
    cv::imshow("Everything", cur_copy2);
    cv::waitKey(0);
    saveResult(cur_copy2, frame);
#endif
}

int ComputerVisionWeb::intersectsObjective(cv::Mat &img, int index, int frame, cv::Rect &leftStep, bool leftStepOccurred, cv::Rect &rightStep, bool rightStepOccurred){
    auto minSceneDistanceToRectContour = [&](const cv::Rect &stepRect, const vector<cv::Point2f> &scene_contour) -> float{
        vector<cv::Point2f> rectPoints = {
            imageToScene(cv::Point2f(stepRect.tl())),
            imageToScene(cv::Point2f(stepRect.br().x, stepRect.tl().y)),
            imageToScene(cv::Point2f(stepRect.br())),
            imageToScene(cv::Point2f(stepRect.tl().x, stepRect.br().y))};

        float minDistance = FLT_MAX;
        for (const auto &rectPoint : rectPoints)
        {
            for (int i = 0; i < scene_contour.size(); i++)
            {
                float distance = calculatePointToLineDistance(scene_contour[i], scene_contour[(i + 1) % scene_contour.size()], rectPoint);
                minDistance = min(minDistance, distance);
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

    for (int contourIndex = 0; contourIndex < this->contours.size(); contourIndex++) {
        
        Contour &contour = this->contours[contourIndex];
        Contour &scene_contour = this->contoursScene[contourIndex];
        distance_left = minSceneDistanceToRectContour(leftStep, scene_contour.points);
        if (distance_left < minDistance_left) {
            minDistance_left = distance_left;
            id_minDist_left = contourIndex;
        }

        intersects_left = feetIntersectsObjective(leftStep, contour.ipoints); // FALTA ACA, IMPORTANTE
        this->left_intersects[index] = 0;

        if (intersects_left && leftStepOccurred) {
            // Intersection or closeness logic for left step
            this->odist1[index] = 0; // TODO FALTA ACA
            
            // getStepPosition(frame, leftStep); TODO Deberia ser el punto central del pie
            // this->left_foot[index] = getStepPosition(frame, leftStep);
            this->left_foot[index] = frames_info[index].left_position.center;

            this->in_objective1[index] = contourIndex+1;

//            cout << "Intersect left_foot frame: " << frame << " contourIndex: "<< contourIndex <<endl;
            flagIntersect = true;
            this->left_intersects[index] = 1; // FALTA ACA
        } else {
            this->odist1[index] = minDistance_left;
            this->in_objective1[index] = id_minDist_left+1;

            // this->left_foot[index] = getStepPosition(frame, leftStep);
            this->left_foot[index] = frames_info[index].left_position.center;

        }

        distance_right = minSceneDistanceToRectContour(rightStep, scene_contour.points);
        if (distance_right < minDistance_right) {
            minDistance_right = distance_right;
            id_minDist_right = contourIndex;
        }

        intersects_right = feetIntersectsObjective(rightStep, contour.ipoints);
        this->right_intersects[index] = 0;

        if (intersects_right && rightStepOccurred) {
            // Intersection or closeness logic for right step
            this->odist2[index] = 0;

            // this->right_foot[index] = getStepPosition(frame, rightStep);
            this->right_foot[index] = frames_info[index].right_position.center;
            this->in_objective2[index] = contourIndex+1;
            
            // cout << "Intersect right_foot frame: " << frame << " contourIndex: "<< contourIndex <<endl;
            flagIntersect = true;
            this->right_intersects[index] = 1;
        } else {
            this->odist2[index] = minDistance_right;
            this->in_objective2[index] = id_minDist_right+1;

            // this->right_foot[index] = getStepPosition(frame, rightStep);
            this->left_foot[index] = frames_info[index].right_position.center;

        }
        if (flagIntersect) {
            return contourIndex;
        }
    }
    return 1000;
}

// Calculates the distance between a point and a line segment
float ComputerVisionWeb::calculatePointToLineDistance(const cv::Point2f &pointA, const cv::Point2f &pointB, const cv::Point2f &point){
    float segmentLength = cv::norm(pointB - pointA);
    if (segmentLength == 0.0)
        return cv::norm(point - pointA);

    float t = ((point.x - pointA.x) * (pointB.x - pointA.x) + (point.y - pointA.y) * (pointB.y - pointA.y)) / (segmentLength * segmentLength);
    t = max(0.0f, min(1.0f, t));
    cv::Point2f projection = pointA + t * (pointB - pointA);
    return cv::norm(point - projection);
}

bool ComputerVisionWeb::feetIntersectsObjective(cv::Rect rect, vector<cv::Point2i> &contour) {
    //Adjust feet if is too tall (assume it can be at most as tall as wide, and that contact zone will be at most at half size of feet box):
    //1. Feet correction (at most as tall as wide)
    if(rect.width < rect.height) {
        rect.y +=  rect.height - rect.width;
        rect.height = rect.width;
    }
    //2. Take the 50% of feet box as contact zone:
    rect.y += rect.height/2;
    rect.height /= 2;
    
    //Set feet polygon: //TODO HAY QUE PONER LOS PUNTOS DEL PIE AHORA EN VEZ DE EL CUADRADO DEL PIE
    vector<cv::Point2i> feetPoints = {
        {rect.x, rect.y},
        {rect.x + rect.width, rect.y},
        {rect.x + rect.width, rect.y + rect.height},
        {rect.x, rect.y + rect.height}
    };
    return isPolygonIntersection(feetPoints, contour);
}

string ComputerVisionWeb::buildOutput(vector<MarkAndTime> sequence, int maxFrame) { 
    //Get central stimulus central position
    cv::Point2f pcentral = contourCenters[4];

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
    vector<Section> sequences;
    bool first_stimulus = true;
    
    cout << "\n TEST 1 \n" << endl;

    //Get intervals per objective:
    for (int j = 0; j < n_objectives; ++j) {
        Section cur_section;
        item cur_item;
        
        cur_objective = sequence[j].mark_correct; 
        cur_frame = sequence[j].frame;

#ifdef SHOW_DEBUG_TEXT
        cout << "Marking.\n\tCurrent stimulus: " << j << endl;
        cout << "\tCurrent stimulus objective: " << cur_objective << endl;
        cout << "\tCurrent stimulus index: " << cur_frame << endl;
#endif        
        //Booleans for marking errors (assume right first):
        bool step_center = true, step_objective = true, right_objective = true; 
        
        //Advance until both are near the 5 zone (assume that the player can be late):
        for (i = current_seq; i < maxFrame; ++i)
            if(in_objective1[i] == 5 || in_objective2[i] == 5)
                break;

#ifdef SHOW_DEBUG_TEXT
        cout << "\tCurrent sequence start - prev cur_frame: " << current_seq << endl;
#endif        
        //Update start of current seq: if stimulus presentation is higher than presence in zone 5, start from stimulus presentation frame
        current_seq = (cur_frame >= i)? cur_frame : i;
        cout << "\n current_seq \n" << current_seq << endl;

#ifdef SHOW_DEBUG_TEXT
        cout << "\tCurrent sequence start - after cur_frame: " << current_seq << endl;
#endif        

        //Search for center position exit, considered as the first step out of center zone:
        bool step_out_detected1 = false, step_out_detected2 = false;
        int sure_frame1, sure_frame2;

        cout << "\n TEST 2 \n" << endl;

        //Search for left exit:
        for (i = current_seq; i < maxFrame; ++i) {
            if(left_step[i] == 0) //Continue until a step is detected
                continue;
#ifdef SHOW_DEBUG_TEXT
        cout << "\tCurrent frame - search left position exit: " << i << endl;
#endif        
            //Left foot steps out:
            if(in_objective1[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
        cout << "\tExit frame left - position exit: " << i << endl;
        cout << "\tExit frame left - code found: " << in_objective1[i] << endl;
#endif        
                current_center_exit1 = i;
                p_out1 = left_foot[i];
                sure_frame1 = i;
                step_out_detected1 = true;
                break;
            }            
        }

        //Search for right exit:
        for (i = current_seq; i < maxFrame; ++i) {
            if(right_step[i] == 0) //Continue until a step is detected
                continue;
#ifdef SHOW_DEBUG_TEXT
        cout << "\tCurrent frame - search right position exit: " << i << endl;
#endif        
            //Right foot steps out:
            if(in_objective2[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
        cout << "\tExit frame right - position exit: " << i << endl;
        cout << "\tExit frame right - code found: " << in_objective2[i] << endl;
#endif        
                current_center_exit2 = i;
                p_out2 = right_foot[i];
                sure_frame2 = i;
                step_out_detected2 = true;
                break;
            }            
        }

        cout << "\n TEST 3 \n" << endl;

        
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
            cv::Point2f p1, p2, p_out_s = imageToScene(p_out1), p_center = contourCentersScene[4]; //Take central point as reference
            float d1, d2, d, 
                  d_center_obj = sqrt((p_out_s.x - p_center.x)*(p_out_s.x - p_center.x) + (p_out_s.y - p_center.y)*(p_out_s.y - p_center.y));
#ifdef SHOW_DEBUG_TEXT
                cout << "\tProcessing take-off left..." << endl;
                cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << endl;
                cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << endl;
#endif

            //Check backwards first relevant change in left step keeping going far sure feet and near center
            for (i = sure_frame1; i >= current_seq; --i) {
#ifdef SHOW_DEBUG_TEXT
                cout << "\tTake-off - left - frame: " << i << endl;
#endif                        
                if(first) { //Search for end of first stepping
                    if(!stepping && left_step[i] == 1) { //A step
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tTake-off - left - start stepping... " << endl;
#endif                        
                        stepping = true;
                        il_last = i;
                        il_initial = i;
                        lindex_1 = i;
                    } else if(stepping && left_step[i] == 0) { //Is stepping, so check if it stops doing so
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tTake-off - left - stop stepping... " << endl;
#endif                  
                        il_last_stepping = i-1;      
                        stepping = false;
                        first = false; //First index ready
                    }
                } else { //Search for first of following step
                    if(left_step[i] == 1) { //First position of next step
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tTake-off - left - left following found... " << endl;
#endif        
                        il_last = i;
                        lindex_2 = i;
                        //Significant displacement criterion
                        p1 = imageToScene(left_foot[lindex_1]);
                        p2 = imageToScene(left_foot[lindex_2]);
                        d  = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
                        //d1 = sqrt((p_out_s.x - p1.x)*(p_out_s.x - p1.x) + (p_out_s.y - p1.y)*(p_out_s.y - p1.y));//L2 norm
                        //d2 = sqrt((p_out_s.x - p2.x)*(p_out_s.x - p2.x) + (p_out_s.y - p2.y)*(p_out_s.y - p2.y));//L2 norm
                        d1 = magnitude(projectVector(cv::Point2f(p1.x - p_out_s.x, p1.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                        d2 = magnitude(projectVector(cv::Point2f(p2.x - p_out_s.x, p2.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                        
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << endl;
                        cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << endl;
                        cout << "\tFoot 1 (x,y): " << p1.x << ", " << p1.y << " at index " << lindex_1 << endl;
                        cout << "\tFoot 2 (x,y): " << p2.x << ", " << p2.y << " at index " << lindex_2 << endl;                        
                        cout << "\tTake-off - left - distance between steps: " << d << endl;
                        cout << "\tTake-off - left - projected distance between 1st step and objective: " << d1 << endl;
                        cout << "\tTake-off - left - projected distance between 2nd step and objective: " << d2 << endl;
                        cout << "\tTake-off - left - distance between center and objective: " << d_center_obj << endl;
#endif
                        if(d1 < d2 && d > relevant_change && il_initial - il_last_stepping < static_step && d2 < d_center_obj && odist1[i] != 0) { //While the step is approaching to the objective and not farther than center, keep searching...
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - left - keeps approaching objective... " << endl;
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
                        cout << "\tTake-off - left - il initial:" << il_initial << endl;
                        cout << "\tTake-off - left - il last stepping:" << il_last_stepping << endl;
                        cout << "\tTake-off - left - stepping diff:" << il_initial - il_last_stepping << endl;
#endif                        
                        
                        if(il_initial - il_last_stepping >= static_step) { //Static step, so this is considered the take-off step
                            il_found = il_initial + 1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - left - static step: Found at " << il_found << "." << endl;
#endif
                            break;
                        }
                        
                        if(d <= relevant_change) { // If two little steps, assume second is the take_off
                            il_found = lindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - left - irrelevant step distance: Found at " << il_found << "." << endl;
#endif        

                            break;
                        }
                        
                        if(d1 > d2) { //If p1 is farther than p2, it is the take-off step
                            il_found = il_initial + 1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - left - first step farther than second: Found at " << il_found << ". " << endl;
#endif
                        } else { //Else, p2 is.                            
                            il_found = lindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - left - second step farther than first: Found at " << il_found << ". " << endl;
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
                cout << "\tTake-off - left - not found so take last step: Found at " << il_found << ". " << endl;
#endif
            }

            //Now check first relevant change in right step
            stepping = false; 
            first = true;
            p_out_s = imageToScene(p_out2);
            d_center_obj = sqrt((p_out_s.x - p_center.x)*(p_out_s.x - p_center.x) + (p_out_s.y - p_center.y)*(p_out_s.y - p_center.y));
#ifdef SHOW_DEBUG_TEXT
            cout << "\tProcessing take-off right..." << endl;
            cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << endl;
            cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << endl;
#endif
            for (i = sure_frame2; i >= current_seq; --i) {
#ifdef SHOW_DEBUG_TEXT
                cout << "\tTake-off - right - frame: " << i << endl;
#endif                        
                if(first) { //Search for end of first stepping
                    if(!stepping && right_step[i] == 1) { //A step
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tTake-off - right - start stepping... " << endl;
#endif                        
                        stepping = true;
                        ir_last = i;
                        ir_initial = i;
                        rindex_1 = i;
                    } else if(stepping && right_step[i] == 0) { //Is stepping, so check if it stops doing so
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tTake-off - right - stop stepping... " << endl;
#endif                  
                        ir_last_stepping = i+1;
                        stepping = false;
                        first = false; //First index ready
                    }
                } else { //Search for first of following step
                    if(right_step[i] == 1) { //First position of next step
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tTake-off - right - right following found... " << endl;
#endif        
                        ir_last = i;
                        rindex_2 = i;
                        //Significant displacement criterion
                        p1 = imageToScene(right_foot[rindex_1]);
                        p2 = imageToScene(right_foot[rindex_2]);
                        d  = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
//                        d1 = sqrt((p_out_s.x - p1.x)*(p_out_s.x - p1.x) + (p_out_s.y - p1.y)*(p_out_s.y - p1.y));//L2 norm
//                        d2 = sqrt((p_out_s.x - p2.x)*(p_out_s.x - p2.x) + (p_out_s.y - p2.y)*(p_out_s.y - p2.y));//L2 norm
                        d1 = magnitude(projectVector(cv::Point2f(p1.x - p_out_s.x, p1.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                        d2 = magnitude(projectVector(cv::Point2f(p2.x - p_out_s.x, p2.y - p_out_s.y), cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));

#ifdef SHOW_DEBUG_TEXT
                        cout << "\tCenter (x,y): " << p_center.x << ", " << p_center.y << endl;
                        cout << "\tSure out (x,y): " << p_out_s.x << ", " << p_out_s.y << endl;
                        cout << "\tFoot 1 (x,y): " << p1.x << ", " << p1.y << " at index " << rindex_1 << endl;
                        cout << "\tFoot 2 (x,y): " << p2.x << ", " << p2.y << " at index " << rindex_2 << endl;   
                        cout << "\tTake-off - right - distance between steps: " << d << endl;
                        cout << "\tTake-off - right - projected distance between 1st step and objective: " << d1 << endl;
                        cout << "\tTake-off - right - projected distance between 2nd step and objective: " << d2 << endl;
                        cout << "\tTake-off - right - distance between center and objective: " << d_center_obj << endl;
#endif
                        if(d1 < d2 && d > relevant_change && ir_initial - ir_last_stepping < static_step && d2 < d_center_obj  && odist2[i] != 0) { //While the step is approaching to the objective and not farther than center, keep searching...
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - right - keeps approaching objective... " << endl;
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
                        cout << "\tTake-off - right - ir initial:" << ir_initial << endl;
                        cout << "\tTake-off - right - ir last stepping:" << ir_last_stepping << endl;
                        cout << "\tTake-off - right - stepping diff:" << ir_initial - ir_last_stepping << endl;
#endif        
                        
                        if(ir_initial - ir_last_stepping >= static_step) { //Static step, so this is considered the take-off step
                            ir_found = ir_initial + 1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - right - static step: Found at " << ir_found << "." << endl;
#endif        
                            break;
                        }
                        
                        if(d <= relevant_change) { // If two little steps, assume second is the take_off
                            ir_found = rindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - right - irrelevant step distance: Found at " << ir_found << "." << endl;
#endif        
                            break;
                        }
                        
                        if(d1 > d2) { //If p1 is farther than p2, it is the take-off step
                            ir_found = ir_initial+1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - right - first step farther than second: Found at " << ir_found << "." << endl;
#endif
                        } else { //Else, p2 is.                            
                            ir_found = rindex_2+1;
#ifdef SHOW_DEBUG_TEXT
                            cout << "\tTake-off - right - second step farther than first: Found at " << ir_found << "." << endl;
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
                cout << "\tTake-off - right - not found so take last step: Found at " << ir_found << ". " << endl;
#endif
            }
            
            //Set take-off frame:
            if(pl_found && pr_found) {
                if(il_found < ir_found) { //Take-off is from left
                    cur_item.code = 5;
                    cur_item.d_l = odist1[il_found];
                    cur_item.d_r = odist2[ir_found];
                    cur_item.step_l = true;
                    cur_item.step_r = false;
                    cur_item.frame = il_found;
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = il_found;
#ifdef SHOW_DEBUG_TEXT
                    cout << "\tTake-off: Both found - decided left" << endl;
#endif
                } else {
                    cur_item.code = 5;
                    cur_item.d_l = odist1[il_found];
                    cur_item.d_r = odist2[ir_found];
                    cur_item.step_l = false;
                    cur_item.step_r = true;
                    cur_item.frame = ir_found;                    
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = ir_found;
#ifdef SHOW_DEBUG_TEXT
                    cout << "\tTake-off: Both found - decided right" << endl;
#endif
                }
            } else if(pl_found) {
                cur_item.code = 5;
                cur_item.d_l = odist1[il_found];
                cur_item.d_r = odist2[il_last];
                cur_item.step_l = true;
                cur_item.step_r = false;
                cur_item.frame = il_found;
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = il_found;
#ifdef SHOW_DEBUG_TEXT
                cout << "\tTake-off: One found - left" << endl;
#endif
            } else if (pr_found) {
                cur_item.code = 5;
                cur_item.d_l = odist1[il_found];
                cur_item.d_r = odist2[ir_found];
                cur_item.step_l = false;
                cur_item.step_r = true;
                cur_item.frame = ir_found;                                    
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = ir_found;
#ifdef SHOW_DEBUG_TEXT
                cout << "\tTake-off: One found - right" << endl;
#endif
            } else { //if none, use max
                cur_item.code = 5;
                cur_item.d_l = odist1[il_found];
                cur_item.d_r = odist2[ir_found];
                cur_item.step_l = il_found;
                cur_item.step_r = ir_found;
                cur_item.frame = il_found<ir_found? il_found : ir_found;                                    
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = il_found<ir_found? il_found : ir_found;
#ifdef SHOW_DEBUG_TEXT
                cout << "\tTake-off: None found" << endl;
#endif
            }
#ifdef SHOW_DEBUG_TEXT
            cout << "\tTake-off: " << cur_item.frame << endl;
            cout << "\tTake-off - left?: " << cur_item.step_l << endl;
#endif
        //end if step_out_detected
        } else { 
#ifdef SHOW_DEBUG_TEXT
            cout << "\tNo step out found. " << endl;
#endif

            break; //No step out, means irrelevant rest of video
        }
        
        cout << "\n TEST 4 \n" << endl;

        //Search for arrival info
        int last_arrival_frame = -1; //Value is -1 if no one steps, and the corresponding frame if it steps
        int nearest_arrival_code; //Stepping or not, it is the nearest arrival code
        int last_real = -1;
        float nearest_arrival_distance = FLT_MAX;
        bool still_near_center_l = true, still_near_center_r = true, real_arrival = false, is_left = true;
        for (int i = current_center_exit1<current_center_exit2?current_center_exit1:current_center_exit2; i < maxFrame; ++i) {
#ifdef SHOW_DEBUG_TEXT
            cout << "\tArrival - frame: " << i << endl;
#endif            
            //There might still be one of the feet near center
            if(in_objective1[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
                cout << "\tArrival - check left far center... " << endl;
#endif            
                still_near_center_l = false;
                if(left_step[i] == 1) { 
#ifdef SHOW_DEBUG_TEXT
                    cout << "\tArrival - check left far center step - code: " << in_objective1[i] << endl;
                    cout << "\tArrival - check left far center step - distance: " << odist1[i] << endl;
                    cout << "\tArrival - check left far center step - last_real: " << last_real << endl;
#endif            
                    if(odist1[i]==0 && last_real != in_objective1[i]) { //Registers the first arrival with this code
                        last_arrival_frame = i;
                        last_real = nearest_arrival_code = in_objective1[i];
                        nearest_arrival_distance = 0;
                        real_arrival = true;
                        is_left = true;
#ifdef SHOW_DEBUG_TEXT
                cout << "\tArrival - left real arrival found at: " << i << endl;
#endif            
                    } 
                    if(!real_arrival && odist1[i] < nearest_arrival_distance) {
                        nearest_arrival_distance = odist1[i];
                        last_arrival_frame = i;
                        nearest_arrival_code = in_objective1[i];
                        is_left = true;
#ifdef SHOW_DEBUG_TEXT
                cout << "\tArrival - left near arrival found at: " << i << endl;
                cout << "\tArrival - left near arrival - nearest arrival distance: " << nearest_arrival_distance << endl;
#endif            
                    }
                }
            }
            if(in_objective2[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
                cout << "\tArrival - check right far center... " << endl;
#endif            
                still_near_center_r = false;
                if(right_step[i] == 1) { 
#ifdef SHOW_DEBUG_TEXT
                    cout << "\tArrival - check right far center step - code: " << in_objective2[i] << endl;
                    cout << "\tArrival - check right far center step - distance: " << odist2[i] << endl;
                    cout << "\tArrival - check right far center step - last_real: " << last_real << endl;
#endif            
                    if(odist2[i]==0 && last_real != in_objective2[i]) { //Registers the first arrival with this code
                        last_arrival_frame = i;
                        last_real = nearest_arrival_code = in_objective2[i];
                        nearest_arrival_distance = 0;
                        real_arrival = true;
                        is_left = false;
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tArrival - right real arrival found at: " << i << endl;
#endif            
                    } 
                    if(!real_arrival && odist2[i] < nearest_arrival_distance) {
                        nearest_arrival_distance = odist2[i];
                        last_arrival_frame = i;
                        nearest_arrival_code = in_objective2[i];
                        is_left = false;
#ifdef SHOW_DEBUG_TEXT
                        cout << "\tArrival - right near arrival found at: " << i << endl;
                        cout << "\tArrival - right near arrival - nearest arrival distance: " << nearest_arrival_distance << endl;
#endif            
                    }
                }

            }
            
            if(!still_near_center_l && in_objective1[i] == 5 && left_step[i] == 1) { //First step returning to center
#ifdef SHOW_DEBUG_TEXT
                cout << "\tArrival - left step returning to center found at: " << i << endl;
#endif            

                next_seq = i;
                break;
            }
            if(!still_near_center_r && in_objective2[i] == 5 && right_step[i] == 1) { //First step returning to center
#ifdef SHOW_DEBUG_TEXT
                cout << "\tArrival - right step returning to center found at: " << i << endl;
#endif            
                next_seq = i;
                break;
            }
            
        } 
        
        cout << "\n TEST 5 \n" << endl;

        //Set arrival errors and arrival time
        if(!real_arrival || nearest_arrival_code != cur_objective) {
            if(real_arrival && nearest_arrival_code != cur_objective)
                right_objective = false;
            if(!real_arrival)
                step_objective = false; 
        }
        cur_item.code = nearest_arrival_code;
        cur_item.d_l = odist1[last_arrival_frame];
        cur_item.d_r = odist2[last_arrival_frame];
        cur_item.step_l = is_left? 1:0;
        cur_item.step_r = is_left? 0:1;
        cur_item.frame = last_arrival_frame;
        cur_section.items.push_back(cur_item);
        cur_section.arrival_frame = last_arrival_frame;
        cur_section.arrival_code = nearest_arrival_code;

#ifdef SHOW_DEBUG_TEXT
        cout << "\tArrival - arrival frame: " << cur_section.arrival_frame << endl;
        cout << "\tArrival - arrival code: " << cur_section.arrival_code << endl;
        if(is_left)
            cout << "\tArrival - arrival distance: " << odist1[last_arrival_frame] << endl;
        else
            cout << "\tArrival - arrival distance: " << odist2[last_arrival_frame] << endl;
#endif            

        
        //Check if steps in center on return
        bool still_far_center_l = true, still_far_center_r = true, ready_l = false, ready_r = false;
        step_center = false;
        for (int i = next_seq; i < maxFrame; ++i) {
#ifdef SHOW_DEBUG_TEXT
            cout << "\tReturn to center - frame: " << i << endl;
#endif            
            //There might still be one of the feet near center
            if(in_objective1[i] == 5) {
                still_far_center_l = false;
                if(left_step[i] == 1) {
#ifdef SHOW_DEBUG_TEXT
            cout << "\tReturn to center - found left at: " << i << endl;
#endif            
                    step_center = true;                    
                    break;
                }
            }
            if(in_objective2[i] == 5) {
                still_far_center_r = false;
                if(right_step[i] == 1) {
#ifdef SHOW_DEBUG_TEXT
                    cout << "\tReturn to center - found right at: " << i << endl;
#endif            
                    step_center = true;
                    break;
                }
            }
            
            if(!still_far_center_l && in_objective1[i] != 5)
                ready_l = true;
                
            if(!still_far_center_r && in_objective2[i] != 5)
                ready_r = true;
            
            if(ready_l && ready_r) {
#ifdef SHOW_DEBUG_TEXT
            cout << "\tReturn to center - both ready at: " << i << endl;
#endif            
                break;
            }
        } 
        
        //Set error and store in sequences
        cur_section.error = (right_objective && step_objective && step_center) ? false : true;
#ifdef SHOW_DEBUG_TEXT
        cout << "\tFinal stimulus error: " << cur_section.error << endl;
#endif            

        sequences.push_back(cur_section);
        
        //Set next beggining to next sequence of centers
        current_seq = next_seq;
        first_stimulus = false;
    } //end stimuli sequence
    
    return toJSON(sequences);
}

size_t writeCallback(void* contents, size_t size, size_t nmemb, string* s) {
    size_t newLength = size * nmemb;
    s->append((char*)contents, newLength);
    return newLength;
}

bool ComputerVisionWeb::callApi(const string& videoUrl) {
    CURL* curl;
    CURLcode res;
    string readBuffer;
    
    curl = curl_easy_init();
    if(curl) {
        string api_url = "http://blazepose-api:5000/process-video";
        string json_payload = "{\"video_url\": \"" + videoUrl + "\"}";

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
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

            } catch (const exception& e) {
                cerr << "Error al parsear el JSON: " << e.what() << endl;
            }
        }
        curl_easy_cleanup(curl);
        return res == CURLE_OK;
    }
    return false;
}

string ComputerVisionWeb::mainFunction(string contourjson, string videoUrl, string imageUrl, string jsonString, string frameRate) {
    if (callApi(videoUrl)) {
        cout << "Procesamiento exitoso, datos recibidos desde la API de pose." << endl;
        int frame_count = 0;

        cout << "Muestra de los dos primero frames: \n"<< endl;
        for (const auto& frame : frames_info) {
            if (frame_count >= 2) {
                break;
            }

            cout << "Frame Index: " << frame.frame_index << endl;
            cout << "Step Detection: " << (frame.stepDetection ? "True" : "False") << endl;
            cout << "Step Side: " << frame.stepSide << endl;
            cout << "Left Heel Position: (" << frame.left_position.heel.x << ", " << frame.left_position.heel.y << ")" << endl;
            cout << "Left Foot Index Position: (" << frame.left_position.foot_index.x << ", " << frame.left_position.foot_index.y << ")" << endl;
            cout << "Left Ankle Pos    ition: (" << frame.left_position.ankle.x << ", " << frame.left_position.ankle.y << ")" << endl;
            cout << "Right Heel Position: (" << frame.right_position.heel.x << ", " << frame.right_position.heel.y << ")" << endl;
            cout << "Right Foot Index Position: (" << frame.right_position.foot_index.x << ", " << frame.right_position.foot_index.y << ")" << endl;
            cout << "Right Ankle Position: (" << frame.right_position.ankle.x << ", " << frame.right_position.ankle.y << ")\n" << endl;

            frame_count++;
        }

    } else {
        cout << "Error al llamar a la API pose-IA." << endl;
    }
    
    // String contornos se debe pasar a vector<Contour>
    istringstream iss(contourjson);

    Json::Value root;
    iss >> root;

    string string_calib_w = to_string(root["response"]["calib_w"].asInt());
    string string_calib_h = to_string(root["response"]["calib_h"].asInt());
    
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

    // String sequence se debe pasar a vector<MarkAndTime>
    vector<MarkAndTime> sequence = parseSimpleJson(jsonString);

    // Video e imagen
    downloadMedia(videoUrl, imageUrl);

    string urlVideo = "/usr/src/app/mcp-vision-detection/video.mp4";
    string urlBG = "/usr/src/app/mcp-vision-detection/bg.jpg";

    calib_w = stoi(string_calib_w);
    calib_h = stoi(string_calib_h);


    cv::VideoCapture vtest;
    vtest.open(urlVideo);

    float frame_rate = 0.0f;
    if (vtest.isOpened())
    {
        frame_rate = stof(frameRate);
    }
    else
    {
        cout << "El video no abrio!!" << endl;
        return "Error al abrir video";
    }

    cout << "Sequence:\n";
    for (const auto& markTime : sequence) {
        cout << "Mark: " << markTime.mark_correct << ", Time: " << markTime.frame << endl;
    }

    cout << "URL del video procesado: " << videoUrl << endl;
    cout << "URL de la imagen de fondo procesada: " << imageUrl << endl;
    cout << "Frame rate del video: " << frame_rate << endl;
    cout << "calib_w: " << calib_w << endl;
    cout << "calib_h: " << calib_h << endl;

    cv::Mat current, result, result_big;

    bool first = true;
    uint frame = 0, maxFrame, time = 0, msec_per_frame = 1000 / frame_rate,
        initial_msec = 0,
        final_msec = INT_MAX; // final_msec = 10000;

    cv::Mat fg;
    // Insert background calibration image
    cv::Mat bg = cv::imread(urlBG);

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("Background", bg);
#endif


    map<int, int> msecs;
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
            cout << "Dimensions: " << real_w << "x" << real_h << endl;
            cout << "Dimensions Calib: " << calib_w << "x" << calib_h << endl;

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

    map<int, int>::iterator frame_it = msecs.begin();

    vtest.open(urlVideo);
    if (!vtest.isOpened())
    {
        cout << "El video no abrio la segunda vez!!" << endl;
        return "El video no abrio la segunda vez!!";
    }
    
    contours = contornos;
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

    //Recalculate homography image --> scene with contour centers
    H = recalibrateHomography();

    //Get proyected contours:
    for (const auto &c : contornos) {
        Contour contorno;
        contorno.indiceContorno = c.indiceContorno;
        contorno.x = c.x;
        contorno.y = c.y;
        contorno.z = c.z;
        for (const auto &p : c.points) 
            contorno.points.push_back(imageToScene(p));
        contoursScene.push_back(contorno); 
    }
    
    cout << "Recalibration: " << endl;
    for(int i=0 ; i< contourCenters.size() ; ++i) {
        cv::Point2f p = imageToScene(contourCenters[i]);
        cout << "\tObjective " << i+1 << ": " << p.x << ", " << p.y << endl; 
        contourCentersScene.push_back(p);
    }

    // Adjust tracking and get steps
    left_foot.resize(maxFrame);
    right_foot.resize(maxFrame);
    left_step.resize(maxFrame, 0);
    right_step.resize(maxFrame, 0);
    left_rects_s.resize(maxFrame);
    right_rects_s.resize(maxFrame);
    in_objective1.resize(maxFrame, 0);
    in_objective2.resize(maxFrame, 0);
    odist1.resize(maxFrame, 0.0);
    odist2.resize(maxFrame, 0.0);
    left_intersects.resize(maxFrame, 0);
    right_intersects.resize(maxFrame, 0);

#ifdef MEMORY_DEBUG
    cerr << "End calibration init...\n\nStart step processing..." << endl;
#endif

    uint j_cur = 0, n_objectives = sequence.size();
    int cur_objective = sequence[0].mark_correct;

    // This for shall be removed, doesnt do anything for the new version
    for (uint i = 1; i <= maxFrame; ++i)
    {
        frame = frame_it->first;
#ifdef MEMORY_DEBUG
        cerr << "\tStep processing - Frame: " << frame << endl;
#endif

#ifdef SHOW_INTERMEDIATE_RESULTS
        cout << "Frame: " << frame << endl;
        cout << "Time: " << frame_it->second << " [msecs]" << endl;
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
            // cout << "Frame: " << frame << "; OFrame: " << oframe << endl;
            if(frame < oframe) 
                break;
            int cobjective = m.mark_correct;
            cur_objective = cobjective;
            j_cur = j;
        }

        // cv::Rect player_roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
        // ft.player_roi.push_back(player_roi);
        // // Set candidates and track them:
        // ft.setFeetPositionsByBBox(frame, player_roi, result3);
        // ft.trackPositions(frame, player_roi, result3, current, frame_it->second, i);
        processAvailableStepsWithCoverageArea(i, cur_objective);
        
#ifdef SHOW_INTERMEDIATE_RESULTS
        cout << "Current Objective: " << cur_objective << endl;
#endif        
        frame_it++;
    }

    vtest.release();

#ifdef MEMORY_DEBUG
    cerr << "End step processing...\n\nStart step completion..." << endl;
#endif

#ifdef SHOW_INTERMEDIATE_RESULTS
    cout << "Last processed frame: " << frame << endl;
    cout << "Max frame: " << frame << endl;
#endif

    string out = buildOutput(sequence, maxFrame);
    cout << "\n\n OUT: \n" << out << endl;

#ifdef SHOW_FINAL_RESULTS
    cout << "============ OUT ============ \n" << out << endl;
#endif

#ifdef MEMORY_DEBUG
    cerr << "End step completion..." << endl;
#endif

    return out;
}