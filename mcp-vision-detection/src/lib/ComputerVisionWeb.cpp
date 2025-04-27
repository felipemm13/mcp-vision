#include "ComputerVisionWeb.h"

// ========================================================================
// Variables y utilidades globales
// ========================================================================
cv::VideoWriter videoWriter;
bool isVideoWriterInitialized = false;
bool showFrames = false;

namespace
{
    size_t callback(const char *in, size_t size, size_t num, string *out)
    {
        const size_t totalBytes(size * num);
        out->append(in, totalBytes);
        return totalBytes;
    }
}

// ========================================================================
// Constructor
// ========================================================================
ComputerVisionWeb::ComputerVisionWeb(){}

// =========================
// Funciones de descarga
// =========================
size_t writeData(void *ptr, size_t size, size_t nmemb, FILE *stream){
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

bool downloadFile(const string &url, const string &outFilename){
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

void downloadMedia(const string &videoUrl, const string &imageUrl){
    string videoFilename = "/usr/src/app/mcp-vision-detection/video.mp4";
    string imageFilename = "/usr/src/app/mcp-vision-detection/bg.jpg";

    if (downloadFile(videoUrl, videoFilename))
        cout << "Video downloaded successfully." << endl;
    else
        cout << "Failed to download video." << endl;

    if (downloadFile(imageUrl, imageFilename))
        cout << "Image downloaded successfully." << endl;
    else
        cout << "Failed to download image." << endl;
}

// =======================
// makeExpandedPolygon
// =======================
vector<cv::Point2i> makeExpandedPolygon(const Position& pos, const cv::Mat& mask, float max_distance, int num_directions = 36){
    vector<cv::Point2i> expanded_polygon;
    cv::Point2f average = (pos.ankle + pos.heel) * 0.5f;
    vector<cv::Point2f> expansion_points = {average, cv::Point2f(pos.foot_index)};

    if (mask.empty()) {
        // Si no hay mask, hacemos un polígono mínimo
        expanded_polygon.push_back(pos.heel);
        expanded_polygon.push_back(pos.foot_index);
        expanded_polygon.push_back(pos.ankle);
        return expanded_polygon;
    }

    for (const auto& start_point : expansion_points)
    {
        for (int i = 0; i < num_directions; ++i)
        {
            float angle = float((2 * CV_PI / num_directions) * i);
            cv::Point2f direction(std::cos(angle), std::sin(angle));

            float dist;
            for (dist = 0; dist <= max_distance; dist += 1.0f)
            {
                cv::Point2i expanded_point = cv::Point2i(start_point + direction * dist);
                // Revisar límites de la imagen
                if (expanded_point.y < 0 || expanded_point.y >= mask.rows ||
                    expanded_point.x < 0 || expanded_point.x >= mask.cols)
                {
                    break;
                }
                // Si en la mascara ya no hay pie, se frena
                if (mask.at<uchar>(expanded_point.y, expanded_point.x) == 0)
                {
                    break;
                }
            }
            cv::Point2i final_point = cv::Point2i(start_point + direction * (dist - 1));
            expanded_polygon.push_back(final_point);
        }
    }

    vector<cv::Point2i> hull;
    cv::convexHull(expanded_polygon, hull);
    return hull;
}

// ========================================================
// Funciones de transformación y calibración
// ========================================================
cv::Point2f ComputerVisionWeb::imageToScene(cv::Point2i p){
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0,0) = p.x * this->calib_w / this->real_w;
    pin.at<double>(1,0) = p.y * this->calib_h / this->real_h;
    pin.at<double>(2,0) = 1;
    cv::Mat pout = this->H * pin;
    return cv::Point2f(
        float(pout.at<double>(0,0) / pout.at<double>(2,0)),
        float(pout.at<double>(1,0) / pout.at<double>(2,0))
    );
}

cv::Point2f ComputerVisionWeb::transformInv(cv::Point2f p){
    cv::Mat pin(3, 1, CV_64FC1);
    pin.at<double>(0, 0) = (p.x * calib_w) / real_w;
    pin.at<double>(1, 0) = (p.y * calib_h) / real_h;
    pin.at<double>(2, 0) = 1;

    cv::Mat pout = pin; 
    return cv::Point2f(
        float(pout.at<double>(0, 0) / pout.at<double>(2, 0)),
        float(pout.at<double>(1, 0) / pout.at<double>(2, 0))
    );
}

cv::Mat ComputerVisionWeb::recalibrateHomography(){
    vector<cv::Point2f> scenePoints(9);
    scenePoints[0] = cv::Point2f(141.421356237f, 141.421356237f);
    scenePoints[1] = cv::Point2f(0.f, 200.f);
    scenePoints[2] = cv::Point2f(-141.421356237f, 141.421356237f);
    scenePoints[3] = cv::Point2f(200.f, 0.f);
    scenePoints[4] = cv::Point2f(0.f, 0.f);
    scenePoints[5] = cv::Point2f(-200.f, 0.f);
    scenePoints[6] = cv::Point2f(141.421356237f, -141.421356237f);
    scenePoints[7] = cv::Point2f(0.f, -200.f);
    scenePoints[8] = cv::Point2f(-141.421356237f, -141.421356237f);

    return cv::findHomography(contourCenters, scenePoints, cv::RANSAC, 5);
}

cv::Point2f ComputerVisionWeb::getStepPosition(int frame, cv::Rect &feet){
    return cv::Point2f(feet.x + feet.width/2.0f,
                       feet.y + feet.height/2.0f);
}

// ============================
// Cálculo de distancias
// ============================
float ComputerVisionWeb::distance(cv::Point2f &p1, cv::Point2f &p2){
    float dx = p1.x - p2.x, dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

float ComputerVisionWeb::calculatePointToLineDistance(const cv::Point2f &pointA,
                                                      const cv::Point2f &pointB,
                                                      const cv::Point2f &point)
{
    float segmentLength = cv::norm(pointB - pointA);
    if (segmentLength == 0.0f)
        return cv::norm(point - pointA);

    float t = ((point.x - pointA.x)*(pointB.x - pointA.x) +
               (point.y - pointA.y)*(pointB.y - pointA.y))
              / (segmentLength * segmentLength);

    t = max(0.0f, min(1.0f, t));
    cv::Point2f projection = pointA + t*(pointB - pointA);
    return cv::norm(point - projection);
}

// ========================================================================
// Intersección de pies con contorno
// ========================================================================
bool ComputerVisionWeb::feetIntersectsObjective(vector<cv::Point2i> &footPolygon,
                                                vector<cv::Point2i> &contour)
{
    return isPolygonIntersection(footPolygon, contour);
}

// ========================================================================
// IntersectsObjective
// ========================================================================
int ComputerVisionWeb::intersectsObjective(cv::Mat img,
                                           int index,
                                           vector<cv::Point2i> &leftStep,
                                           bool leftStepOccurred,
                                           vector<cv::Point2i> &rightStep,
                                           bool rightStepOccurred)
{
    auto minSceneDistanceToRectContour = [&](const vector<cv::Point2i> &stepPolygon,
                                             const vector<cv::Point2f> &scene_contour)
    {
        float minDistance = FLT_MAX;
        for (const auto &polygonPoint : stepPolygon)
        {
            cv::Point2f scenePoint = imageToScene(cv::Point2f(polygonPoint));
            for (int i = 0; i < (int)scene_contour.size(); ++i)
            {
                float distance = calculatePointToLineDistance(
                                     scene_contour[i],
                                     scene_contour[(i + 1) % scene_contour.size()],
                                     scenePoint);
                minDistance = std::min(minDistance, distance);
            }
        }
        return minDistance;
    };

    float distance_left = 0,  minDistance_left = FLT_MAX;
    float distance_right = 0, minDistance_right= FLT_MAX;
    float id_minDist_left  = FLT_MAX;
    float id_minDist_right = FLT_MAX;

    bool intersects_left  = false;
    bool intersects_right = false;
    bool flagIntersect    = false;

#ifdef SHOW_FINAL_RESULTS
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;
    int thickness = 2;
    for (int i = 0; i < (int)contours.size(); ++i) {
        string text = to_string(i + 1);
        vector<cv::Point> &points = contours[i].ipoints;
        if (points.size() > 1)
            cv::polylines(img, points, true, cv::Scalar(255, 0, 0), thickness);

        cv::Point2f &center = contourCenters[i];
        cv::putText(img, text,
                    cv::Point(rint(center.x), rint(center.y) - 15),
                    fontFace, fontScale, cv::Scalar(200, 255, 0), thickness);
    }
#endif

    for (int contourIndex = 0; contourIndex < (int)this->contours.size(); contourIndex++)
    {
        Contour &contour       = this->contours[contourIndex];
        Contour &scene_contour = this->contoursScene[contourIndex];

        distance_left = minSceneDistanceToRectContour(leftStep, scene_contour.points);
        if (distance_left < minDistance_left)
        {
            minDistance_left = distance_left;
            id_minDist_left = float(contourIndex);
        }

        intersects_left = feetIntersectsObjective(leftStep, contour.ipoints);
        this->left_intersects[index] = 0;

        if (intersects_left && leftStepOccurred)
        {
#ifdef SHOW_FINAL_RESULTS
            vector<cv::Point> int_points_left;
            for (const auto &pt : contour.ipoints)
                int_points_left.push_back(cv::Point((int)pt.x, (int)pt.y));
            cv::polylines(img, int_points_left, true, cv::Scalar(0, 255, 0), 2);
#endif
            this->odist1[index]        = 0;
            this->left_foot[index]     = frames_info[index].left_position.center;
            this->in_objective1[index] = contourIndex + 1;
            this->left_intersects[index] = 1;
            flagIntersect = true;
        }
        else
        {
            this->odist1[index]        = minDistance_left;
            this->in_objective1[index] = int(id_minDist_left + 1);
            this->left_foot[index]     = frames_info[index].left_position.center;
        }

        distance_right = minSceneDistanceToRectContour(rightStep, scene_contour.points);
        if (distance_right < minDistance_right)
        {
            minDistance_right = distance_right;
            id_minDist_right  = float(contourIndex);
        }

        intersects_right = feetIntersectsObjective(rightStep, contour.ipoints);
        this->right_intersects[index] = 0;

        if (intersects_right && rightStepOccurred)
        {
#ifdef SHOW_FINAL_RESULTS
            vector<cv::Point> int_points_right;
            for (const auto &pt : contour.ipoints)
                int_points_right.push_back(cv::Point((int)pt.x, (int)pt.y));
            cv::polylines(img, int_points_right, true, cv::Scalar(0, 255, 0), 2);
#endif
            this->odist2[index]         = 0;
            this->right_foot[index]     = frames_info[index].right_position.center;
            this->in_objective2[index]  = contourIndex + 1;
            this->right_intersects[index] = 1;
            flagIntersect = true;
        }
        else
        {
            this->odist2[index]        = minDistance_right;
            this->in_objective2[index] = int(id_minDist_right + 1);
            this->right_foot[index]    = frames_info[index].right_position.center;
        }

        if (flagIntersect) {
            return contourIndex;
        }
    }
    return 1000;
}

// ========================================================================
// processAvailableStepsWithCoverageArea
// ========================================================================
void ComputerVisionWeb::processAvailableStepsWithCoverageArea(int index,
                                                              int cur_objective,
                                                              cv::Mat cur_copy)
{
    int pos_correction = frames_to_store / 2 + 3;
    if (index >= pos_correction)
    {
        processStepsWithCoverageArea(index, cur_objective, cur_copy);
    }
}

// ========================================================================
// processStepsWithCoverageArea
// ========================================================================
void ComputerVisionWeb::processStepsWithCoverageArea(int index,
                                                     int cur_objective,
                                                     cv::Mat cur_copy){
    // Accedemos a pies
    vector<cv::Point2i> &left  = left_feet[index];
    vector<cv::Point2i> &right = right_feet[index];

    bool leftStepOccurred  = left_step[index];
    bool rightStepOccurred = right_step[index];

    int index_contour = intersectsObjective(cur_copy,
                                            index,
                                            left,  leftStepOccurred,
                                            right, rightStepOccurred);

#ifdef SHOW_FINAL_RESULTS
    cout << "Processed step index: " << index << endl;

    if (left_step[index])
    {
        cout << "Step on left foot: " << this->odist1[index]
             << " to " << in_objective1[index] << " objective." << endl;
        cv::Point2f p = transformInv(left_foot[index]);
        cout << "Left foot position: " << p.x << ", " << p.y << endl;
    }

    if (right_step[index])
    {
        cout << "Step on right foot: " << this->odist2[index]
             << " to " << in_objective2[index] << " objective." << endl;
        cv::Point2f p = transformInv(right_foot[index]);
        cout << "Right foot position: " << p.x << ", " << p.y << endl;
    }

    if (!isVideoWriterInitialized && !cur_copy.empty())
    {
        int frame_width  = cur_copy.cols;
        int frame_height = cur_copy.rows;
        videoWriter.open("output_video.avi",
                         cv::VideoWriter::fourcc('M','J','P','G'),
                         30,
                         cv::Size(frame_width, frame_height),
                         true);

        if (!videoWriter.isOpened())
            cout << "Error: Could not open the video file for writing" << endl;
        else
            isVideoWriterInitialized = true;
    }

    if (!cur_copy.empty())
    {
        int fontFace  = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.8;
        int thickness = 2;

        cv::Scalar left_color  = leftStepOccurred
                                 ? cv::Scalar(0, 255, 0)
                                 : cv::Scalar(100, 0, 255);

        cv::Scalar right_color = rightStepOccurred
                                 ? cv::Scalar(0, 255, 0)
                                 : cv::Scalar(100, 0, 255);

        int left_thickness  = leftStepOccurred  ? 4 : 2;
        int right_thickness = rightStepOccurred ? 4 : 2;

        cv::Point2f p_left  = transformInv(left_foot[index]);
        cv::Point2f p_right = transformInv(right_foot[index]);

        cv::circle(cur_copy, cv::Point(rint(p_left.x), rint(p_left.y)),
                   3, left_color, thickness);
        string left_text = leftStepOccurred ? "L" : "";
        cv::putText(cur_copy, left_text,
                    cv::Point(rint(p_left.x)+10, rint(p_left.y)),
                    fontFace, fontScale, left_color, thickness);

        cv::circle(cur_copy, cv::Point(rint(p_right.x), rint(p_right.y)),
                   3, right_color, thickness);
        string right_text = rightStepOccurred ? "R" : "";
        cv::putText(cur_copy, right_text,
                    cv::Point(rint(p_right.x)+10, rint(p_right.y)),
                    fontFace, fontScale, right_color, thickness);

        cv::polylines(cur_copy, left_feet[index], true, left_color, left_thickness);
        cv::polylines(cur_copy, right_feet[index], true, right_color, right_thickness);

        string frame_text = "Frame: " + to_string(index);
        cv::putText(cur_copy, frame_text, cv::Point(10, 30),
                    fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);

        vector<string> textData = {
            "left_intersects: " + to_string(this->left_intersects[index]),
            "right_intersects: " + to_string(this->right_intersects[index]),
            "odist1: " + to_string(this->odist1[index]),
            "odist2: " + to_string(this->odist2[index]),
            "in_objective1: " + to_string(this->in_objective1[index]),
            "in_objective2: " + to_string(this->in_objective2[index]),
            "left_foot: (" + to_string(int(this->left_foot[index].x)) + ", " +
                             to_string(int(this->left_foot[index].y)) + ")",
            "right_foot: (" + to_string(int(this->right_foot[index].x)) + ", " +
                              to_string(int(this->right_foot[index].y)) + ")"
        };

        int yOffset = 30;
        for (int i = 0; i < (int)textData.size(); ++i)
        {
            cv::putText(cur_copy, textData[i],
                        cv::Point(cur_copy.cols - 300, 20 + i*yOffset),
                        fontFace, 0.5, cv::Scalar(0, 0, 255), 2);
        }

        if (showFrames)
        {
            cv::resizeWindow("Everything", 1280, 720);
            cv::imshow("Everything", cur_copy);
            int key = cv::waitKey(100);
            if (key == 'q' || key == 'Q')
            {
                showFrames = false;
                cv::destroyWindow("Everything");
            }
        }

        if (videoWriter.isOpened())
            videoWriter.write(cur_copy);
    }
#endif
}

// ========================================================================
// callApi
// ========================================================================
size_t writeCallback(void* contents, size_t size, size_t nmemb, string* s)
{
    size_t newLength = size * nmemb;
    s->append((char*)contents, newLength);
    return newLength;
}

bool ComputerVisionWeb::callApi(const string& videoUrl){
    CURL* curl;
    CURLcode res;
    string readBuffer;
    ImagemConverter converter; // de ConvertImage.h

    curl = curl_easy_init();
    if (curl)
    {
        // Local Docker -> "http://localhost:5000" 
        // AWS Docker -> "http://blazepose-api:5000"
        string api_url = "http://blazepose-api-segmentation:5000/process-video";
        // string api_url = "http://localhost:5000";

        string json_payload = "{\"video_url\": \"" + videoUrl + "\"}";

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);

        if(res != CURLE_OK)
        {
            cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
        }
        else
        {
            try
            {
                nlohmann::json responseJson = nlohmann::json::parse(readBuffer);
                nlohmann::json frames = responseJson["frames_info"];

                for (auto& frame : frames)
                {
                    FrameInfo frameInfo;
                    frameInfo.frame_index  = frame["frame_index"];
                    frameInfo.stepDetection= frame["stepDetection"];
                    frameInfo.stepSide     = frame["stepSide"];

                    // Posiciones
                    frameInfo.left_position.heel
                        = cv::Point2i(frame["left_position"]["heel"][0],
                                      frame["left_position"]["heel"][1]);
                    frameInfo.left_position.foot_index
                        = cv::Point2i(frame["left_position"]["foot_index"][0],
                                      frame["left_position"]["foot_index"][1]);
                    frameInfo.left_position.ankle
                        = cv::Point2i(frame["left_position"]["ankle"][0],
                                      frame["left_position"]["ankle"][1]);
                    frameInfo.left_position.center
                        = cv::Point2i(frame["left_position"]["center"][0],
                                      frame["left_position"]["center"][1]);

                    frameInfo.right_position.heel
                        = cv::Point2i(frame["right_position"]["heel"][0],
                                      frame["right_position"]["heel"][1]);
                    frameInfo.right_position.foot_index
                        = cv::Point2i(frame["right_position"]["foot_index"][0],
                                      frame["right_position"]["foot_index"][1]);
                    frameInfo.right_position.ankle
                        = cv::Point2i(frame["right_position"]["ankle"][0],
                                      frame["right_position"]["ankle"][1]);
                    frameInfo.right_position.center
                        = cv::Point2i(frame["right_position"]["center"][0],
                                      frame["right_position"]["center"][1]);

                    // Decodificar la máscara
                    if (frame.find("segmentation_mask") != frame.end() &&
                        !frame["segmentation_mask"].is_null())
                    {
                        string mask_base64 = frame["segmentation_mask"];
                        frameInfo.segmentation_mask = converter.str2mat(mask_base64);
                    }
                    else
                    {
                        frameInfo.segmentation_mask = cv::Mat();
                    }

                    frames_info.push_back(frameInfo);
                }
            }
            catch (const exception& e)
            {
                cerr << "Error al parsear el JSON: " << e.what() << endl;
            }
        }
        curl_easy_cleanup(curl);
        return (res == CURLE_OK);
    }
    return false;
}

// ========================================================================
// buildOutput
// ========================================================================
string ComputerVisionWeb::buildOutput(vector<MarkAndTime> sequence, int maxFrame){
    const int relevant_change = 10; //Number of centimeters for relevant movement
    const int static_step = 15;     //Number of frames for considering no displacement

    int current_seq = 0, current_center_exit1 = 0, current_center_exit2 = 0;
    int next_seq = 0;
    cv::Point p_out1, p_out2; // El punto central del pie al momento de salir del centro
    
    //Variables para estímulos
    uint n_objectives = (uint)sequence.size(); // Number of stimulis 
    vector<Section> sequences; // Sections for output
    bool first_stimulus = true;

    // Recorremos cada estímulo en la secuencia
    for (int j = 0; j < (int)n_objectives; ++j){
        Section cur_section;
        item cur_item;

        int cur_objective = sequence[j].mark_correct; // Current objective (0, .., 8)
        int cur_frame     = sequence[j].frame;        // Current frame 

#ifdef SHOW_DEBUG_TEXT
        cout << "Marking.\n\tCurrent stimulus: " << j << endl;
        cout << "\tCurrent stimulus objective: " << cur_objective << endl;
        cout << "\tCurrent stimulus index: " << cur_frame << endl;
#endif

        // Estas cosas se asumen por ser el inicio del ejercicio
        bool step_center     = true; 
        bool step_objective  = true; 
        bool right_objective = true;

        // 1) Ajustar al centro
        int i = skipUntilFootIsInCenter(current_seq, maxFrame);
        current_seq = (cur_frame >= i) ? cur_frame : i;

#ifdef SHOW_DEBUG_TEXT
        cout << "\tCurrent sequence start - after cur_frame: " << current_seq << endl;
#endif

        // 2) Buscar salida del centro
        bool step_out_detected1 = false, step_out_detected2 = false;
        findCenterExit(current_seq, maxFrame,
                       step_out_detected1, step_out_detected2,
                       current_center_exit1, current_center_exit2,
                       p_out1, p_out2);

        if (step_out_detected1 || step_out_detected2){
            // 3) Take-off
            int il_found = 0, ir_found = 0;
            bool pl_found = false, pr_found = false;

            // Pasamos relevant_change y static_step a computeTakeOff
            computeTakeOff(current_seq,
                           current_center_exit1,
                           current_center_exit2,
                           il_found,
                           ir_found,
                           pl_found,
                           pr_found,
                           relevant_change,
                           static_step);

            // Asignar “take-off” al item
            if (pl_found && pr_found){
                if (il_found < ir_found)
                {
                    cur_item.code   = 5;
                    cur_item.d_l    = odist1[il_found];
                    cur_item.d_r    = odist2[ir_found];
                    cur_item.step_l = true;
                    cur_item.step_r = false;
                    cur_item.frame  = il_found;
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = il_found;
                }
                else
                {
                    cur_item.code   = 5;
                    cur_item.d_l    = odist1[il_found];
                    cur_item.d_r    = odist2[ir_found];
                    cur_item.step_l = false;
                    cur_item.step_r = true;
                    cur_item.frame  = ir_found;
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = ir_found;
                }
            }
            else if (pl_found)
            {
                cur_item.code   = 5;
                cur_item.d_l    = odist1[il_found];
                cur_item.d_r    = odist2[ir_found];
                cur_item.step_l = true;
                cur_item.step_r = false;
                cur_item.frame  = il_found;
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = il_found;
            }
            else if (pr_found)
            {
                cur_item.code   = 5;
                cur_item.d_l    = odist1[il_found];
                cur_item.d_r    = odist2[ir_found];
                cur_item.step_l = false;
                cur_item.step_r = true;
                cur_item.frame  = ir_found;
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = ir_found;
            }
            else
            {
                cur_item.code   = 5;
                cur_item.d_l    = odist1[il_found];
                cur_item.d_r    = odist2[ir_found];
                cur_item.step_l = (bool)il_found;
                cur_item.step_r = (bool)ir_found;
                cur_item.frame  = (il_found < ir_found ? il_found : ir_found);
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = (il_found < ir_found ? il_found : ir_found);
            }

            // 4) Arrival
            int last_arrival_frame      = -1;
            int nearest_arrival_code    = 0;
            bool is_left                = true;
            bool real_arrival           = false;
            float nearest_arrival_distance = FLT_MAX;

            int earliest_center_exit = (current_center_exit1 < current_center_exit2
                                        ? current_center_exit1
                                        : current_center_exit2);

            computeArrival(
                earliest_center_exit,
                maxFrame,
                cur_objective,
                last_arrival_frame,
                nearest_arrival_code,
                is_left,
                real_arrival,
                nearest_arrival_distance
            );

            cur_item.code  = nearest_arrival_code;
            cur_item.d_l   = (last_arrival_frame >= 0 ? odist1[last_arrival_frame] : -1.f);
            cur_item.d_r   = (last_arrival_frame >= 0 ? odist2[last_arrival_frame] : -1.f);
            cur_item.step_l= (is_left ? 1 : 0);
            cur_item.step_r= (is_left ? 0 : 1);
            cur_item.frame = last_arrival_frame;

            cur_section.items.push_back(cur_item);
            cur_section.arrival_frame = last_arrival_frame;
            cur_section.arrival_code  = nearest_arrival_code;

            // Validar si llegó al objetivo correcto
            if (!real_arrival || nearest_arrival_code != cur_objective)
            {
                if (real_arrival && nearest_arrival_code != cur_objective)
                    right_objective = false;
                if (!real_arrival)
                    step_objective = false;
            }

            // 5) Retorno al centro
            bool hasReturned = checkReturnToCenter(next_seq, maxFrame);
            if (!hasReturned) step_center = false;

            // 6) Evaluar error
            cur_section.error = (right_objective && step_objective && step_center) ? false : true;
            sequences.push_back(cur_section);

            // Avanza la secuencia
            current_seq    = next_seq;
            first_stimulus = false;
        }
        else
        {
#ifdef SHOW_DEBUG_TEXT
            cout << "\tNo step out found. " << endl;
#endif
            break; // No salió del centro => se termina
        }
    }

    // 7) JSON final
    string json  = buildFinalJson(sequences);
    string json2 = buildDetailedJson(sequences);

    cout << "====================================\n" << json2 << endl;
    cout << "====================================\n" << endl;

    return json;
}

// ========================================================================
// mainFunction
// ========================================================================
string ComputerVisionWeb::mainFunction(string contourjson,
                                       string videoUrl,
                                       string imageUrl,
                                       string jsonString,
                                       string frameRate)
{
    if (callApi(videoUrl))
    {
        cout << "Procesamiento exitoso, datos recibidos desde la API de pose." << endl;
        int frame_count = 0;
        cout << "Muestra de los dos primero frames:\n" << endl;
        for (const auto& f : frames_info)
        {
            if (frame_count >= 2) break;
            cout << "Frame Index: " << f.frame_index << endl;
            cout << "Step Detection: " << (f.stepDetection ? "True" : "False") << endl;
            cout << "Step Side: " << f.stepSide << endl;
            cout << "Left Heel Position: (" << f.left_position.heel.x << ", "
                 << f.left_position.heel.y << ")" << endl;
            cout << "Right Heel Position: (" << f.right_position.heel.x << ", "
                 << f.right_position.heel.y << ")\n" << endl;
            frame_count++;
        }
    }
    else
    {
        cout << "Error al llamar a la API pose-IA." << endl;
    }

    // Parse de contornos
    {
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
                cv::Point2f p { (float)punto["x"].asInt(), (float)punto["y"].asInt() };
                contorno.points.push_back(p);
                contorno.ipoints.push_back(p);
            }
            contornos.push_back(contorno);
        }
        calib_w = stoi(string_calib_w);
        calib_h = stoi(string_calib_h);
    }

    // Parse de la secuencia
    vector<MarkAndTime> sequence;
    {
        auto json = nlohmann::json::parse(jsonString);
        for (auto &item : json)
        {
            MarkAndTime mark;
            mark.mark_correct = item["mark_correct"].get<int>();
            mark.frame        = item["frame"].get<int>();
            sequence.push_back(mark);
        }
    }

    // Descarga
    downloadMedia(videoUrl, imageUrl);

    // Abrir el video
    string urlVideo = "/usr/src/app/mcp-vision-detection/video.mp4";
    string urlBG    = "/usr/src/app/mcp-vision-detection/bg.jpg";

    cv::VideoCapture vtest(urlVideo);

    float f_rate = 0.0f;
    if (vtest.isOpened())
        f_rate = stof(frameRate);
    else
    {
        cout << "El video no abrio!!" << endl;
        return "Error al abrir video";
    }

    cout << "Sequence:\n";
    for (auto &m : sequence)
        cout << "Mark: " << m.mark_correct << ", Time: " << m.frame << endl;

    cout << "URL del video procesado: " << videoUrl << endl;
    cout << "URL de la imagen de fondo procesada: " << imageUrl << endl;
    cout << "Frame rate del video: " << f_rate << endl;
    cout << "calib_w: " << calib_w << endl;
    cout << "calib_h: " << calib_h << endl;

    cv::Mat current;
    bool first = true;
    unsigned int frame = 0, maxFrame = 0;
    unsigned int time = 0, msec_per_frame = (unsigned int)(1000.0f / f_rate);

    cv::Mat bg = cv::imread(urlBG);

    map<int,int> msecs;
    while(true)
    {
        vtest >> current;
        if (current.empty()) break;

        ++frame;
        msecs[frame] = time;
        time += msec_per_frame;

        if (first)
        {
            first = false;
            real_w = current.cols;
            real_h = current.rows;

            if (real_w != bg.cols || real_h != bg.rows)
                cv::resize(bg, bg, current.size());

            float scaleX = float(real_w) / float(calib_w);
            float scaleY = float(real_h) / float(calib_h);

            for (auto &ct : contornos)
            {
                for (auto &p : ct.points)
                {
                    p.x *= scaleX;
                    p.y *= scaleY;
                }
            }
        }
    }
    vtest.release();
    maxFrame = frame;

    // Se calculan contornos
    contours = contornos;
    {
        float xx, yy;
        int n;
        for (auto &c : contornos)
        {
            xx = yy = 0;
            n = 0;
            for (auto &p : c.points)
            {
                xx += p.x;
                yy += p.y;
                ++n;
            }
            contourCenters.push_back(cv::Point2f(xx/n, yy/n));
        }
    }

    // Recalibrar homografía
    H = recalibrateHomography();

    // Proyectar contornos a "contoursScene"
    for (auto &c : contornos)
    {
        Contour cont;
        cont.indiceContorno = c.indiceContorno;
        cont.x = c.x; cont.y = c.y; cont.z = c.z;
        for (auto &p : c.points)
            cont.points.push_back(imageToScene(p));
        contoursScene.push_back(cont);
    }

    cout << "Recalibration: " << endl;
    for (int i=0; i<(int)contourCenters.size(); i++)
    {
        cv::Point2f p = imageToScene(contourCenters[i]);
        cout << "\tObjective " << i+1 << ": " << p.x << ", " << p.y << endl;
        contourCentersScene.push_back(p);
    }

    // Inicializar arrays
    left_foot.resize(maxFrame);
    right_foot.resize(maxFrame);
    left_step.resize(maxFrame, false);
    right_step.resize(maxFrame, false);
    left_rects_s.resize(maxFrame);
    right_rects_s.resize(maxFrame);
    in_objective1.resize(maxFrame, 0);
    in_objective2.resize(maxFrame, 0);
    odist1.resize(maxFrame, 0.f);
    odist2.resize(maxFrame, 0.f);
    left_intersects.resize(maxFrame, 0);
    right_intersects.resize(maxFrame, 0);
    left_feet.resize(maxFrame);
    right_feet.resize(maxFrame);

    // Expand distance (para makeExpandedPolygon)
    float expand_distance = 10.0f;

    for (int i = 0; i < (int)maxFrame; ++i)
    {
        cv::Mat foot_mask = frames_info[i].segmentation_mask;

        left_feet[i]  = makeExpandedPolygon(frames_info[i].left_position,
                                            foot_mask,
                                            expand_distance);
        right_feet[i] = makeExpandedPolygon(frames_info[i].right_position,
                                            foot_mask,
                                            expand_distance);

        if (frames_info[i].stepDetection)
        {
            if (frames_info[i].stepSide == "Both")
            {
                left_step[i]  = true;
                right_step[i] = true;
            }
            else if (frames_info[i].stepSide == "Left")
            {
                left_step[i]  = true;
            }
            else if (frames_info[i].stepSide == "Right")
            {
                right_step[i] = true;
            }
        }
    }

    // Re-abrir video
    vtest.open(urlVideo);
    if (!vtest.isOpened())
    {
        cout << "El video no abrio la segunda vez!!" << endl;
        return "El video no abrio la segunda vez!!";
    }

    auto frame_it = msecs.begin();
    unsigned int j_cur = 0;
    unsigned int nObjs = (unsigned int)sequence.size();
    int cur_objective = (nObjs > 0) ? sequence[0].mark_correct : -1;

    cv::Mat cur_copy;
    for (unsigned int i = 0; i < maxFrame; ++i)
    {
        frame = frame_it->first;
        vtest >> current;

        // Ajustar objective actual
        for (unsigned int j = j_cur; j < nObjs; ++j)
        {
            MarkAndTime &m = sequence[j];
            if (frame < (unsigned int)m.frame) break;
            cur_objective = m.mark_correct;
            j_cur = j;
        }

#ifdef SHOW_INTERMEDIATE_RESULTS
        current.copyTo(cur_copy);
#endif
        processAvailableStepsWithCoverageArea(i, cur_objective, cur_copy);
        frame_it++;
    }
    vtest.release();

    // Por último, llamamos buildOutput
    string out = buildOutput(sequence, maxFrame);

#ifdef SHOW_FINAL_RESULTS
    cout << "============ OUT ============ \n" << out << endl;
    cout << "============ END ============ \n" << endl;
#endif

    return out;
}

// ========================================================================
// aux functions for buildOutput
// ========================================================================
int ComputerVisionWeb::skipUntilFootIsInCenter(int startFrame, int maxFrame){
    for (int i = startFrame; i < maxFrame; ++i)
    {
        if (in_objective1[i] == 5 || in_objective2[i] == 5)
            return i;
    }
    return maxFrame; 
}

void ComputerVisionWeb::findCenterExit(int startFrame, int maxFrame,
                                       bool &step_out_detected1, bool &step_out_detected2,
                                       int &current_center_exit1, int &current_center_exit2,
                                       cv::Point &p_out1, cv::Point &p_out2)
{
    // Pie izquierdo
    for (int i = startFrame; i < maxFrame; ++i)
    {
        if (!left_step[i]) continue;
        if (in_objective1[i] != 5)
        {
            current_center_exit1 = i;
            p_out1 = cv::Point(left_foot[i].x, left_foot[i].y);
            step_out_detected1 = true;
            break;
        }
    }

    // Pie derecho
    for (int i = startFrame; i < maxFrame; ++i)
    {
        if (!right_step[i]) continue;
        if (in_objective2[i] != 5)
        {
            current_center_exit2 = i;
            p_out2 = cv::Point(right_foot[i].x, right_foot[i].y);
            step_out_detected2 = true;
            break;
        }
    }

    // Si uno no se detectó, copiamos del otro
    if (!step_out_detected1 && step_out_detected2)
    {
        p_out1 = p_out2;
        current_center_exit1 = current_center_exit2;
    }
    else if (!step_out_detected2 && step_out_detected1)
    {
        p_out2 = p_out1;
        current_center_exit2 = current_center_exit1;
    }
}

void ComputerVisionWeb::computeTakeOff(int current_seq,
                                       int sure_frame1,
                                       int sure_frame2,
                                       int &il_found,
                                       int &ir_found,
                                       bool &pl_found,
                                       bool &pr_found,
                                       int relevant_change,
                                       int static_step){
    // =====================
    // Pie izquierdo
    // =====================
    {
        bool stepping = false, first = true;
        int il_last = current_seq + 1;
        int il_last_stepping = 0;
        int lindex_1 = 0, lindex_2 = 0;
        il_found = sure_frame1 + 1;
        pl_found = false;

        // Convertimos a "scene" la posición de salida
        cv::Point2f p_out_s = imageToScene(cv::Point2f(left_foot[sure_frame1]));
        cv::Point2f p_center = contourCentersScene[4]; // marca 5
        float d_center_obj = cv::norm(p_out_s - p_center);

        auto projectVector = [&](cv::Point2f v, cv::Point2f onto){
            float dotp = v.x * onto.x + v.y * onto.y;
            float onto_len_sq = onto.x*onto.x + onto.y*onto.y;
            if (onto_len_sq == 0) return cv::Point2f(0,0);
            float scale = dotp / onto_len_sq;
            return cv::Point2f(onto.x * scale, onto.y * scale);
        };
        auto magnitude = [&](cv::Point2f v){
            return sqrt(v.x*v.x + v.y*v.y);
        };

        for (int i = sure_frame1; i >= current_seq; --i)
        {
            if (first)
            {
                if (!stepping && left_step[i])
                {
                    stepping   = true;
                    il_last    = i;
                    lindex_1   = i;
                }
                else if (stepping && !left_step[i])
                {
                    il_last_stepping = i - 1;
                    stepping = false;
                    first    = false;
                }
            }
            else
            {
                if (left_step[i])
                {
                    il_last  = i;
                    lindex_2 = i;

                    cv::Point2f p1 = imageToScene(left_foot[lindex_1]);
                    cv::Point2f p2 = imageToScene(left_foot[lindex_2]);
                    float d  = cv::norm(p2 - p1);

                    float d1 = magnitude(projectVector(
                                cv::Point2f(p1.x - p_out_s.x, p1.y - p_out_s.y),
                                cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                    float d2 = magnitude(projectVector(
                                cv::Point2f(p2.x - p_out_s.x, p2.y - p_out_s.y),
                                cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));

                    if (d1 < d2 && d > relevant_change &&
                        (lindex_1 - il_last_stepping) < static_step &&
                        d2 < d_center_obj && odist1[i] != 0)
                    {
                        stepping   = true;
                        first      = true;
                        lindex_1   = lindex_2;
                        continue;
                    }
                    pl_found = true;
                    if ((lindex_1 - il_last_stepping) >= static_step)
                    {
                        il_found = lindex_1 + 1;
                        break;
                    }
                    if (d <= relevant_change)
                    {
                        il_found = lindex_2 + 1;
                        break;
                    }
                    if (d1 > d2)
                        il_found = lindex_1 + 1;
                    else
                        il_found = lindex_2 + 1;
                    break;
                }
            }
        }

        if (!pl_found)
        {
            il_found = il_last + 1;
            pl_found = true;
        }
    }

    // =====================
    // Pie derecho
    // =====================
    {
        bool stepping = false, first = true;
        int ir_last = current_seq + 1;
        int ir_last_stepping = 0;
        int rindex_1 = 0, rindex_2 = 0;
        ir_found = sure_frame2 + 1;
        pr_found = false;

        cv::Point2f p_out_s = imageToScene(cv::Point2f(right_foot[sure_frame2]));
        cv::Point2f p_center = contourCentersScene[4]; 
        float d_center_obj = cv::norm(p_out_s - p_center);

        auto projectVector = [&](cv::Point2f v, cv::Point2f onto){
            float dotp = v.x * onto.x + v.y * onto.y;
            float onto_len_sq = onto.x*onto.x + onto.y*onto.y;
            if (onto_len_sq == 0) return cv::Point2f(0,0);
            float scale = dotp / onto_len_sq;
            return cv::Point2f(onto.x * scale, onto.y * scale);
        };
        auto magnitude = [&](cv::Point2f v){
            return sqrt(v.x*v.x + v.y*v.y);
        };

        for (int i = sure_frame2; i >= current_seq; --i)
        {
            if (first)
            {
                if (!stepping && right_step[i])
                {
                    stepping   = true;
                    ir_last    = i;
                    rindex_1   = i;
                }
                else if (stepping && !right_step[i])
                {
                    ir_last_stepping = i + 1;
                    stepping = false;
                    first    = false;
                }
            }
            else
            {
                if (right_step[i])
                {
                    ir_last  = i;
                    rindex_2 = i;

                    cv::Point2f p1 = imageToScene(right_foot[rindex_1]);
                    cv::Point2f p2 = imageToScene(right_foot[rindex_2]);
                    float d  = cv::norm(p2 - p1);

                    float d1 = magnitude(projectVector(
                                cv::Point2f(p1.x - p_out_s.x, p1.y - p_out_s.y),
                                cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));
                    float d2 = magnitude(projectVector(
                                cv::Point2f(p2.x - p_out_s.x, p2.y - p_out_s.y),
                                cv::Point2f(p_center.x - p_out_s.x, p_center.y - p_out_s.y)));

                    if (d1 < d2 && d > relevant_change &&
                        (rindex_1 - ir_last_stepping) < static_step &&
                        d2 < d_center_obj && odist2[i] != 0)
                    {
                        stepping   = true;
                        first      = true;
                        rindex_1   = rindex_2;
                        continue;
                    }
                    pr_found = true;
                    if ((rindex_1 - ir_last_stepping) >= static_step)
                    {
                        ir_found = rindex_1 + 1;
                        break;
                    }
                    if (d <= relevant_change)
                    {
                        ir_found = rindex_2 + 1;
                        break;
                    }
                    if (d1 > d2)
                        ir_found = rindex_1 + 1;
                    else
                        ir_found = rindex_2 + 1;
                    break;
                }
            }
        }
        if (!pr_found)
        {
            ir_found = ir_last + 1;
            pr_found = true;
        }
    }
}

void ComputerVisionWeb::computeArrival(int startFrame,
                                       int maxFrame,
                                       int cur_objective,
                                       int &last_arrival_frame,
                                       int &nearest_arrival_code,
                                       bool &is_left,
                                       bool &real_arrival,
                                       float &nearest_arrival_distance){
    bool still_near_center_l = true, still_near_center_r = true;
    bool done = false;
    int last_real = -1;

    for (int i2 = startFrame; i2 < maxFrame && !done; ++i2)
    {
        // Pie izquierdo
        if (in_objective1[i2] != 5)
        {
            still_near_center_l = false;
            if (left_step[i2])
            {
                // Si odist1[i2]==0 => intersección real
                if (odist1[i2] == 0 && last_real != in_objective1[i2])
                {
                    last_arrival_frame       = i2;
                    last_real                = nearest_arrival_code = in_objective1[i2];
                    nearest_arrival_distance = 0;
                    real_arrival             = true;
                    is_left                  = true;
                }
                if (!real_arrival && odist1[i2] < nearest_arrival_distance)
                {
                    nearest_arrival_distance = odist1[i2];
                    last_arrival_frame       = i2;
                    nearest_arrival_code     = in_objective1[i2];
                    is_left                  = true;
                }
            }
        }

        // Pie derecho
        if (in_objective2[i2] != 5)
        {
            still_near_center_r = false;
            if (right_step[i2])
            {
                if (odist2[i2] == 0 && last_real != in_objective2[i2])
                {
                    last_arrival_frame       = i2;
                    last_real                = nearest_arrival_code = in_objective2[i2];
                    nearest_arrival_distance = 0;
                    real_arrival             = true;
                    is_left                  = false;
                }
                if (!real_arrival && odist2[i2] < nearest_arrival_distance)
                {
                    nearest_arrival_distance = odist2[i2];
                    last_arrival_frame       = i2;
                    nearest_arrival_code     = in_objective2[i2];
                    is_left                  = false;
                }
            }
        }

        // Revisar si ya regresó al centro con uno de los pies
        if (!still_near_center_l && in_objective1[i2] == 5 && left_step[i2])
            done = true;
        if (!still_near_center_r && in_objective2[i2] == 5 && right_step[i2])
            done = true;
    }
}

bool ComputerVisionWeb::checkReturnToCenter(int startSearchFrame, int maxFrame){
    bool still_far_center_l = true, still_far_center_r = true;
    bool step_center = false;
    bool ready_l = false, ready_r = false;

    for (int i2 = startSearchFrame; i2 < maxFrame; ++i2)
    {
        if (in_objective1[i2] == 5)
        {
            still_far_center_l = false;
            if (left_step[i2])
            {
                step_center = true;
                break;
            }
        }
        if (in_objective2[i2] == 5)
        {
            still_far_center_r = false;
            if (right_step[i2])
            {
                step_center = true;
                break;
            }
        }
        if(!still_far_center_l && in_objective1[i2] != 5)
            ready_l = true;
        if(!still_far_center_r && in_objective2[i2] != 5)
            ready_r = true;
        if (ready_l && ready_r)
            break;
    }
    return step_center;
}

string ComputerVisionWeb::buildFinalJson(const vector<Section> &sequences){
    string json = "[\n";
    for(size_t i = 0; i < sequences.size(); ++i)
    {
        const Section& sec = sequences[i];
        json += "    {\n";
        json += "    \"id_sequence\": " + to_string(i) + ",\n";
        json += "    \"takeoff_frame\": " + to_string(sec.takeoff_frame) + ",\n";
        json += "    \"arrival_frame\": " + to_string(sec.arrival_frame) + ",\n";
        json += "    \"error\": " + string(sec.error ? "true" : "false") + "\n";
        json += "    }";
        if (i < sequences.size() - 1) json += ",";
        json += "\n";
    }
    json += "]";
    return json;
}

string ComputerVisionWeb::buildDetailedJson(const vector<Section> &sequences){
    string json2 = "[\n";
    for(size_t i = 0; i < sequences.size(); ++i)
    {
        const Section& sec = sequences[i];
        json2 += "    {\n";
        json2 += "    \"id_sequence\": " + to_string(i) + ",\n";
        json2 += "    \"takeoff_frame\": " + to_string(sec.takeoff_frame) + ",\n";
        json2 += "    \"arrival_frame\": " + to_string(sec.arrival_frame) + ",\n";
        json2 += "    \"error\": " + string(sec.error ? "true" : "false") + ",\n";
        json2 += "    \"items\": [\n";
        for (size_t j = 0; j < sec.items.size(); ++j)
        {
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
        if (i < sequences.size() - 1) json2 += ",";
        json2 += "\n";
    }
    json2 += "]";
    return json2;
}