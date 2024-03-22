#include "ComputerVisionWeb.h"

ComputerVisionWeb::ComputerVisionWeb()
{
}


void ComputerVisionWeb::initTracker(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog)
{
    //// MOG2 ////
    int historyMOG = 200;
    bool bShadowDetection = true;
    /* //PREVIOS:
        int varThreshold = 45;
        int  nmixtures = 3;
        double backgroundRatio = 0.6;
    */
    int varThreshold = 35;
    int nmixtures = 3;
    double backgroundRatio = 0.6;

    mog = cv::createBackgroundSubtractorMOG2(historyMOG, varThreshold,
                                             bShadowDetection);
    mog->setNMixtures(nmixtures);
    mog->setBackgroundRatio(backgroundRatio);
    mog->setShadowValue(0);
    mog->setShadowThreshold(0.3);
    mog->setDetectShadows(true);
}

void ComputerVisionWeb::trainMog(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog, cv::Mat &img, cv::Mat &fg, cv::Mat &bg, double learningRate)
{
    bool inter_bg = true; // Activate intermittent background learning
    if (inter_bg)
        mog->apply(bg, fg, 2 * learningRate); // Double the reinforce over moving objects
    mog->apply(img, fg, learningRate);
}

void ComputerVisionWeb::trainMog(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog, cv::Mat &img, cv::Mat &fg, double learningRate)
{
    mog->apply(img, fg, learningRate);
}

cv::Mat ComputerVisionWeb::maskBiggest(cv::Mat &fg, cv::Mat &labels, cv::Mat &stats, cv::Mat &big_mask)
{

    int i, j, x, y, w, h, bindex = -1, rnum, max = 0, cnum = stats.rows;
    big_mask = cv::Mat::zeros(fg.size(), CV_8UC1);
    cv::Mat r = cv::Mat::zeros(5, 1, CV_32SC1);

    for (i = 1; i < cnum; ++i)
    {
        rnum = stats.at<int>(i, 4);
        if (rnum > max)
        {
            max = rnum;
            bindex = i;
            x = stats.at<int>(i, 0);
            y = stats.at<int>(i, 1);
            w = stats.at<int>(i, 2);
            h = stats.at<int>(i, 3);
        }
    }

    if (bindex == -1)
    { // No intersecting area
        std::cout << "NO Blob!!" << std::endl;
        return r;
    }
    std::cout << "Biggest index is " << bindex << " with " << max << " pixels." << std::endl;

    int x1 = x, y1 = y, x2 = x + w - 1, y2 = y + h - 1;

    r.at<int>(0) = x;
    r.at<int>(1) = y;
    r.at<int>(2) = w;
    r.at<int>(3) = h;
    r.at<int>(4) = bindex;
    int ostep = big_mask.step;
    uchar *odata = big_mask.data;
    for (i = y1; i <= y2; ++i)
        for (j = x1; j <= x2; ++j)
            if (labels.at<int>(i, j) == bindex)
                odata[i * ostep + j] = 255;

    return r;
}

//// Segmentation and Foot Boxes ////
cv::Mat ComputerVisionWeb::presegmentation(cv::Ptr<cv::BackgroundSubtractorMOG2> mog, cv::Mat &current, cv::Mat &labels, cv::Mat &r)
{

    cv::Mat processMasked, foreGround, centroids, stats;

    foreGround = cv::Mat::zeros(current.size(), CV_8UC1);
    mog->apply(current, foreGround, 0);
    cv::dilate(foreGround, foreGround, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
    cv::erode(foreGround, foreGround, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("Full Segmentation", foreGround);
#endif

    cv::connectedComponentsWithStats(foreGround, labels, stats, centroids, 4, CV_32S);

    cv::Mat big_mask;
    r = maskBiggest(foreGround, labels, stats, big_mask);

    return big_mask;
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

// Per frame: Two feet. By foot: (x y w h code xp yp d)
//  (x,y,w,h): foot rect                (left_step, right_step)
//  code:                               (in_objective1, in_objective2)
//      0: No step
//    1-9: Step to nearest objective
//  (xp,yp): Feet contact point         (left_foot, right_foot)
//  d: distance to nearest center       (odist1, odist2)
std::string ComputerVisionWeb::buildJsonData(FeetTracker &ft)
{

    std::string output = "";
    std::string output_l_x = "";
    std::string output_l_y = "";
    std::string output_l_w = "";
    std::string output_l_h = "";
    std::string output_l_code = "";
    std::string output_l_xp = "";
    std::string output_l_yp = "";
    std::string output_l_d = "";
    std::string output_l_intersects = "";
    std::string output_l_step = "";

    std::string output_r_x = "";
    std::string output_r_y = "";
    std::string output_r_w = "";
    std::string output_r_h = "";
    std::string output_r_code = "";
    std::string output_r_xp = "";
    std::string output_r_yp = "";
    std::string output_r_d = "";
    std::string output_r_intersects = "";
    std::string output_r_step = "";

    int i, n = ft.left_step.size();

    for (i = 0; i < n; ++i)
    {
        cv::Rect &rl = ft.left_rects_s[i];
        cv::Rect &rr = ft.right_rects_s[i];
        output_l_x += "{\"integerValue\": " + std::to_string(rl.x) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_y += "{\"integerValue\": " + std::to_string(rl.y) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_w += "{\"integerValue\": " + std::to_string(rl.width) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_h += "{\"integerValue\": " + std::to_string(rl.height) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_code += "{\"integerValue\": " + std::to_string(ft.in_objective1[i] + 1) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_xp += "{\"doubleValue\": " + std::to_string(ft.left_foot[i].x) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_yp += "{\"doubleValue\": " + std::to_string(ft.left_foot[i].y) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_d += "{\"doubleValue\": " + std::to_string(ft.odist1[i]) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_intersects += "{\"integerValue\": " + std::to_string(ft.left_intersects[i]) + ",\"frame\": " + std::to_string(i) + "},";
        output_l_step += "{\"boolValue\": " + std::to_string(ft.left_step[i]) + ",\"frame\": " + std::to_string(i) + "},";

        output_r_x += "{\"integerValue\": " + std::to_string(rr.x) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_y += "{\"integerValue\": " + std::to_string(rr.y) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_w += "{\"integerValue\": " + std::to_string(rr.width) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_h += "{\"integerValue\": " + std::to_string(rr.height) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_code += "{\"integerValue\": " + std::to_string(ft.in_objective2[i] + 1) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_xp += "{\"doubleValue\": " + std::to_string(ft.right_foot[i].x) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_yp += "{\"doubleValue\": " + std::to_string(ft.right_foot[i].y) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_d += "{\"doubleValue\": " + std::to_string(ft.odist2[i]) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_intersects += "{\"integerValue\": " + std::to_string(ft.right_intersects[i]) + ",\"frame\": " + std::to_string(i) + "},";
        output_r_step += "{\"boolValue\": " + std::to_string(ft.right_step[i]) + ",\"frame\": " + std::to_string(i) + "},";
    }

    output = "{\"fields\" : {";
    output += "\"Width\":{\"integerValue\": " + std::to_string(ft.real_w) + "},";
    output += "\"Height\":{\"integerValue\": " + std::to_string(ft.real_h) + "},";
    output += "\"Total frames\":{\"integerValue\": " + std::to_string(n) + "},";

    output += "\"Left\": { \"mapValue\": { \"fields\": {";
    output += "\"d\": { \"arrayValue\": {\"values\": [" + output_l_d.substr(0, output_l_d.size() - 1) + "] } },";
    output += "\"code\": { \"arrayValue\": { \"values\": [" + output_l_code.substr(0, output_l_code.size() - 1) + "] } },";
    output += "\"y\": { \"arrayValue\": { \"values\": [" + output_l_y.substr(0, output_l_y.size() - 1) + "] } },";
    output += "\"yp\": { \"arrayValue\": { \"values\": [" + output_l_yp.substr(0, output_l_yp.size() - 1) + "] } },";
    output += "\"x\": { \"arrayValue\": { \"values\": [" + output_l_x.substr(0, output_l_x.size() - 1) + "] } },";
    output += "\"w\": { \"arrayValue\": { \"values\": [" + output_l_w.substr(0, output_l_w.size() - 1) + "] } },";
    output += "\"xp\": { \"arrayValue\": { \"values\": [" + output_l_xp.substr(0, output_l_xp.size() - 1) + "] } },";
    output += "\"h\": { \"arrayValue\": { \"values\": [" + output_l_h.substr(0, output_l_h.size() - 1) + "] } },";
    output += "\"intersects\": { \"arrayValue\": { \"values\": [" + output_l_intersects.substr(0, output_l_intersects.size() - 1) + "] } },";
    output += "\"step\": { \"arrayValue\": { \"values\": [" + output_l_step.substr(0, output_l_step.size() - 1) + "] } }";
    output += "}}},";

    output += "\"Right\": { \"mapValue\": { \"fields\": {";
    output += "\"d\": { \"arrayValue\": {\"values\": [" + output_r_d.substr(0, output_r_d.size() - 1) + "] } },";
    output += "\"code\": { \"arrayValue\": { \"values\": [" + output_r_code.substr(0, output_r_code.size() - 1) + "] } },";
    output += "\"y\": { \"arrayValue\": { \"values\": [" + output_r_y.substr(0, output_r_y.size() - 1) + "] } },";
    output += "\"yp\": { \"arrayValue\": { \"values\": [" + output_r_yp.substr(0, output_r_yp.size() - 1) + "] } },";
    output += "\"x\": { \"arrayValue\": { \"values\": [" + output_r_x.substr(0, output_r_x.size() - 1) + "] } },";
    output += "\"w\": { \"arrayValue\": { \"values\": [" + output_r_w.substr(0, output_r_w.size() - 1) + "] } },";
    output += "\"xp\": { \"arrayValue\": { \"values\": [" + output_r_xp.substr(0, output_r_xp.size() - 1) + "] } },";
    output += "\"h\": { \"arrayValue\": { \"values\": [" + output_r_h.substr(0, output_r_h.size() - 1) + "] } },";
    output += "\"intersects\": { \"arrayValue\": { \"values\": [" + output_r_intersects.substr(0, output_r_intersects.size() - 1) + "] } },";
    output += "\"step\": { \"arrayValue\": { \"values\": [" + output_r_step.substr(0, output_r_step.size() - 1) + "] } }";
    output += "}}}";
    output += "}}";

    return output;
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
    auto j = json::parse(jsonString);
    std::vector<MarkAndTime> marks;
    
    for (const auto& item : j) {
        MarkAndTime mark;
        mark.mark_correct = item["mark_correct"].get<int>();
        mark.frame = item["frame"].get<int>();
        marks.push_back(mark);
    }
    
    return marks;
}

std::vector<item> filterCloseFrames(const std::vector<item> &marks)
{
    std::vector<item> filteredMarks;   // Marcas después de filtrar las cercanas.
    filteredMarks.push_back(marks[0]); // Añade la primera marca como punto de partida.

    for (size_t i = 1; i < marks.size(); ++i)
    {
        const auto &currentMark = marks[i];
        auto &lastAddedMark = filteredMarks.back();

        if (currentMark.frame - lastAddedMark.frame >= 20)
        {
            filteredMarks.push_back(currentMark);
        }
    }

    return filteredMarks; // Retorna las marcas filtradas.
}

std::vector<item> insertUniqueIntersectMarks(const std::vector<item> &items)
{
    std::vector<item> aux_sequence;

    bool isSearching = true; // Estado para buscar el inicio de una nueva secuencia de interesects.

    for (size_t i = 0; i < items.size(); ++i)
    {
        if (items[i].code == 5 && items[i].intersects == 1)
        {
            if (isSearching)
            {
                // Encuentra el inicio de una nueva secuencia y lo agrega a la secuencia.
                aux_sequence.push_back(items[i]);
                isSearching = false; // Detiene la búsqueda hasta encontrar un ítem con diferentes características.
            }
        }
        else
        {
            isSearching = true; // Reactiva la búsqueda al encontrar ítems con diferentes características.
        }
    }

    std::stringstream sss;
    sss << "\n";
    for (const auto &it : aux_sequence)
    {
        sss << "Frame: " << it.frame << ", Code: " << it.code
            << ", Intersects: " << it.intersects << ", d_l: " << it.d_l << ", d_r: " << it.d_r << "\n";
    }

    std::cout << sss.str() << std::endl;

    return aux_sequence;
}

void insertMidFrameMarks(std::vector<MarkAndTime> &sequence)
{
    sequence.insert(sequence.begin(), {5, 0});
    int aux = 0;

    for (size_t i = 1; i < sequence.size() - 1; i++)
    {
        int currentFrame = sequence[i].frame;
        int nextFrame = sequence[i + 1].frame;
        int midFrame = currentFrame + (nextFrame - currentFrame) / 2;

        if (midFrame != currentFrame && midFrame != nextFrame)
        {
            sequence.insert(sequence.begin() + i + 1, {5, midFrame});
            i++;
            aux = (nextFrame - currentFrame) / 2;
        }
    }

    int lastFrame = sequence.back().frame;
    sequence.push_back({5, lastFrame + aux});
}

std::vector<item> compressMarks(const std::vector<item> &marks)
{
    std::vector<item> compressedMarks;
    if (marks.empty())
        return compressedMarks;

    item lastMark = marks[0];
    for (size_t i = 1; i < marks.size(); i++)
    {
        if (marks[i].code != lastMark.code)
        {
            compressedMarks.push_back(lastMark);
            lastMark = marks[i];
        }
        else
        {
            lastMark.frame = marks[i].frame;
        }
    }

    if (!compressedMarks.empty() && compressedMarks.back().frame != lastMark.frame)
    {
        compressedMarks.push_back(lastMark);
    }
    else if (compressedMarks.empty())
    {
        compressedMarks.push_back(lastMark);
    }

    return compressedMarks;
}

std::vector<Section> divideItemsIntoSequences(const std::vector<item>& items) {
    std::vector<Section> sequences;
    Section currentSection;

    // Función auxiliar para verificar si una sección contiene solo ítems con código 5
    auto sectionContainsOnlyFives = [](const Section& sec) {
        for (const auto& itm : sec.items) {
            if (itm.code != 5) {
                return false; // Si encuentra algo que no es un 5, devuelve falso
            }
        }
        return true; // Si todos son 5s, devuelve verdadero
    };

    if (!items.empty()) {
        currentSection.items.push_back(items[0]);
    }

    for (size_t i = 1; i < items.size(); ++i) {
        const item& current_item = items[i];
        const item& previous_item = items[i - 1];

        // Condición principal para revisar si necesitamos empezar una nueva sección
        if (current_item.code == 5 && (previous_item.code != 5 || (previous_item.code == 5 && previous_item.intersects == 0)) && current_item.intersects == 1) {
            // Añadir el ítem actual a la sección actual antes de verificar
            currentSection.items.push_back(current_item);
            // Verificar si la sección actual contiene solo 5s antes de finalizarla
            if (!sectionContainsOnlyFives(currentSection)) {
                sequences.push_back(currentSection);
            }
            currentSection = Section(); // Resetear la sección actual para empezar una nueva
        }

        // Agregar el ítem actual a la sección en construcción si no hemos empezado una nueva sección
        if (currentSection.items.empty() || currentSection.items.back().frame != current_item.frame) {
            currentSection.items.push_back(current_item);
        }
    }

    // Verificar y añadir la última sección si no está vacía y no contiene solo 5s
    if (!currentSection.items.empty() && !sectionContainsOnlyFives(currentSection)) {
        sequences.push_back(currentSection);
    }

    return sequences;
}

int calculateTakeoffFrame(std::vector<item> sequence) {
    for (size_t i = 1; i < sequence.size(); ++i) {
        const item& current_item = sequence[i];
        const item& previous_item = sequence[i - 1];

        if ( ((current_item.code != 5) || (current_item.code == 5 && current_item.intersects == 0)) && (previous_item.code == 5 && previous_item.intersects == 1)) {
            return current_item.frame;
        }
    }
}

std::pair<int, int> calculateArrivalFrame(std::vector<item> sequence) {
    int aux = 5;
    int outFrame = 0;
    for (size_t i = sequence.size() - 1; i > 0 ; i--) {
        const item& current_item = sequence[i - 1];

        if (current_item.intersects == 1 && (current_item.code != aux)){
            aux = current_item.code;
            outFrame = current_item.frame; 
            for (size_t k = sequence.size() - 1; k > 0; k--) {
                const item& current_itemsito = sequence[k - 1];

                if (current_itemsito.intersects == 1 && current_itemsito.code == aux ){
                    outFrame = current_itemsito.frame;
                }

                if ( (current_itemsito.code != aux) && (current_itemsito.intersects == 1) ){
                    return std::make_pair(outFrame, aux);
                }
            }
        }
    }
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
    return json;
}

std::string ComputerVisionWeb::buildFinalOutput(std::string jsonData, std::vector<MarkAndTime> sequence) {
    // Convertir jsonData a objeto JSON
    auto j = json::parse(jsonData);

    // Lista para almacenar los resultados
    std::vector<item> items;

    // Asumiendo que ambos, Left y Right, tienen la misma cantidad de frames
    int totalFrames = j["fields"]["Total frames"]["integerValue"];

    for (int i = 0; i < totalFrames; i++)
    {
        int code = j["fields"]["Left"]["mapValue"]["fields"]["code"]["arrayValue"]["values"][i]["integerValue"];
        int intersects = j["fields"]["Left"]["mapValue"]["fields"]["intersects"]["arrayValue"]["values"][i]["integerValue"];
        float d_l = j["fields"]["Left"]["mapValue"]["fields"]["d"]["arrayValue"]["values"][i]["doubleValue"];
        float d_r = j["fields"]["Right"]["mapValue"]["fields"]["d"]["arrayValue"]["values"][i]["doubleValue"];
        int step_l = j["fields"]["Left"]["mapValue"]["fields"]["step"]["arrayValue"]["values"][i]["boolValue"];
        int step_r = j["fields"]["Right"]["mapValue"]["fields"]["step"]["arrayValue"]["values"][i]["boolValue"];
        
        items.push_back({code, intersects, i, d_l, d_r, step_l, step_r});
    }

    // dividir items en secuencias y calcular frames de despegue y llegada
    auto sequences = divideItemsIntoSequences(items);
    for (auto& seq : sequences) {
        seq.takeoff_frame = calculateTakeoffFrame(seq.items);
        auto result = calculateArrivalFrame(seq.items);
        seq.arrival_frame = result.first;
        seq.arrival_code = result.second;
    }

    return toJSON(sequences);;
}

std::string ComputerVisionWeb::mainFunction(std::string contourjson, std::string videoUrl, std::string imageUrl, std::string jsonString) {
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

    int frame_rate = 0;
    if (vtest.isOpened())
    {
        frame_rate = vtest.get(cv::CAP_PROP_FPS);
        std::cout << frame_rate << std::endl;
    }
    else
    {
        std::cout << "El video no abrio!!" << std::endl;
        return "Error al abrir video";
    }

    // Antes de cerrar la función, imprime las variables modificadas

    // Imprimir contornos
    std::cout << "Contornos:\n";
    for (const auto& contorno : contornos) {
        std::cout << "Contorno - X: " << contorno.x << ", Y: " << contorno.y << ", Z: " << contorno.z << ", indiceContorno: " << contorno.indiceContorno << "\n";
        std::cout << "Puntos:";
        for (const auto& punto : contorno.points) {
            std::cout << " (" << punto.x << ", " << punto.y << ")";
        }
        std::cout << std::endl;
    }

    // Para sequence, suponiendo que parseSimpleJson y MarkAndTime están definidos correctamente
    std::cout << "Sequence:\n";
    for (const auto& markTime : sequence) {
        std::cout << "Mark: " << markTime.mark_correct << ", Time: " << markTime.frame << std::endl;
    }

    // Para videoUrl e imageUrl, si han cambiado
    std::cout << "URL del video procesado: " << videoUrl << std::endl;
    std::cout << "URL de la imagen de fondo procesada: " << imageUrl << std::endl;

    // Para frame_rate, que ya se imprime en el código original
    // Se imprime de nuevo por si necesitas un recordatorio
    std::cout << "Frame rate del video: " << frame_rate << std::endl;

    // Finalmente para calib_w y calib_h que se convierten de string a int
    std::cout << "calib_w: " << calib_w << std::endl;
    std::cout << "calib_h: " << calib_h << std::endl;


    // Feet Tracking
    /// Start feet tracking
    double learningRate = 0.005;
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog;
    initTracker(mog);
    cv::Mat current, result, result_big;

    // Train MoG
    bool first = true;
    uint frame = 0, maxFrame, time = 0, msec_per_frame = 1000 / frame_rate,
         initial_msec = 2001, // final_msec = 10000;
        // initial_msec = 0,
        final_msec = INT_MAX;
    cv::Mat fg;
#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "MoG Training.\n\tInit time: " << initial_msec << std::endl;
    std::cout << "\tFinal time: " << final_msec << " [msecs]" << std::endl;
#endif

#ifdef MEMORY_DEBUG
    std::cerr << "Start MoG Training..." << std::endl;
#endif

    // NEW: Insert background calibration image to reinforce background
    cv::Mat bg = cv::imread(urlBG);

#ifdef SHOW_INTERMEDIATE_RESULTS
    cv::imshow("Background", bg);
#endif

    // Pretrain with bg
    int num_bg = 50;
    cv::Mat fga;
    for (int i = 0; i < num_bg; ++i)
    {
        mog->apply(bg, fga, learningRate);
#ifdef SHOW_INTERMEDIATE_RESULTS
        cv::imshow("Pretrain FG", fga);
        cv::waitKey(5);
#endif
    }

    std::map<int, int> msecs;
    while (1)
    {
        vtest >> current;

        if (current.empty())
        {
#ifdef SHOW_INTERMEDIATE_RESULTS
            std::cout << "Video ended." << std::endl;
#endif
            break;
        }
        ++frame;

#ifdef MEMORY_DEBUG
        std::cerr << "\tMoG Training - Frame: " << frame << std::endl;
#endif

        msecs[frame] = time;
#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Frame number: " << frame << std::endl;
        std::cout << "Frame time: " << time << " [msec]" << std::endl;
#endif
        time += msec_per_frame;
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

        if (time > initial_msec && time < final_msec)
        {
            trainMog(mog, current, fg, learningRate);
#ifdef SHOW_INTERMEDIATE_RESULTS
            cv::imshow("Current Image.jpg", current);
            cv::imshow("Train FG.jpg", fg);
            cv::waitKey(5);
#endif
        }
    }
    vtest.release();

    maxFrame = frame;

#ifdef MEMORY_DEBUG
    std::cerr << "End MoG Training...\n\n Start calibration init..." << std::endl;
#endif

    // Calibrate scene
    std::vector<cv::Point2f> scenePoints;
    std::map<int, std::vector<cv::Point2i>> objectiveImPos;

    setScenePoints(scenePoints);

    std::map<int, int>::iterator frame_it = msecs.begin();
    cv::Mat painted_seg(current.size(), CV_8UC3), painted_seg2(current.size(), CV_8UC3), slabels, pseg2, pseg3, cur_copy;
    cv::Rect rin, rext, rext2;
    std::map<int, cv::Mat> inside_samples;
    std::map<int, cv::Rect> gt_bboxes;
    std::map<int, cv::Rect> alg_bboxes, alg_bboxes2, alg_bboxes3;

    extendedTrackedContours econtours;
    FeetTracker ft(maxFrame);
    int x1, y1, x2, y2;
    // IMPORTANT PARAMETER!! Determines the sensitivity for considering a step:
    float MIN_DISPLACEMENT_RATE = 0.005;
    FeetTracker::min_displacement = MIN_DISPLACEMENT_RATE * bg.rows;
    if (FeetTracker::min_displacement < 3)
        FeetTracker::min_displacement = 3;
    std::cout << "Min displacement of feet to not consider it as step: " << FeetTracker::min_displacement << std::endl;

    ft.total_frames = maxFrame;
    ft.frame_count = 0;

    vtest.open(urlVideo);
    if (!vtest.isOpened())
    {
        std::cout << "El video no abrio la segunda vez!!" << std::endl;
        return "El video no abrio la segunda vez!!";
    }

    // Set Inverse homography and scene points for optimizing next function
    std::string H_string = "[1.9752780221722848,-1.3315569413105433,614.94473698828381,"
                           "0.0014149777183884429,-0.094732686297120838,437.58441154428328,"
                           "-6.3759787960841727e-05,-0.0019745161334410294,1]";

    // Remover los corchetes al inicio y al final de la cadena
    H_string = H_string.substr(1, H_string.size() - 2);

    // Crear un stream de la cadena para extraer los valores flotantes
    std::istringstream ss(H_string);

    // Crear la matriz de OpenCV, inicializada con ceros
    cv::Mat H = cv::Mat::zeros(3, 3, CV_64FC1);

    // Variables temporales para almacenar los valores extraídos
    double value;
    char comma; // Para ignorar las comas en el stream

    // Un vector para almacenar todos los valores numéricos
    std::vector<double> values;

    // Extraer los valores de la cadena
    while (ss >> value)
    {
        values.push_back(value);
        ss >> comma; // Leer y descartar la coma
    }

    // Verificar si tenemos la cantidad correcta de valores para llenar la matriz H
    if (values.size() == 9)
    {
        // Asignar los valores a la matriz H
        int idx = 0;
        for (int i = 0; i < H.rows; ++i)
        {
            for (int j = 0; j < H.cols; ++j)
            {
                H.at<double>(i, j) = values[idx++];
            }
        }
    }
    else
    {
        std::cerr << "Error: Número incorrecto de elementos en la cadena H_string." << std::endl;
    }
    ft.Hinv = H.inv();
    ft.scenePoints = scenePoints;
    ft.contours = contornos;
    ft.real_w = real_w;
    ft.real_h = real_h;
    ft.calib_w = calib_w;
    ft.calib_h = calib_h;

    // Adjust tracking and get steps
    ft.left_foot.resize(maxFrame);
    ft.right_foot.resize(maxFrame);
    ft.left_step.resize(maxFrame, 0);
    ft.right_step.resize(maxFrame, 0);
    ft.Dx_left_s.resize(maxFrame);
    ft.Dy_left_s.resize(maxFrame);
    ft.Dx_right_s.resize(maxFrame);
    ft.Dy_right_s.resize(maxFrame);
    ft.left_rects_s.resize(maxFrame);
    ft.right_rects_s.resize(maxFrame);
    ft.in_objective.resize(maxFrame, -1);
    ft.in_objective1.resize(maxFrame, 0);
    ft.in_objective2.resize(maxFrame, 0);
    ft.odist1.resize(maxFrame, 0.0);
    ft.odist2.resize(maxFrame, 0.0);
    ft.left_intersects.resize(maxFrame, 0);
    ft.right_intersects.resize(maxFrame, 0);

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

    std::map<int, std::vector<cv::Point2i>> objectiveImPosAdj;
    int x, y;
    for (uint i = 1; i <= 9; ++i)
    {
        std::vector<cv::Point2i> &pos = objectiveImPos[i];
        std::vector<cv::Point2i> new_pos;
        for (uint j = 0; j < 4; ++j)
        {
            cv::Point2i p = pos[j];
            x = (p.x * real_w) / calib_w;
            y = (p.y * real_h) / calib_h;
            new_pos.push_back(cv::Point2i(x, y));
        }
        objectiveImPosAdj[i] = new_pos;
    }
    ft.objImPos = objectiveImPosAdj;
#ifdef MEMORY_DEBUG
    std::cerr << "End calibration init...\n\nStart step processing..." << std::endl;
#endif

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
        cv::Mat rr;
        result = presegmentation(mog, current, slabels, rr);
        if (!result.empty())
        {
            rin.x = rr.at<int>(0);
            rin.y = rr.at<int>(1);
            rin.width = rr.at<int>(2);
            rin.height = rr.at<int>(3);
            alg_bboxes[frame] = rin;

#ifdef SHOW_INTERMEDIATE_RESULTS
            cv::imshow("Biggest blob mask.jpg", result);
            cv::Mat result3C;
            cv::cvtColor(result, result3C, cv::COLOR_GRAY2BGR);
            cv::rectangle(result3C, rin, cv::Scalar(0, 0, 255), 1);
            cv::imshow("Crude segmentation using MoG model with bbox", result3C);
#endif
            // std::vector< std::vector<cv::Point> > contours;
            std::vector<cv::Point> big_contour;
            extendedContour ex_contour;
            cv::Mat result3 = ex_contour.extendContour(result, slabels, rr, rext2);

            alg_bboxes3[frame] = rext2;

            econtours.addContour(frame, ex_contour);

#ifdef SHOW_INTERMEDIATE_RESULTS
            cv::cvtColor(result3, painted_seg2, cv::COLOR_GRAY2BGR);
            cv::rectangle(painted_seg2, gt_bboxes[frame], cv::Scalar(0, 0, 255), 1);
            cv::rectangle(painted_seg2, rext2, cv::Scalar(0, 255, 0), 1);
            std::vector<std::vector<cv::Point>> cpaint;
            cpaint.push_back(ex_contour.cfinal);
            cv::drawContours(painted_seg2, cpaint, 0, cv::Scalar(0, 255, 255), 2);
            cv::imshow("Segmentation using MoG model.jpg", result);
            cv::imshow("Region Result Contour Extended.jpg", painted_seg2);
            std::cout << "Frame: " << frame << std::endl;
            cv::waitKey(0);

//            cv::resize(painted_seg2, pseg3, cv::Size(painted_seg2.cols*3,painted_seg2.rows*3));
//            cv::imshow("Region Result Contour Extended.jpg", pseg3);
#endif
            //            if(frame == 201)
            //                std::cout << "Stop Here!" << std::endl;
            // Adjust rect to include one more pixel around (for detecting skeleton end points):
            x1 = rext2.x - 1;
            y1 = rext2.y - 1;
            x2 = rext2.x + rext2.width;
            y2 = rext2.y + rext2.height;
            if (x1 < 0)
                x1 = 0;
            if (y1 < 0)
                y1 = 0;
            if (x2 >= current.cols)
                x2 = current.cols - 1;
            if (y2 >= current.rows)
                y2 = current.rows - 1;

            cv::Rect player_roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
            ft.player_roi.push_back(player_roi);
            // Set candidates and track them:
            ft.setFeetPositionsByBBox(frame, player_roi, result3);
            ft.trackPositions(frame, player_roi, result3, current, frame_it->second, i);
#ifdef SHOW_INTERMEDIATE_RESULTS
            std::cout << "Frame: " << frame << std::endl;
#endif
            ft.processAvailableStepsWithCoverageArea(i);
            // ft.processAvailableStepsWithDistanceToCenter(i);
        }

        frame_it++;

        //        cv::waitKey(0);
        //        if(cv::waitKey(5) != -1)
        //           break;
    }

    vtest.release();

#ifdef MEMORY_DEBUG
    std::cerr << "End step processing...\n\nStart step completion..." << std::endl;
#endif

#ifdef SHOW_INTERMEDIATE_RESULTS
    std::cout << "Last processed frame: " << frame << std::endl;
    std::cout << "Max frame: " << frame << std::endl;
#endif
    ft.completeTracking(frame - FeetTracker::frames_to_store / 2 + 1);

    // Complete foot positioning
    int pos_correction = FeetTracker::frames_to_store / 2 + 1;
    for (uint i = maxFrame - pos_correction; i <= maxFrame; ++i)
    {
        ft.processStepsWithCoverageArea(i - 1, i, ft.sframes[i]);
    }

    std::string jsonData = buildJsonData(ft);
    std::string out = buildFinalOutput(jsonData, sequence);
    std::cout << out << std::endl;

#ifdef SHOW_FINAL_RESULTS
    std::cout << out << std::endl;
#endif

#ifdef MEMORY_DEBUG
    std::cerr << "End step completion..." << std::endl;
#endif

    return out;
}