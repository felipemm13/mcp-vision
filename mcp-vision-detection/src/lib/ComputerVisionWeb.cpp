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
    //std::cout << "Biggest index is " << bindex << " with " << max << " pixels." << std::endl;

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

std::vector<MarkAndTime> parseSimpleJson(const std::string &jsonString)
{
    std::vector<MarkAndTime> marks;
    std::istringstream stream(jsonString);
    std::string line;
    
    while (std::getline(stream, line))
    {
        if (line.find("mark_correct") != std::string::npos)
        {
        MarkAndTime mark;
            mark.mark_correct = std::stoi(line.substr(line.find(":") + 1));
            std::getline(stream, line);
            mark.frame = std::stoi(line.substr(line.find(":") + 1));
        marks.push_back(mark);
        }
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

std::vector<item> buildArrivals(std::vector<MarkAndTime> real_sequence, std::vector<item> player_sequence)
{
    std::vector<item> arrivals;
    bool isSequenceStart = true; // Suponemos que el inicio de la lista puede ser el inicio de una secuencia

    for (size_t i = 0; i < player_sequence.size(); ++i)
    {
        // Comprobar si estamos al inicio de una secuencia de intersects == 1
        if (player_sequence[i].intersects == 1 && (isSequenceStart || player_sequence[i - 1].intersects == 0))
        {
            arrivals.push_back(player_sequence[i]); // Agregar el item al vector de llegadas
            isSequenceStart = false;                // Actualizar el indicador de inicio de secuencia
        }
        else if (player_sequence[i].intersects == 0)
        {
            isSequenceStart = true; // Si encontramos un intersects == 0, el próximo item con intersects == 1 será el inicio de una nueva secuencia
        }
    }
    return arrivals;
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
    return 0;
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
    return std::make_pair(0, 0);

}

void calculateArrivalFrameWithError(Section &sequence) {
    float aux = FLT_MAX;
    int outFrame = 0;
    float minDistance = FLT_MAX; 
    
    for (size_t i = sequence.items.size() - 1; i > 0 ; i--) {
        const item& current_item = sequence.items[i - 1];

        if (current_item.code != 5){
            aux = current_item.code;

            for (size_t j = sequence.items.size() - 1; j > 0 ; j--) {
                const item& current_itemsito = sequence.items[j - 1];
                if (current_itemsito.code != aux){
                    return ;
                }else{
                    float currentMinDistance = std::min(current_itemsito.d_l, current_itemsito.d_r);
                    if (currentMinDistance < minDistance) {
                        minDistance = currentMinDistance;
                        sequence.arrival_frame = current_itemsito.frame;
                    }
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

void calculateError(std::vector<Section> &user_sequence, std::vector<MarkAndTime> &real_sequences){
    for (int i = 0 ; i < user_sequence.size() ; i++) {
        // Verificamos si estamos en la última iteración
        if (i == user_sequence.size() - 1) {
            // Para la última marca, solo necesitas verificar si el usuario fue a la marca correcta.
            if (user_sequence[i].arrival_code == real_sequences[i].mark_correct) {
                user_sequence[i].error = false;
            } else {
                user_sequence[i].error = true;
            }
        } else {
            /* Verificamos si el usuario fue a la marca correcta && Si el usuario llegó a la marca en el tiempo correcto*/
            if (user_sequence[i].arrival_code == real_sequences[i].mark_correct && (user_sequence[i].arrival_frame >= real_sequences[i].frame && user_sequence[i].arrival_frame < real_sequences[i+1].frame) ){
                user_sequence[i].error = false;
            }else{
                user_sequence[i].error = true;
            }
        }
    }
}

std::string ComputerVisionWeb::buildFinalOutput(std::string jsonData, std::vector<MarkAndTime> sequence)
{
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

    calculateError(sequences ,sequence);

    for (auto& seq : sequences) {
        if (seq.error == 1){
            calculateArrivalFrameWithError(seq);
        }
    }

    return toJSON(sequences);;
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
    
    for (int i = 0; i < maxFrame; ++i) {
        std::cout << "Frame Index: " << i << "\n\tLeft: " << ft.in_objective1[i] << "\n\tRight: " << ft.in_objective2[i] << std::endl;
        
    }
    
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


// Per frame: Two feet. By foot: (x y w h code xp yp d)
//  (x,y,w,h): foot rect                (left_step, right_step)
//  code:                               (in_objective1, in_objective2)
//      0: No step
//    1-9: Step to nearest objective
//  (xp,yp): Feet contact point         (left_foot, right_foot)
//  d: distance to nearest center       (odist1, odist2)
std::string ComputerVisionWeb::buildFinalOutputImproved(FeetTracker &ft, std::vector<MarkAndTime> sequence, int maxFrame) {
    //in_objective1, in_objective2
    //odist1, odist2
    
    for (int i = 0; i <= maxFrame; ++i) {
        std::cout << "Frame: " << i << "\n\tLeft: " << ft.in_objective1[i] << "\n\tRight: " << ft.in_objective2[i] << std::endl;
        
    }
    
    //Get central stimulus central position
    cv::Point2f pcentral = ft.contourCentersScene[4];
    
    
    const int relevant_change = 40; //Number of centimeters for considering relevant change in position
    
    int current_seq = 0, //index for starting current sequence
        current_center_exit = 0, //index for exiting center on current sequence
        next_seq = 0;  //index for starting next sequence
    cv::Point p_out;
    
    //Variables for stimuli sequence:
    uint n_objectives = sequence.size();
    int i, cur_objective, cur_frame;

    //Divide items in stimuli sequence data and calculate frames de despegue y llegada
    // Lista para almacenar los resultados
    std::vector<Section> sequences;
    
    //Get intervals per objective:
    for (int j = 0; j < n_objectives; ++j) {
        Section cur_section;
        item cur_item;
        
        cur_objective = sequence[j].mark_correct; 
        cur_frame = sequence[j].frame;

#ifdef SHOW_DEBUG_TEXT
        std::cout << "Marking.\n\tCurrent stimuli: " << j << std::endl;
        std::cout << "\tCurrent stimuli objective: " << cur_objective << std::endl;
        std::cout << "\tCurrent stimuli frame: " << cur_frame << std::endl;
#endif        
        //Booleans for marking errors (assume right first):
        bool step_center = true, step_objective = true, right_objective = true; 
        
        //Advance until both are near the 5 zone (assume that the player can be late):
        for (i = current_seq; i <= maxFrame; ++i)
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
        bool step_out_detected = false, out_left;
        for (i = current_seq; i <= maxFrame; ++i) {
            if(ft.left_step[i] == 0 && ft.right_step[i] == 0) //Continue until a step is detected
                continue;
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tCurrent frame - search position exit: " << i << std::endl;
#endif        
            //Left foot steps out:
            if(!step_out_detected && ft.left_step[i] == 1 && ft.in_objective1[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tExit frame left - position exit: " << i << std::endl;
        std::cout << "\tExit frame left - code found: " << ft.in_objective1[i] << std::endl;
#endif        
                current_center_exit = i;
                step_out_detected = true;
            }
            //Right foot steps out:
            if(!step_out_detected && ft.right_step[i] == 1 && ft.in_objective2[i] != 5) {
#ifdef SHOW_DEBUG_TEXT
        std::cout << "\tExit frame right - position exit: " << i << std::endl;
        std::cout << "\tExit frame right - code found: " << ft.in_objective2[i] << std::endl;
#endif        
                current_center_exit = i;
                step_out_detected = true;
            }
            
            //Get a coherent position to compare by ensuring that both feet are near the same zone and take one of them as step:
            if(ft.in_objective1[i] != 5 && ft.in_objective2[i] == ft.in_objective1[i]) {
            //Left foot steps out:
                if(ft.left_step[i] == 1) {
#ifdef SHOW_DEBUG_TEXT
                    std::cout << "\tExit frame left - sure position at: " << i << std::endl;
                    std::cout << "\tExit frame left - sure position code found: " << ft.in_objective1[i] << std::endl;
                    std::cout << "\tExit frame left - sure position: " << ft.left_foot[i].x << ", " << ft.left_foot[i].y << std::endl;
#endif        
                    p_out = ft.left_foot[i];
                    out_left = true;
                    break;
                }
                //Right foot steps out:
                if(ft.right_step[i] == 1) {
#ifdef SHOW_DEBUG_TEXT
                    std::cout << "\tExit frame right - sure position: " << i << std::endl;
                    std::cout << "\tExit frame right - sure position code found: " << ft.in_objective2[i] << std::endl;
                    std::cout << "\tExit frame right - sure position: " << ft.right_foot[i].x << ", " << ft.right_foot[i].y << std::endl;
#endif        
                    p_out = ft.right_foot[i];
                    out_left = false;
                    break;
                }
            }
        }
        
        if(step_out_detected) { //It shall be detected... if not maybe end of stimuli sequence or player skip some stimuli
            //Check which is the first leg going on the objective direction
            int lindex_1, lindex_2, rindex_1, rindex_2;
            bool stepping = false, first = true, pl_found = false, pr_found = false;
            cv::Point2f p1, p2;
            float d1, d2, dc1, dc2, d, d_max = 0;
            int max_index = current_seq + 1;
            bool max_is_left = true;
            
            //Check first relevant change in left step
            for (i = current_seq; i <= current_center_exit; ++i) {
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off - left - frame: " << i << std::endl;
#endif                        
                if(first) { //Search for end of first stepping
                    if(ft.left_step[i] == 1) { //A step
#ifdef SHOW_DEBUG_TEXT
                std::cout << "\tTake-off - left - start stepping... " << std::endl;
#endif                        
                        stepping = true;
                    } else if(stepping && ft.left_step[i] == 0) { //Is stepping, so check if it stops doing so
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - left - stop stepping... " << std::endl;
#endif                        
                        lindex_1 = i-1;
                        stepping = false;
                        first = false; //First index ready
                    }
                } else { //Search for first of following step
                    if(ft.left_step[i] == 1) { //First position of next step
                        lindex_2 = i;
                        cv::Point2f p_out_s;
                        //Significant displacement criterion
                        p1 = ft.imageToScene(ft.left_foot[lindex_1]);
                        p2 = ft.imageToScene(ft.left_foot[lindex_2]);
                        p_out_s = ft.imageToScene(p_out);
                        d1 = sqrt((p_out_s.x - p1.x)*(p_out_s.x - p1.x) + (p_out_s.y - p1.y)*(p_out_s.y - p1.y));//L2 norm
                        d2 = sqrt((p_out_s.x - p2.x)*(p_out_s.x - p2.x) + (p_out_s.y - p2.y)*(p_out_s.y - p2.y));//L2 norm
                        dc1 = sqrt((pcentral.x - p1.x)*(pcentral.x - p1.x) + (pcentral.y - p1.y)*(pcentral.y - p1.y));//L2 norm
                        dc2 = sqrt((pcentral.x - p2.x)*(pcentral.x - p2.x) + (pcentral.y - p2.y)*(pcentral.y - p2.y));//L2 norm
                        d  = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
                        
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - Two left found - first index: " << lindex_1 << std::endl;
                        std::cout << "\tTake-off - Two left found - second index: " << lindex_2 << std::endl;
                        std::cout << "\tTake-off - Two left found - d1: " << d1 << std::endl;
                        std::cout << "\tTake-off - Two left found - d2: " << d2 << std::endl;
                        std::cout << "\tTake-off - Two left found - d_relevant: " << d << std::endl;
#endif        

                        
                        if(d >= relevant_change && d2 < d1 && dc1 < dc2) { //It approaches to center exit and goes far from center
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - Two left found - left found. " << std::endl;
#endif
                            //lindex_1+1 will be the take-off frame if left index < right index
                            pl_found = true;
                            break;
                        } else { //Non-significant step or step in wrong direction, keep searching
                            if(d2 < d1 && dc1 < dc2) { //Store it in case no one accomplish the strong criterion
                                if(d > d_max) { 
                                    d_max = d;
                                    max_index = lindex_1;
                                    max_is_left = true;
                                }
                            }
                            
                            stepping = true;                            
                            first = true; //Consider as first again
                            lindex_1 = lindex_2;
                        }
                    }                        
                }
            }

            
            //Now check first relevant change in right step
            stepping = false; 
            first = true;
            for (i = current_seq; i <= current_center_exit; ++i) {
                if(first) { //Search for end of first stepping
                    if(ft.right_step[i] == 1) { //A step
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - right - start stepping... " << std::endl;
#endif                        
                        stepping = true;
                    } else if(stepping && ft.right_step[i] == 0) { //Is stepping, so check if it stops doing so
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - right - stop stepping... " << std::endl;
#endif                        
                        rindex_1 = i-1;
                        stepping = false;
                        first = false; //First index ready
                    }
                } else { //Search for first of following step
                    if(ft.right_step[i] == 1) { //First position of next step
                        rindex_2 = i;
                        cv::Point2f p_out_s;
                        //Significant displacement criterion
                        p1 = ft.imageToScene(ft.right_foot[rindex_1]);
                        p2 = ft.imageToScene(ft.right_foot[rindex_2]);
                        p_out_s = ft.imageToScene(p_out);
                        d1 = sqrt((p_out_s.x - p1.x)*(p_out_s.x - p1.x) + (p_out_s.y - p1.y)*(p_out_s.y - p1.y));//L2 norm
                        d2 = sqrt((p_out_s.x - p2.x)*(p_out_s.x - p2.x) + (p_out_s.y - p2.y)*(p_out_s.y - p2.y));//L2 norm
                        dc1 = sqrt((pcentral.x - p1.x)*(pcentral.x - p1.x) + (pcentral.y - p1.y)*(pcentral.y - p1.y));//L2 norm
                        dc2 = sqrt((pcentral.x - p2.x)*(pcentral.x - p2.x) + (pcentral.y - p2.y)*(pcentral.y - p2.y));//L2 norm
                        d  = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
#ifdef SHOW_DEBUG_TEXT
                        std::cout << "\tTake-off - Two right found - first index: " << rindex_1 << std::endl;
                        std::cout << "\tTake-off - Two right found - second index: " << rindex_2 << std::endl;
                        std::cout << "\tTake-off - Two right found - d1: " << d1 << std::endl;
                        std::cout << "\tTake-off - Two right found - d2: " << d2 << std::endl;
                        std::cout << "\tTake-off - Two right found - d_relevant: " << d << std::endl;
#endif        

                        if(d >= relevant_change && d2 < d1 && dc1 < dc2) { //It approaches to center exit 
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off - Two right found - right found. " << std::endl;
#endif
                            //rindex_1 will be the take-off frame if left index > right index
                            pr_found = true;
                            break;
                        } else { //Non-significant step or step in wrong direction, keep searching
                            if(d2 < d1 && dc1 < dc2) { //Store it in case no one accomplish the strong criterion
                                if(d > d_max) { 
                                    d_max = d;
                                    max_index = rindex_1;
                                    max_is_left = false;
                                }
                            }
                            stepping = true;                            
                            first = true; //Consider as first again
                            rindex_1 = rindex_2;
                        }
                    }                        
                }
            }   

            //Set take-off frame:
            if(pl_found && pr_found) {
                if(lindex_1 < rindex_1) { //Take-off is from left
                    cur_item.code = 5;
                    cur_item.d_l = ft.odist1[lindex_1];
                    cur_item.d_r = ft.odist2[rindex_1];
                    cur_item.step_l = true;
                    cur_item.step_r = false;
                    cur_item.frame = lindex_1+1;
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = lindex_1+1;
                } else {
                    cur_item.code = 5;
                    cur_item.d_l = ft.odist1[lindex_1];
                    cur_item.d_r = ft.odist2[rindex_1];
                    cur_item.step_l = false;
                    cur_item.step_r = true;
                    cur_item.frame = rindex_1+1;                    
                    cur_section.items.push_back(cur_item);
                    cur_section.takeoff_frame = rindex_1+1;
                }
            } else if(pl_found) {
                cur_item.code = 5;
                cur_item.d_l = ft.odist1[lindex_1];
                cur_item.d_r = ft.odist2[rindex_1];
                cur_item.step_l = true;
                cur_item.step_r = false;
                cur_item.frame = lindex_1+1;
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = lindex_1+1;

            } else if (pr_found) {
                cur_item.code = 5;
                cur_item.d_l = ft.odist1[lindex_1];
                cur_item.d_r = ft.odist2[rindex_1];
                cur_item.step_l = false;
                cur_item.step_r = true;
                cur_item.frame = rindex_1+1;                                    
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = rindex_1+1;
            } else { //if none, use max
                cur_item.code = 5;
                cur_item.d_l = ft.odist1[max_index];
                cur_item.d_r = ft.odist2[max_index];
                cur_item.step_l = max_is_left;
                cur_item.step_r = !max_is_left;
                cur_item.frame = max_index+1;                                    
                cur_section.items.push_back(cur_item);
                cur_section.takeoff_frame = max_index+1;
            }
#ifdef SHOW_DEBUG_TEXT
                            std::cout << "\tTake-off: " << cur_item.frame << std::endl;
                            std::cout << "\tTake-off - left?: " << cur_item.step_l << std::endl;
#endif
        //end if step_out_detected
        } else { 
            break; //No step out, means irrelevant rest of video
        }
        
        //Search for arrival info
        int last_arrival_frame = -1; //Value is -1 if no one steps, and the corresponding frame if it steps
        int nearest_arrival_code; //Stepping or not, it is the nearest arrival code
        int last_real = -1;
        float nearest_arrival_distance = FLT_MAX;
        bool still_near_center_l = true, still_near_center_r = true, real_arrival = false, is_left = true;
        for (int i = current_center_exit; i <= maxFrame; ++i) {
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
        for (int i = next_seq; i <= maxFrame; ++i) {
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
        
    } //end stimuli sequence
    
    return toJSON(sequences);
}

cv::Mat recalibrateHomography(std::vector<cv::Point2f> &contourCenters) {
    std::vector<cv::Point2f> scenePoints;
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


        

std::string ComputerVisionWeb::mainFunction(std::string contourjson, std::string videoUrl, std::string imageUrl, std::string jsonString, std::string frameRate) {
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
        initial_msec = 0, // final_msec = 10000;
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

    learningRate = 0.01;
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
    float MIN_DISPLACEMENT_RATE = 0.008;
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

    ft.scenePoints = scenePoints;
    ft.contours = contornos;
    ft.real_w = real_w;
    ft.real_h = real_h;
    ft.calib_w = calib_w;
    ft.calib_h = calib_h;

    
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
    ft.contourCenters = contourCenters;

    //Recalculate homography image --> scene with contour centers
    ft.H = recalibrateHomography(contourCenters);

    //Get proyected contours:
    for (const auto &c : contornos) {
        Contour contorno;
        contorno.indiceContorno = c.indiceContorno;
        contorno.x = c.x;
        contorno.y = c.y;
        contorno.z = c.z;
        for (const auto &p : c.points) 
            contorno.points.push_back(ft.imageToScene(p));
        ft.contoursScene.push_back(contorno); 
    }
    
    
    std::cout << "Recalibration: " << std::endl;
    for(int i=0; i<contourCenters.size(); ++i) {
        cv::Point2f p = ft.imageToScene(contourCenters[i]);
        std::cout << "\tObjective " << i+1 << ": " << p.x << ", " << p.y << std::endl; 
        ft.contourCentersScene.push_back(p);
    }

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
//            std::cout << "Frame: " << frame << "; OFrame: " << oframe << std::endl;
            if(frame < oframe) 
                break;
            int cobjective = m.mark_correct;
            cur_objective = cobjective;
            j_cur = j;
        
        }
        
#ifdef SHOW_INTERMEDIATE_RESULTS
        std::cout << "Current Objective: " << cur_objective << std::endl;
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
            ft.processAvailableStepsWithCoverageArea(i, cur_objective);
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
        ft.processStepsWithCoverageArea(i - 1, i, ft.sframes[i], cur_objective);
        // ft.processStepsWithDistanceToCenter(i - 1, i, ft.sframes[i]);
    }

    std::string jsonData = buildJsonData(ft);
    std::string out = buildFinalOutputFinal(ft, sequence, maxFrame);

#ifdef SHOW_FINAL_RESULTS
    std::cout << "============ OUT ============ \n" << out << std::endl;
#endif

#ifdef MEMORY_DEBUG
    std::cerr << "End step completion..." << std::endl;
#endif

    return out;
}