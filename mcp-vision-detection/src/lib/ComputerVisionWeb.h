#ifndef MAIN_H
#define MAIN_H

#define SHOW_INTERMEDIATE_RESULTS
#define SHOW_FINAL_RESULTS
#define SHOW_DEBUG_TEXT
#define MEMORY_DEBUG

#include "CommonDefinitions.h"
#include "ExtendedContour.h"
#include "ConvertImage.h"

using namespace std;

class ComputerVisionWeb
{
public:
    ComputerVisionWeb();

    // ============================
    // Funciones principales
    // ============================
    // (Versión nueva de mainFunction)
    string mainFunction(string contourjson, string videoUrl, string imageUrl, string jsonString, string frameRate);

    // (Versión nueva de buildOutput)
    string buildOutput(vector<MarkAndTime> sequence, int maxFrame);

    // (Mantiene recalibrateHomography, transformInv y imageToScene del código base,
    //  pero se respeta si el nuevo cambia algo interno en la implementación .cpp)
    cv::Mat recalibrateHomography();
    cv::Point2f transformInv(cv::Point2f p);
    cv::Point2f imageToScene(cv::Point2i p);

    // (Del código base: no hay conflicto, así que se conserva si todavía se requiere)
    cv::Point2f getStepPosition(int frame, cv::Rect &feet);

    // ============================
    // Funciones "nuevas" de proceso
    // (El nuevo recibe 3 parámetros en el 1er caso, 3 también en el 2do, etc.)
    // ============================
    void processStepsWithCoverageArea(int index, int cur_objective, cv::Mat cur_copy);
    void processAvailableStepsWithCoverageArea(int index, int cur_objective, cv::Mat cur_copy);

    // ============================
    // Intersecciones y cálculos
    // ============================
    bool feetIntersectsObjective(vector<cv::Point2i> &footPolygon, vector<cv::Point2i> &contour);
    int intersectsObjective(cv::Mat img,
                            int index,
                            vector<cv::Point2i> &leftStep,
                            bool leftStepOccurred,
                            vector<cv::Point2i> &rightStep,
                            bool rightStepOccurred);
    float calculatePointToLineDistance(const cv::Point2f &pointA,
                                       const cv::Point2f &pointB,
                                       const cv::Point2f &point);
    float distance(cv::Point2f &p1, cv::Point2f &p2);

    // ============================
    // Llamada a la API
    // ============================
    bool callApi(const string& videoUrl);

    // ============================
    // (Opcional) Función para dibujar
    // ============================
    void drawObjectives(cv::Mat &img, int id, int cur_objective);

    // ============================
    // Variables miembro
    // ============================
    vector<FrameInfo> frames_info;
    vector<Contour> contornos;
    vector<Contour> contours;
    vector<Contour> contoursScene;
    vector<cv::Point2f> contourCenters;
    vector<cv::Point2f> contourCentersScene;

    const int frames_to_store = 7;
    const float cm_to_feet_contact = 10;
    const float max_cm_to_center = 15;

    vector<cv::Rect> left_rects_s;
    vector<cv::Rect> right_rects_s;

    vector<cv::Point2f> left_foot;  // Pixel position
    vector<cv::Point2f> right_foot; // Pixel position

    vector<int> in_objective1;
    vector<int> in_objective2;
    vector<float> odist1;
    vector<float> odist2;
    vector<int> left_intersects;
    vector<int> right_intersects;

    // Polígonos de pies
    vector<vector<cv::Point2i>> right_feet;
    vector<vector<cv::Point2i>> left_feet;

    // Step flags
    vector<bool> left_step;
    vector<bool> right_step;

    cv::Mat H;
    int real_w, real_h;
    int calib_w;
    int calib_h;
};

#endif // MAIN_H
