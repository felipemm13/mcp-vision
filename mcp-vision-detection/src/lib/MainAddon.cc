#include "MainAddon.h"

void MainAddon::Init(Local<Object> target) {
  Nan::HandleScope scope;
  Nan::SetMethod(target, "createImage", CreateImage);
  Nan::SetMethod(target, "setCalibrationAutomatic", SetCalibrationAutomatic);
  Nan::SetMethod(target, "setCalibrationSemiAutomatic", SetCalibrationSemiAutomatic);
}

NAN_METHOD(MainAddon::CreateImage) {
    cv::Mat output = cv::Mat::zeros( 120, 350, CV_8UC3 );
    putText(
      output,
      "Hello World :)",
      cv::Point(15,70),
      cv::FONT_HERSHEY_PLAIN,
      3,
      cv::Scalar(0,255,0),
      4
    );
    cv::imwrite("hello-world.png", output);

}



NAN_METHOD(MainAddon::SetCalibrationAutomatic){
  v8::Isolate* isolate = info.GetIsolate();
  
  v8::String::Utf8Value v8Email(isolate, info[0]);
  std::string email(*v8Email);
  
  v8::String::Utf8Value v8Screenshot(isolate, info[1]);
  std::string screenshot(*v8Screenshot);
  
  Calibration_fixed mainCalibration = Calibration_fixed();
  auto response = mainCalibration.getMarksAutomatic(email,screenshot);
  // Crear un objeto JSON para contener los valores de respuesta
  v8::Local<v8::Object> responseObject = v8::Object::New(isolate);
  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("status").ToLocalChecked(),
                      Nan::New(std::get<0>(response))); // Estado de la respuesta

  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("message").ToLocalChecked(),
                      Nan::New(std::get<1>(response).c_str()).ToLocalChecked()); // Mensaje de la respuesta

  // Convertir el tercer elemento de la tuple, que es un vector de PointWithContour
  std::vector<PointWithContour> pointsWithContour = std::get<2>(response);
  v8::Local<v8::Array> pointsArray = Nan::New<v8::Array>(pointsWithContour.size());

  for (size_t i = 0; i < pointsWithContour.size(); ++i) {
      v8::Local<v8::Object> pointObject = v8::Object::New(isolate);
      pointObject->Set(isolate->GetCurrentContext(),
                       Nan::New("x").ToLocalChecked(),
                       Nan::New(pointsWithContour[i].punto.x));
      pointObject->Set(isolate->GetCurrentContext(),
                       Nan::New("y").ToLocalChecked(),
                       Nan::New(pointsWithContour[i].punto.y));
      pointObject->Set(isolate->GetCurrentContext(),
                       Nan::New("z").ToLocalChecked(),
                       Nan::New(pointsWithContour[i].punto.z));
      pointObject->Set(isolate->GetCurrentContext(),
                       Nan::New("indiceContorno").ToLocalChecked(),
                       Nan::New(static_cast<uint32_t>(pointsWithContour[i].indiceContorno)));
      
      // Convertir el vector cv::Point a un array JavaScript
      v8::Local<v8::Array> contourArray = Nan::New<v8::Array>(pointsWithContour[i].contorno.size());
      for (size_t j = 0; j < pointsWithContour[i].contorno.size(); ++j) {
          v8::Local<v8::Object> contourPoint = v8::Object::New(isolate);
          contourPoint->Set(isolate->GetCurrentContext(),
                            Nan::New("x").ToLocalChecked(),
                            Nan::New(pointsWithContour[i].contorno[j].x));
          contourPoint->Set(isolate->GetCurrentContext(),
                            Nan::New("y").ToLocalChecked(),
                            Nan::New(pointsWithContour[i].contorno[j].y));
          contourArray->Set(isolate->GetCurrentContext(), j, contourPoint);
      }

      pointObject->Set(isolate->GetCurrentContext(),
                       Nan::New("contorno").ToLocalChecked(),
                       contourArray);
      
      pointsArray->Set(isolate->GetCurrentContext(), i, pointObject);
  }

  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("points").ToLocalChecked(),
                      pointsArray);
                      
  // Convierte el objeto de respuesta en una cadena JSON
  v8::Local<v8::String> jsonResponse = JSON::Stringify(isolate->GetCurrentContext(), responseObject).ToLocalChecked();
  info.GetReturnValue().Set(jsonResponse);
}

NAN_METHOD(MainAddon::SetCalibrationSemiAutomatic){
  v8::Isolate* isolate = info.GetIsolate();
  
  v8::String::Utf8Value v8Email(isolate, info[0]);
  std::string email(*v8Email);
  
  v8::String::Utf8Value v8Screenshot(isolate, info[1]);
  std::string screenshot(*v8Screenshot);
  
  v8::String::Utf8Value v8Mark1_x(isolate, info[2]);
  std::string mark1_x(*v8Mark1_x);

  v8::String::Utf8Value v8Mark1_y(isolate, info[3]);
  std::string mark1_y(*v8Mark1_y);

    
  v8::String::Utf8Value v8Mark2_x(isolate, info[4]);
  std::string mark2_x(*v8Mark2_x);

  v8::String::Utf8Value v8Mark2_y(isolate, info[5]);
  std::string mark2_y(*v8Mark2_y);

    
  v8::String::Utf8Value v8Mark3_x(isolate, info[6]);
  std::string mark3_x(*v8Mark3_x);

  v8::String::Utf8Value v8Mark3_y(isolate, info[7]);
  std::string mark3_y(*v8Mark3_y);

    
  v8::String::Utf8Value v8Mark4_x(isolate, info[8]);
  std::string mark4_x(*v8Mark4_x);

  v8::String::Utf8Value v8Mark4_y(isolate, info[9]);
  std::string mark4_y(*v8Mark4_y);

    
  v8::String::Utf8Value v8Mark5_x(isolate, info[10]);
  std::string mark5_x(*v8Mark5_x);

  v8::String::Utf8Value v8Mark5_y(isolate, info[11]);
  std::string mark5_y(*v8Mark5_y);

    
  v8::String::Utf8Value v8Mark6_x(isolate, info[12]);
  std::string mark6_x(*v8Mark6_x);

  v8::String::Utf8Value v8Mark6_y(isolate, info[13]);
  std::string mark6_y(*v8Mark6_y);

  Calibration_3 mainCalibration = Calibration_3();
  auto response = mainCalibration.getMarksSemiAutomatic(email,screenshot,
                                                        mark1_x,mark1_y,
                                                        mark2_x,mark2_y,
                                                        mark3_x,mark3_y,
                                                        mark4_x,mark4_y,
                                                        mark5_x,mark5_y,
                                                        mark6_x,mark6_y);

  info.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, response.second.c_str()).ToLocalChecked());

}
