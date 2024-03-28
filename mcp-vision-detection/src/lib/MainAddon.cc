#include "MainAddon.h"

void MainAddon::Init(Local<Object> target) {
  Nan::HandleScope scope;
  Nan::SetMethod(target, "createImage", CreateImage);
  Nan::SetMethod(target, "setCalibrationAutomatic", SetCalibrationAutomatic);
  Nan::SetMethod(target, "setCalibrationSemiAutomatic", SetCalibrationSemiAutomatic);
  Nan::SetMethod(target, "autoAnalysis", AutoAnalysis);
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
  
  v8::String::Utf8Value v8Screenshot(isolate, info[0]);
  std::string screenshot(*v8Screenshot);
  
  Calibration_fixed mainCalibration = Calibration_fixed();
  auto response = mainCalibration.getMarksAutomatic(screenshot);
  // Crear un objeto JSON para contener los valores de respuesta
  v8::Local<v8::Object> responseObject = v8::Object::New(isolate);
  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("status").ToLocalChecked(),
                      Nan::New(std::get<0>(response))); // Estado de la respuesta

  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("message").ToLocalChecked(),
                      Nan::New(std::get<1>(response).c_str()).ToLocalChecked()); // Mensaje de la respuesta


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
                      

  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("H_string").ToLocalChecked(),
                      Nan::New(std::get<3>(response).c_str()).ToLocalChecked());

  responseObject->Set(isolate->GetCurrentContext(),
                    Nan::New("calib_w").ToLocalChecked(),
                    Nan::New(std::get<4>(response)));

  responseObject->Set(isolate->GetCurrentContext(),
                    Nan::New("calib_h").ToLocalChecked(),
                    Nan::New(std::get<5>(response)));
  
  // Convierte el objeto de respuesta en una cadena JSON
  v8::Local<v8::String> jsonResponse = JSON::Stringify(isolate->GetCurrentContext(), responseObject).ToLocalChecked();
  info.GetReturnValue().Set(jsonResponse);
}

NAN_METHOD(MainAddon::SetCalibrationSemiAutomatic) {
  v8::Isolate* isolate = info.GetIsolate();
  v8::String::Utf8Value v8Screenshot(isolate, info[0]);
  std::string screenshot(*v8Screenshot);

  v8::Local<v8::Array> marksArray = v8::Local<v8::Array>::Cast(info[1]);
  std::vector<std::pair<double, double>> marks;

  for (uint32_t i = 0; i < marksArray->Length(); ++i) {
      v8::Local<v8::Value> markValue;
      if (!marksArray->Get(isolate->GetCurrentContext(), i).ToLocal(&markValue)) {
          isolate->ThrowException(v8::Exception::TypeError(v8::String::NewFromUtf8(isolate, "No se pudo acceder al elemento del arreglo").ToLocalChecked()));
          return;
      }

      if (!markValue->IsObject()) {
          isolate->ThrowException(v8::Exception::TypeError(v8::String::NewFromUtf8(isolate, "El elemento no es un objeto").ToLocalChecked()));
          return;
      }

      v8::Local<v8::Object> markObject = markValue.As<v8::Object>();

      v8::Local<v8::Value> xValue = markObject->Get(isolate->GetCurrentContext(), v8::String::NewFromUtf8(isolate, "x").ToLocalChecked()).ToLocalChecked();
      v8::Local<v8::Value> yValue = markObject->Get(isolate->GetCurrentContext(), v8::String::NewFromUtf8(isolate, "y").ToLocalChecked()).ToLocalChecked();

      double x = xValue->NumberValue(isolate->GetCurrentContext()).FromJust();
      double y = yValue->NumberValue(isolate->GetCurrentContext()).FromJust();

      marks.push_back(std::make_pair(x, y));
  }
  // ============================================================================================= /
  Calibration_fixed semi_mainCalibration = Calibration_fixed();
  auto response = semi_mainCalibration.getMarksSemiAutomatic(screenshot,
                                                              marks[0].first, marks[0].second,
                                                              marks[1].first, marks[1].second,
                                                              marks[2].first, marks[2].second,
                                                              marks[3].first, marks[3].second,
                                                              marks[4].first, marks[4].second,
                                                              marks[5].first, marks[5].second);

  // Crear un objeto JSON para contener los valores de respuesta
  v8::Local<v8::Object> responseObject = v8::Object::New(isolate);
  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("status").ToLocalChecked(),
                      Nan::New(std::get<0>(response))); // Estado de la respuesta

  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("message").ToLocalChecked(),
                      Nan::New(std::get<1>(response).c_str()).ToLocalChecked()); // Mensaje de la respuesta


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
                      

  responseObject->Set(isolate->GetCurrentContext(),
                      Nan::New("H_string").ToLocalChecked(),
                      Nan::New(std::get<3>(response).c_str()).ToLocalChecked());

  responseObject->Set(isolate->GetCurrentContext(),
                    Nan::New("calib_w").ToLocalChecked(),
                    Nan::New(std::get<4>(response)));

  responseObject->Set(isolate->GetCurrentContext(),
                    Nan::New("calib_h").ToLocalChecked(),
                    Nan::New(std::get<5>(response)));
  
  // Convierte el objeto de respuesta en una cadena JSON
  v8::Local<v8::String> jsonResponse = JSON::Stringify(isolate->GetCurrentContext(), responseObject).ToLocalChecked();
  info.GetReturnValue().Set(jsonResponse);
}

NAN_METHOD(MainAddon::AutoAnalysis) {
  
  v8::Isolate* isolate = info.GetIsolate();

  v8::String::Utf8Value v8Contourjson(isolate, info[0]);
  std::string contourjson(*v8Contourjson);

  v8::String::Utf8Value v8VideoUrl(isolate, info[1]);
  std::string videoUrl(*v8VideoUrl);

  v8::String::Utf8Value v8ImageUrl(isolate, info[2]);
  std::string imageUrl(*v8ImageUrl);

  v8::String::Utf8Value v8JsonString(isolate, info[3]);
  std::string jsonString(*v8JsonString);

  // ============================================================================================= /
  ComputerVisionWeb CVW = ComputerVisionWeb();
  std::string response = CVW.mainFunction(contourjson, videoUrl, imageUrl, jsonString);

  // Crea un nuevo objeto de respuesta
  v8::Local<v8::Object> responseObject = Nan::New<v8::Object>();
  
  // Configura la propiedad 'output' en el objeto de respuesta
  Nan::Set(responseObject, Nan::New("output").ToLocalChecked(), Nan::New(response).ToLocalChecked());

  // Convierte el objeto de respuesta en una cadena JSON
  v8::Local<v8::Context> context = Nan::GetCurrentContext();
  v8::Local<v8::String> jsonResponse = Nan::To<v8::String>(
    JSON::Stringify(context, responseObject).ToLocalChecked()
  ).ToLocalChecked();

  info.GetReturnValue().Set(jsonResponse);
}