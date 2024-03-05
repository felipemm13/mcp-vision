#ifndef MainAddon_H
#define MainAddon_H

#include <nan.h>

#include "Calibration_fixed.h"
#include "ConvertImage.h"

using namespace v8;
using namespace node;

class MainAddon: public Nan::ObjectWrap {
public:
  static void Init(Local<Object> target);
  static NAN_METHOD(CreateImage);
  

  static NAN_METHOD(SetDetection);
  static NAN_METHOD(SetCalibrationAutomatic);
  static NAN_METHOD(SetCalibrationSemiAutomatic);  


};

#endif