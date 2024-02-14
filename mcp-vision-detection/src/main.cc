#include <nan.h>

#include "lib/MainAddon.h"

using namespace v8;

extern "C" void init(Local<Object> target) {
  MainAddon::Init(target);
}

NODE_MODULE(server, init)
