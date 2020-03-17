#include <sys/time.h>
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace tflite;


#define IM_SIZE 224*224*3
#define CLASSES 3

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

unsigned long long getms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1.0e+3 + tv.tv_usec/1000;
}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <image>\n");
    return 1;
  }

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(argv[1]);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //tflite::PrintInterpreterState(interpreter.get());


  Mat image = imread(argv[2]);
  resize(image, image, Size(224, 224), 0, 0, INTER_LINEAR);
  cvtColor(image, image, CV_BGR2RGB);
  image.convertTo(image, CV_32FC3);
  // Fill input buffers
  memcpy(interpreter->typed_input_tensor<float>(0), image.data, IM_SIZE*sizeof(float));

  // Run inference
  long long s = getms();
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("inference time = %lld ms\n", getms() - s);
  //tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  float* output = interpreter->typed_output_tensor<float>(0);
  for(int i = 0; i < CLASSES; i++)
    printf("Class %d: %.2f\n", i, output[i]);

  return 0;
}
