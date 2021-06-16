#include <sys/time.h>
#include <cstdio>
#include <vector>
#include <math.h>
#include <map>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace tflite;

float kIoU = 0.3;
float kThreshold = 0.7;

typedef struct Box {
  float cx;
  float cy;
  float w;
  float h;
  float confidence;
} Box;

#define WIDTH 128
#define HEIGHT 128
#define IM_SIZE WIDTH*HEIGHT*3
#define CLASSES 10

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

std::vector<Box> Gen(int input_width, int input_height, std::vector<int> strides, float min_scale=0.1484375, float max_scale=0.75) {

  std::vector<Box> anchor_boxes;
  float anchor_offset_x = 0.5;
  float anchor_offset_y = 0.5;

  for(size_t i = 0; i < strides.size(); i++) {
    int feature_map_width = input_width/strides[i];
    int feature_map_height = input_height/strides[i];

    for(int y = 0; y < feature_map_height; y++) {
      for(int x = 0; x < feature_map_width; x++) {
        float cx = ((float)x + anchor_offset_x)/(float)feature_map_width;
        float cy = ((float)y + anchor_offset_y)/(float)feature_map_height;
        float w = 1.0;
        float h = 1.0;
        Box anchor_box;
        anchor_box.w = w;
        anchor_box.h = h;
        anchor_box.cx = cx;
        anchor_box.cy = cy;
        anchor_boxes.push_back(anchor_box);
        anchor_boxes.push_back(anchor_box);
      }
    }
  }
  return anchor_boxes;
}

//std::map<int, float>

std::vector<Box> Extract(int width, int height, std::vector<Box> anchor_boxes, std::vector<Box> detection_boxes, float *scores, float threshold) {

  std::vector<Box> candidate_boxes;
  for(size_t i = 0; i < anchor_boxes.size(); i++) {
    if(scores[i] < threshold)
      continue;

    float cx = detection_boxes[i].cx/(float)width + anchor_boxes[i].cx;
    float cy = detection_boxes[i].cy/(float)height + anchor_boxes[i].cy;
    float w = detection_boxes[i].w/(float)width;
    float h = detection_boxes[i].h/(float)height;

    Box candidate_box = {cx, cy, w, h, scores[i]};
    candidate_boxes.push_back(candidate_box);
  //  printf("%d, %f, %f, %f ,%f\n", i, detection_boxes[i].cx, detection_boxes[i].cy, detection_boxes[i].w, detection_boxes[i].h);
  }
  return candidate_boxes;
}

int BoxConfidenceArgmax(std::vector<Box> boxes) {

  float max_confidence = 0;
  int max_index = 0;
  for(size_t i = 0; i < boxes.size(); i++) {
    if(boxes[i].confidence > max_confidence) {
      max_confidence = boxes[i].confidence;
      max_index = i;
    }
  }
  return max_index;
}

std::vector<Box> NonMaxSupression(std::vector<Box> candidate_boxes) {

  std::vector<Box> detection_boxes;
  while(candidate_boxes.size() > 0) {
    size_t max_index = BoxConfidenceArgmax(candidate_boxes);
    Box selected_box = candidate_boxes[max_index];
    candidate_boxes.erase(candidate_boxes.begin() + max_index);
    detection_boxes.push_back(selected_box);

    float selected_box_left = selected_box.cx - 0.5*selected_box.w;
    float selected_box_top = selected_box.cy - 0.5*selected_box.h;
    float selected_box_right = selected_box.cx + 0.5*selected_box.w;
    float selected_box_bottom = selected_box.cy + 0.5*selected_box.h;

    auto box = candidate_boxes.begin();

    while(box != candidate_boxes.end()) {

      float box_left = box->cx - 0.5*box->w;
      float box_top = box->cy - 0.5*box->h;
      float box_right = box->cx + 0.5*box->w;
      float box_bottom = box->cy + 0.5*box->h;

      float union_box_w = std::min(selected_box_right, box_right)
       - std::max(selected_box_left, box_left);
      float union_box_h = std::min(selected_box_bottom, box_bottom)
       - std::max(selected_box_top, box_top);
      float iou = (union_box_w*union_box_h)/(selected_box.w*selected_box.h + box->w*box->h
       - union_box_w*union_box_h);

      if(iou > kIoU && union_box_w > 0 && union_box_h > 0)
        box = candidate_boxes.erase(box);
      else
        ++box;
    }
  }

  return detection_boxes;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <image>\n");
    return 1;
  }

  std::unique_ptr<tflite::FlatBufferModel> model =
   tflite::FlatBufferModel::BuildFromFile(argv[1]);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  std::vector<int> strides;
  strides.push_back(8);
  strides.push_back(16);
  strides.push_back(16);
  strides.push_back(16);

  std::vector<Box> anchor_boxes = Gen(WIDTH, HEIGHT, strides);


  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //tflite::PrintInterpreterState(interpreter.get());
  
  Mat image = imread(argv[2]);
  Mat src;
  image.copyTo(src);
  resize(image, image, Size(WIDTH, HEIGHT), 0, 0, INTER_NEAREST);
  cvtColor(image, image, CV_BGR2RGB);
  image.convertTo(image, CV_32FC3);
  image = (image - 127.5)/127.5;
  //image = ((image/255.0) - 0.5)/0.5;
  // Fill input buffers
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]); 
  printf("%d\n", input_tensor->bytes);
  memcpy(interpreter->typed_input_tensor<float>(0), image.data, IM_SIZE*sizeof(float));

  // Run inference
  long long s = getms();
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("inference time = %lld ms\n", getms() - s);
//  tflite::PrintInterpreterState(interpreter.get());

  TfLiteTensor *box_tensor = interpreter->tensor(interpreter->outputs()[0]);
  TfLiteTensor *score_tensor = interpreter->tensor(interpreter->outputs()[1]);

  float threshold = log(kThreshold/(1.0 - kThreshold));
  std::vector<int> outputs = interpreter->outputs();
  size_t size = box_tensor->bytes/sizeof(float);
  printf("outputs = %d, %d\n", box_tensor->bytes/4, score_tensor->bytes/4);

  std::vector<Box> output_boxes;
  float *detections = interpreter->typed_output_tensor<float>(0);
  for(size_t i = 0; i < size; i=i+16) {
    Box box;
    box.cx = detections[i];
    box.cy = detections[i+1];
    box.w = detections[i+2];
    box.h = detections[i+3];
    output_boxes.push_back(box);
  }

  std::vector<Box> candidate_boxes = Extract(WIDTH, HEIGHT, anchor_boxes, output_boxes, interpreter->typed_output_tensor<float>(1), threshold);
 
  std::vector<Box> detection_boxes = NonMaxSupression(candidate_boxes);
 
  for(size_t i = 0; i < detection_boxes.size(); i++) {
    float lx = (detection_boxes[i].cx - detection_boxes[i].w*0.5)*(float)src.size().width;
    float ly = (detection_boxes[i].cy - detection_boxes[i].h*0.5)*(float)src.size().height;
    float rx = (detection_boxes[i].cx + detection_boxes[i].w*0.5)*(float)src.size().width;
    float ry = (detection_boxes[i].cy + detection_boxes[i].h*0.5)*(float)src.size().height;
    cv::Point pt1(lx, ly);
    cv::Point pt2(rx, ry);
    cv::rectangle(src, pt1, pt2, cv::Scalar(0, 255, 0));
    printf("%d, %f, %f, %f ,%f\n", i, lx, ly, rx, ry);
  }

  imwrite("blazeface.jpg", src);

  return 0;
}
