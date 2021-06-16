#include <sys/time.h>
#include <cstdio>
#include <vector>
#include <math.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "face_detector.h"

using namespace cv;
using namespace std;

float kIoU = 0.3;
float kThreshold = 0.7;
const int kWidth = 128;
const int kHeight = 128;

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

  FaceDetector face_detector(kWidth, kHeight, kThreshold, kIoU);
  face_detector.LoadModel(argv[1]);
  
  Mat image = imread(argv[2]);
  Mat src;
  image.copyTo(src);
  resize(image, image, Size(kWidth, kHeight), 0, 0, INTER_NEAREST);
  cvtColor(image, image, CV_BGR2RGB);
  image.convertTo(image, CV_32FC3);
  image = (image - 127.5)/127.5;

  long long start = getms();
  std::vector<Box> detection_boxes = face_detector.Inference((float*)image.data);
  printf("inference = %lld\n", getms() - start);
  for(size_t i = 0; i < detection_boxes.size(); i++) {
    float lx = (detection_boxes[i].cx - detection_boxes[i].w*0.5)*(float)src.size().width;
    float ly = (detection_boxes[i].cy - detection_boxes[i].h*0.5)*(float)src.size().height;
    float rx = (detection_boxes[i].cx + detection_boxes[i].w*0.5)*(float)src.size().width;
    float ry = (detection_boxes[i].cy + detection_boxes[i].h*0.5)*(float)src.size().height;
    cv::Point pt1(lx, ly);
    cv::Point pt2(rx, ry);
    cv::rectangle(src, pt1, pt2, cv::Scalar(0, 255, 0));
    for(int j = 0; j < 12; j+=2) {
      float cx = detection_boxes[i].points[j]*(float)src.size().width;
      float cy = detection_boxes[i].points[j+1]*(float)src.size().height;
      cv::Point pt3(cx, cy);
      cv::circle(src, pt3, 5, Scalar( 0, 0, 255 ), FILLED, LINE_8);
    }
    printf("%d, %f, %f, %f ,%f\n", i, lx, ly, rx, ry);
  }

  imwrite("blazeface.jpg", src);

  return 0;
}
