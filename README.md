# tflite-models
Train with keras and inference with tflite.

### Run tflite on Raspberry Pi

```bash
sudo apt-get install build-essential git libopencv-dev -y
git clone https://github.com/tensorflow/tensorflow
git clone https://github.com/sepfy/tflite-models.git
```
Follow the instructions <b>Compile natively on Raspberry Pi</b> in Tensorflow Lite [website](https://www.tensorflow.org/lite/guide/build_rpi) to build Tensorflow lite. Copy libtensorflow-lite.a to this project.
```bash
cp tensorflow/tensorflow/lite/tools/make/gen/lib/rpi_armv7/libtensorflow-lite.a tflite-models/
```

Compile this sample:
```bash
make
```
