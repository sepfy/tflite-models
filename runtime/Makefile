INCLUDE = -I ./tensorflow/ -I ./tensorflow/tensorflow/lite/tools/make/downloads/absl/ -I ./tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/

LIB = libtensorflow-lite.a -lpthread `pkg-config opencv --cflags --libs`

all:
	$(CXX) $(INCLUDE) main.cc $(LIB) -o main
