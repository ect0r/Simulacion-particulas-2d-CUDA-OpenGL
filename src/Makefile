NVCC := nvcc
TARGET := particles
SRC := particles.cu
GL_LIBS := -lglut -lGLU -lGL
TARGET_CPU := particles_cpu
SRC_CPU := particles_cpu.cpp
VIEWER_SRC := particles_viewer_opengl.cpp
VIEWER_OBJ := particles_viewer_opengl.o
CXX := g++

all: $(TARGET) $(TARGET_CPU)

$(VIEWER_OBJ): $(VIEWER_SRC) particles_viewer_opengl.hpp
	$(CXX) -O3 -c $(VIEWER_SRC) -o $(VIEWER_OBJ)

$(TARGET): $(SRC) $(VIEWER_OBJ)
	$(NVCC) -O3 $(SRC) $(VIEWER_OBJ) -o $(TARGET) $(GL_LIBS)

$(TARGET_CPU): $(SRC_CPU) $(VIEWER_OBJ)
	$(CXX) -O3 $(SRC_CPU) $(VIEWER_OBJ) -o $(TARGET_CPU) $(GL_LIBS) -lm -pthread

run: $(TARGET)
	./$(TARGET)

run_cpu: $(TARGET_CPU)
	./$(TARGET_CPU)

clean:
	rm -f $(TARGET) $(TARGET_CPU) $(VIEWER_OBJ) particles.ppm particles_*.ppm

.PHONY: all run run_cpu clean
