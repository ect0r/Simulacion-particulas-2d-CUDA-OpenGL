#pragma once
#include <GL/glut.h>
#include <chrono>

struct ParticleViewerState {    // Guarda el estado del visualizador (tamaño, imagen, parámetros controlables y FPS)
    int width;      // Tamaño de la ventana
    int height;
    const void* image;  // Buffer de pixeles
    float* gravity;     // Gravedad y rebote
    float* bouncyness;
    float currentFPS;   // FPS
    float memcpyMs;     // Tiempo del cudaMemcpy DeviceToHost en ms
    float k3Ms;         // Tiempo del kernel resolveCollisions en ms
    float frameMs;      // Tiempo total del frame medido directamente en ms
    std::chrono::high_resolution_clock::time_point lastFrameTime;
    int frameCount;
    float gravityStep;  // Steps
    float bouncynessStep;
    const char* windowTitlePrefix;  // titulo
};

void particleViewerInit(int& argc, char** argv, ParticleViewerState& state, void (*stepFn)());
