#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "particles_viewer_opengl.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (" << (int)err << ")" << std::endl; \
            cudaExitCode = 1; \
            goto cleanup; \
        } \
    } while(0)

const int WIDTH = 800;
const int HEIGHT = 800;
const float PARTICLE_RADIUS = 2.0f;
const float COLLISION_DISTANCE = PARTICLE_RADIUS * 2.0f;
const int CELL_SIZE = 8;
const int GRID_W = (WIDTH + CELL_SIZE - 1) / CELL_SIZE;
const int GRID_H = (HEIGHT + CELL_SIZE - 1) / CELL_SIZE;
const int CELL_COUNT = GRID_W * GRID_H;

const int N = 80000; // partículas

struct Particle {
    float2 pos;
    float2 vel;
    float3 color;
};

struct Pixel {
    unsigned char r, g, b;
};

Particle* d_particles = nullptr;
Particle* d_snapshot = nullptr;
Pixel* d_img = nullptr;
int* d_cellHeads = nullptr;
int* d_nextParticle = nullptr;
Pixel* h_img = nullptr;
Particle* h_particles = nullptr;
ParticleViewerState viewerState;
dim3 block(256);
dim3 grid((N + block.x - 1) / block.x);
float dt = 0.016f;
float gravity = 9.8f;
const float GRAVITY_STEP = 0.5f;
float bouncyness = 0.6f;
const float BOUNCYNESS_STEP = 0.05f;
int cudaExitCode = 0;

// Eventos CUDA para medir tiempos
cudaEvent_t memcpyStart, memcpyStop;
cudaEvent_t k3Start, k3Stop;
cudaEvent_t frameStart, frameStop;

void releaseResources();

__device__ int clampInt(int value, int minValue, int maxValue) {
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

__global__ void updateParticles(Particle* p, float dt, float g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // gravedad
    p[i].vel.y -= g * dt;

    // posición
    p[i].pos.x += p[i].vel.x * dt;
    p[i].pos.y += p[i].vel.y * dt;

    // colisiones con bordes
    if (p[i].pos.x < 0) { p[i].pos.x = 0; p[i].vel.x *= -0.6f; }
    if (p[i].pos.x > WIDTH) { p[i].pos.x = WIDTH; p[i].vel.x *= -0.6f; }

    if (p[i].pos.y < 0) { p[i].pos.y = 0; p[i].vel.y *= -0.6f; }
    if (p[i].pos.y > HEIGHT) { p[i].pos.y = HEIGHT; p[i].vel.y *= -0.6f; }
}

__global__ void buildSpatialGrid(const Particle* particles, int* cellHeads, int* nextParticle) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cellX = clampInt((int)(particles[i].pos.x / CELL_SIZE), 0, GRID_W - 1);
    int cellY = clampInt((int)(particles[i].pos.y / CELL_SIZE), 0, GRID_H - 1);
    int cellIndex = cellY * GRID_W + cellX;

    nextParticle[i] = atomicExch(&cellHeads[cellIndex], i);
}

__global__ void resolveCollisions(const Particle* snapshot, Particle* particles, const int* cellHeads, const int* nextParticle, float restitution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Particle self = snapshot[i];
    int cellX = clampInt((int)(self.pos.x / CELL_SIZE), 0, GRID_W - 1);
    int cellY = clampInt((int)(self.pos.y / CELL_SIZE), 0, GRID_H - 1);

    for (int offsetY = -1; offsetY <= 1; offsetY++) {
        int neighborY = cellY + offsetY;
        if (neighborY < 0 || neighborY >= GRID_H) continue;

        for (int offsetX = -1; offsetX <= 1; offsetX++) {
            int neighborX = cellX + offsetX;
            if (neighborX < 0 || neighborX >= GRID_W) continue;

            int cellIndex = neighborY * GRID_W + neighborX;
            for (int j = cellHeads[cellIndex]; j != -1; j = nextParticle[j]) {
                if (j == i) continue;

                float dx = self.pos.x - snapshot[j].pos.x;
                float dy = self.pos.y - snapshot[j].pos.y;
                float distanceSquared = dx * dx + dy * dy;

                if (distanceSquared <= 0.0001f || distanceSquared >= COLLISION_DISTANCE * COLLISION_DISTANCE) {
                    continue;
                }

                float distance = sqrtf(distanceSquared);
                float overlap = COLLISION_DISTANCE - distance;
                float nx = dx / distance;
                float ny = dy / distance;

                self.pos.x += nx * overlap * 0.5f;
                self.pos.y += ny * overlap * 0.5f;

                float relativeVelocity = (self.vel.x - snapshot[j].vel.x) * nx + (self.vel.y - snapshot[j].vel.y) * ny;
                if (relativeVelocity < 0.0f) {
                    float impulse = -(1.0f + restitution) * relativeVelocity * 0.5f;
                    self.vel.x += nx * impulse;
                    self.vel.y += ny * impulse;
                }
            }
        }
    }

    if (self.pos.x < 0) { self.pos.x = 0; self.vel.x *= -0.6f; }
    if (self.pos.x > WIDTH) { self.pos.x = WIDTH; self.vel.x *= -0.6f; }
    if (self.pos.y < 0) { self.pos.y = 0; self.vel.y *= -0.6f; }
    if (self.pos.y > HEIGHT) { self.pos.y = HEIGHT; self.vel.y *= -0.6f; }

    particles[i] = self;
}

__global__ void renderParticles(Particle* p, Pixel* img) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int x = (int)p[i].pos.x;
    int y = (int)p[i].pos.y;

    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return;

    int idx = y * WIDTH + x;

    img[idx].r = (unsigned char)(p[i].color.x * 255.0f);
    img[idx].g = (unsigned char)(p[i].color.y * 255.0f);
    img[idx].b = (unsigned char)(p[i].color.z * 255.0f);
}

void stepSimulation() {
    cudaEventRecord(frameStart);
    updateParticles<<<grid, block>>>(d_particles, dt, gravity);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_snapshot, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemset(d_cellHeads, 0xff, CELL_COUNT * sizeof(int)));

    buildSpatialGrid<<<grid, block>>>(d_snapshot, d_cellHeads, d_nextParticle);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(k3Start);
    resolveCollisions<<<grid, block>>>(d_snapshot, d_particles, d_cellHeads, d_nextParticle, bouncyness);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(k3Stop);
    cudaEventSynchronize(k3Stop);
    cudaEventElapsedTime(&viewerState.k3Ms, k3Start, k3Stop);

    CUDA_CHECK(cudaMemset(d_img, 0, WIDTH * HEIGHT * sizeof(Pixel)));

    renderParticles<<<grid, block>>>(d_particles, d_img);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Medir el tiempo del cudaMemcpy DeviceToHost
    cudaEventRecord(memcpyStart);
    CUDA_CHECK(cudaMemcpy(h_img, d_img, WIDTH * HEIGHT * sizeof(Pixel), cudaMemcpyDeviceToHost));
    cudaEventRecord(memcpyStop);
    cudaEventSynchronize(memcpyStop);
    cudaEventElapsedTime(&viewerState.memcpyMs, memcpyStart, memcpyStop);

    cudaEventRecord(frameStop);
    cudaEventSynchronize(frameStop);
    cudaEventElapsedTime(&viewerState.frameMs, frameStart, frameStop);

    return;

cleanup:
    releaseResources();
    std::exit(cudaExitCode);
}

void releaseResources() {
    cudaEventDestroy(memcpyStart);
    cudaEventDestroy(memcpyStop);
    cudaEventDestroy(k3Start);
    cudaEventDestroy(k3Stop);
    cudaEventDestroy(frameStart);
    cudaEventDestroy(frameStop);
    if (d_particles) { cudaFree(d_particles); d_particles = nullptr; }
    if (d_snapshot) { cudaFree(d_snapshot); d_snapshot = nullptr; }
    if (d_img) { cudaFree(d_img); d_img = nullptr; }
    if (d_cellHeads) { cudaFree(d_cellHeads); d_cellHeads = nullptr; }
    if (d_nextParticle) { cudaFree(d_nextParticle); d_nextParticle = nullptr; }
    delete[] h_particles;
    h_particles = nullptr;
    delete[] h_img;
    h_img = nullptr;
}

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));

    h_img = new Pixel[WIDTH * HEIGHT];
    h_particles = new Particle[N];
    viewerState.width = WIDTH;
    viewerState.height = HEIGHT;
    viewerState.image = h_img;
    viewerState.gravity = &gravity;
    viewerState.bouncyness = &bouncyness;
    viewerState.currentFPS = 0.0f;
    viewerState.memcpyMs = 0.0f;   // Inicializar a 0
    viewerState.k3Ms = 0.0f;
    viewerState.frameMs = 0.0f;
    viewerState.lastFrameTime = std::chrono::high_resolution_clock::now();
    viewerState.frameCount = 0;
    viewerState.gravityStep = GRAVITY_STEP;
    viewerState.bouncynessStep = BOUNCYNESS_STEP;
    viewerState.windowTitlePrefix = "Particles CUDA + OpenGL";
    atexit(releaseResources);

    // Crear eventos de medición antes de usarlos
    cudaEventCreate(&memcpyStart);
    cudaEventCreate(&memcpyStop);
    cudaEventCreate(&k3Start);
    cudaEventCreate(&k3Stop);
    cudaEventCreate(&frameStart);
    cudaEventCreate(&frameStop);

    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_snapshot, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_img, WIDTH * HEIGHT * sizeof(Pixel)));
    CUDA_CHECK(cudaMalloc(&d_cellHeads, CELL_COUNT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextParticle, N * sizeof(int)));

    // init CPU particles
    for (int i = 0; i < N; i++) {
        h_particles[i].pos = make_float2(
            PARTICLE_RADIUS + float(rand()) / RAND_MAX * (WIDTH - 2.0f * PARTICLE_RADIUS),
            PARTICLE_RADIUS + float(rand()) / RAND_MAX * (HEIGHT - 2.0f * PARTICLE_RADIUS)
        );
        h_particles[i].vel = make_float2(
            (float(rand()) / RAND_MAX - 0.5f) * 100.0f,
            (float(rand()) / RAND_MAX - 0.5f) * 100.0f
        );
        // Amarillo o rojo aleatorio
        if (rand() % 2 == 0) {
            h_particles[i].color = make_float3(1.0f, 1.0f, 0.0f);  // Amarillo
        } else {
            h_particles[i].color = make_float3(0.0f, 1.0f, 0.0f);  // Rojo
        }
    }

    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));

    particleViewerInit(argc, argv, viewerState, stepSimulation);

    stepSimulation();

    glutMainLoop();

cleanup:
    releaseResources();
    return cudaExitCode;
}
