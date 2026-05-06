#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#include "particles_viewer_opengl.hpp"

const int WIDTH = 800;  // Tamaño de la ventana
const int HEIGHT = 800;

const float PARTICLE_RADIUS = 2.0f; 
const float COLLISION_DISTANCE = PARTICLE_RADIUS * 2.0f;    // Radio de partícula para colisiones
const float COLLISION_RESTITUTION = 0.6f;

const int CELL_SIZE = 8;    // Tamaño de celda para la cuadrícula espacial
const int GRID_W = (WIDTH + CELL_SIZE - 1) / CELL_SIZE;     // Numero de celdas en X    
const int GRID_H = (HEIGHT + CELL_SIZE - 1) / CELL_SIZE;    // Numero de celdas en Y
const int CELL_COUNT = GRID_W * GRID_H;

const int N = 80000; // NUM PARTICULAS

struct Particle {   // Posicion, velocidad y color
    float pos_x, pos_y;
    float vel_x, vel_y;
    float color_r, color_g, color_b;
};

struct Pixel {
    unsigned char r, g, b;
};

std::vector<Particle> particles;        // Vector principal de partículas
std::vector<Particle> particlesNext;    // Vector auxiliar para resolver colisiones
std::vector<Pixel> img;
std::vector<int> cellHeads;
std::vector<int> nextParticle;
ParticleViewerState viewerState;    // Estado del visualizador (para mostrar FPS y controlar parámetros)


float dt = 0.016f;
float gravity = 9.8f;
const float GRAVITY_STEP = 0.5f;
float bouncyness = 0.6f;
const float BOUNCYNESS_STEP = 0.05f;
int workerCount = 1;


int clampInt(int value, int minValue, int maxValue) {
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

template <typename Func>
void parallelFor(int count, const Func& func) {
    int threadCount = std::min(workerCount, count);
    if (threadCount <= 1) {
        func(0, count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    int chunkSize = (count + threadCount - 1) / threadCount;
    for (int threadIndex = 0; threadIndex < threadCount; threadIndex++) {
        int start = threadIndex * chunkSize;
        int end = std::min(count, start + chunkSize);
        if (start >= end) break;

        threads.emplace_back([&, start, end]() {
            func(start, end);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}


// -- Inicializa partículas con posiciones, velocidades y colores aleatorios
void initParticles() {
    particles.resize(N);
    for (int i = 0; i < N; i++) {
        particles[i].pos_x = PARTICLE_RADIUS + float(rand()) / RAND_MAX * (WIDTH - 2.0f * PARTICLE_RADIUS);
        particles[i].pos_y = PARTICLE_RADIUS + float(rand()) / RAND_MAX * (HEIGHT - 2.0f * PARTICLE_RADIUS);
        particles[i].vel_x = (float(rand()) / RAND_MAX - 0.5f) * 100.0f;
        particles[i].vel_y = (float(rand()) / RAND_MAX - 0.5f) * 100.0f;
        // Amarillo o azul aleatorio
        if (rand() % 2 == 0) {
            particles[i].color_r = 1.0f;
            particles[i].color_g = 1.0f;
            particles[i].color_b = 0.0f;  // Amarillo
        } else {
            particles[i].color_r = 0.0f;
            particles[i].color_g = 0.65f;
            particles[i].color_b = 1.0f;  // Azul
        }
    }
}

void updateParticles() {
    parallelFor(N, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            // gravedad
            particles[i].vel_y -= gravity * dt;

            // posición
            particles[i].pos_x += particles[i].vel_x * dt;
            particles[i].pos_y += particles[i].vel_y * dt;

            // colisiones con bordes
            if (particles[i].pos_x < 0) { particles[i].pos_x = 0; particles[i].vel_x *= -0.6f; }
            if (particles[i].pos_x > WIDTH) { particles[i].pos_x = WIDTH; particles[i].vel_x *= -0.6f; }

            if (particles[i].pos_y < 0) { particles[i].pos_y = 0; particles[i].vel_y *= -0.6f; }
            if (particles[i].pos_y > HEIGHT) { particles[i].pos_y = HEIGHT; particles[i].vel_y *= -0.6f; }
        }
    });
}

void buildSpatialGrid() {
    std::fill(cellHeads.begin(), cellHeads.end(), -1);
    std::fill(nextParticle.begin(), nextParticle.end(), -1);

    for (int i = 0; i < N; i++) {
        int cellX = clampInt((int)(particles[i].pos_x / CELL_SIZE), 0, GRID_W - 1);
        int cellY = clampInt((int)(particles[i].pos_y / CELL_SIZE), 0, GRID_H - 1);
        int cellIndex = cellY * GRID_W + cellX;

        nextParticle[i] = cellHeads[cellIndex];
        cellHeads[cellIndex] = i;
    }
}

void resolveCollisions() {
    particlesNext = particles;

    parallelFor(N, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            Particle self = particles[i];
            int cellX = clampInt((int)(self.pos_x / CELL_SIZE), 0, GRID_W - 1);
            int cellY = clampInt((int)(self.pos_y / CELL_SIZE), 0, GRID_H - 1);

            for (int offsetY = -1; offsetY <= 1; offsetY++) {
                int neighborY = cellY + offsetY;
                if (neighborY < 0 || neighborY >= GRID_H) continue;

                for (int offsetX = -1; offsetX <= 1; offsetX++) {
                    int neighborX = cellX + offsetX;
                    if (neighborX < 0 || neighborX >= GRID_W) continue;

                    int cellIndex = neighborY * GRID_W + neighborX;
                    for (int j = cellHeads[cellIndex]; j != -1; j = nextParticle[j]) {
                        if (j == i) continue;

                        const Particle& other = particles[j];
                        float dx = self.pos_x - other.pos_x;
                        float dy = self.pos_y - other.pos_y;
                        float distanceSquared = dx * dx + dy * dy;

                        if (distanceSquared <= 0.0001f || distanceSquared >= COLLISION_DISTANCE * COLLISION_DISTANCE) {
                            continue;
                        }

                        float distance = sqrtf(distanceSquared);
                        float overlap = COLLISION_DISTANCE - distance;
                        float nx = dx / distance;
                        float ny = dy / distance;

                        self.pos_x += nx * overlap * 0.5f;
                        self.pos_y += ny * overlap * 0.5f;

                        float relativeVelocity = (self.vel_x - other.vel_x) * nx + (self.vel_y - other.vel_y) * ny;
                        if (relativeVelocity < 0.0f) {
                            float impulse = -(1.0f + bouncyness) * relativeVelocity * 0.5f;
                            self.vel_x += nx * impulse;
                            self.vel_y += ny * impulse;
                        }
                    }
                }
            }

            if (self.pos_x < 0) { self.pos_x = 0; self.vel_x *= -0.6f; }
            if (self.pos_x > WIDTH) { self.pos_x = WIDTH; self.vel_x *= -0.6f; }
            if (self.pos_y < 0) { self.pos_y = 0; self.vel_y *= -0.6f; }
            if (self.pos_y > HEIGHT) { self.pos_y = HEIGHT; self.vel_y *= -0.6f; }

            particlesNext[i] = self;
        }
    });

    particles.swap(particlesNext);
}

void renderParticles() {
    std::fill(img.begin(), img.end(), Pixel{0, 0, 0});

    for (int i = 0; i < N; i++) {
        int x = (int)particles[i].pos_x;
        int y = (int)particles[i].pos_y;

        if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) continue;

        int idx = y * WIDTH + x;
        img[idx].r = (unsigned char)(particles[i].color_r * 255.0f);
        img[idx].g = (unsigned char)(particles[i].color_g * 255.0f);
        img[idx].b = (unsigned char)(particles[i].color_b * 255.0f);
    }
}

void stepSimulation() {
    updateParticles();
    renderParticles();
    buildSpatialGrid();
    resolveCollisions();
}

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr)); // Inicializar semilla para números aleatorios

    workerCount = std::max(1u, std::thread::hardware_concurrency());

    img.resize(WIDTH * HEIGHT); // buffer de píxeles para renderizar
    cellHeads.resize(CELL_COUNT); // para la cuadrícula espacial (para colisiones)
    nextParticle.resize(N);
    particlesNext.resize(N);

    // Configuracion del visualizador
    viewerState.width = WIDTH; 
    viewerState.height = HEIGHT;
    viewerState.image = img.data();

    // Punteros para controlar gravedad y bouncyness desde el visualizador
    viewerState.gravity = &gravity;
    viewerState.bouncyness = &bouncyness;

    // Varables para mostrar FPS
    viewerState.currentFPS = 0.0f;
    viewerState.lastFrameTime = std::chrono::high_resolution_clock::now();
    viewerState.frameCount = 0;
    
    // Pasos para ajustar gravedad y rebote desde el visualizador
    viewerState.gravityStep = GRAVITY_STEP;
    viewerState.bouncynessStep = BOUNCYNESS_STEP;
    viewerState.windowTitlePrefix = "Particles CPU + OpenGL";

    initParticles();    // Inicializar partículas

    particleViewerInit(argc, argv, viewerState, stepSimulation);    // Inicializar visualizador y pasar función de paso de simulación

    stepSimulation();   // Ejecutar paso inicial

    glutMainLoop(); // Bucle principal de OpenGL

    return 0;
}
