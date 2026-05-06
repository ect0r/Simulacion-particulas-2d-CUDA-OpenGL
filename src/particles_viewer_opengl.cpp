#include "particles_viewer_opengl.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Solo se ven y modifica desde este archivo, pero se inicializa desde el main de particles_cpu.cpp
static ParticleViewerState* g_particleViewerState = nullptr;    
static void (*g_particleViewerStepFn)() = nullptr;  // Puntero a funcion de paso de simulación

// Dibuja texto en la pantalla (para mostrar parámetros y FPS)
static void particleViewerRenderText(float x, float y, const char* text) {
    glRasterPos2f(x, y);
    while (*text) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *text);
        ++text;
    }
}

// Actualiza el título de la ventana con los parámetros actuales
static void particleViewerUpdateWindowTitle() {
    if (!g_particleViewerState) return;

    char title[256];
    std::snprintf(
        title,
        sizeof(title),
        "%s - Gravity: %.2f (Up/Down) Bouncyness: %.2f (O/I)",
        g_particleViewerState->windowTitlePrefix,
        *g_particleViewerState->gravity,
        *g_particleViewerState->bouncyness
    );
    glutSetWindowTitle(title);
}

static void particleViewerDisplay() {
    if (!g_particleViewerState) return;

    glClear(GL_COLOR_BUFFER_BIT);   // Limpiar pantalla frame a frame
    glRasterPos2i(-1, -1);
    glDrawPixels(   // Dibujar la imagen de partículas 
        g_particleViewerState->width,
        g_particleViewerState->height,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        g_particleViewerState->image
    );

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, g_particleViewerState->width, g_particleViewerState->height, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(0.0f, 1.0f, 1.0f);
    
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "Gravity: %.2f", *g_particleViewerState->gravity);
    particleViewerRenderText(10, 30, buffer);

    std::snprintf(buffer, sizeof(buffer), "Bouncyness: %.2f", *g_particleViewerState->bouncyness);
    particleViewerRenderText(10, 50, buffer);

    std::snprintf(buffer, sizeof(buffer), "FPS: %.1f", g_particleViewerState->currentFPS);
    particleViewerRenderText(10, 70, buffer);

    // Tiempos y porcentajes de frame (frameMs medido directamente, no calculado desde FPS)
    float frameMs = g_particleViewerState->frameMs;
    float pctMemcpy = (frameMs > 0)
        ? (g_particleViewerState->memcpyMs / frameMs * 100.0f) : 0.0f;
    float pctK3 = (frameMs > 0)
        ? (g_particleViewerState->k3Ms / frameMs * 100.0f) : 0.0f;

    std::snprintf(buffer, sizeof(buffer), "memcpy D->H: %.3f ms (%.1f%% frame)",
                  g_particleViewerState->memcpyMs, pctMemcpy);
    particleViewerRenderText(10, 90, buffer);

    std::snprintf(buffer, sizeof(buffer), "K3 colisiones: %.3f ms (%.1f%% frame)",
                  g_particleViewerState->k3Ms, pctK3);
    particleViewerRenderText(10, 110, buffer);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glutSwapBuffers();
}

static void particleViewerIdle() {
    if (g_particleViewerStepFn) {
        g_particleViewerStepFn();
    }

    if (g_particleViewerState) {
        g_particleViewerState->frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - g_particleViewerState->lastFrameTime).count();

        if (elapsed >= 1000) {
            g_particleViewerState->currentFPS = g_particleViewerState->frameCount * 1000.0f / elapsed;
            g_particleViewerState->frameCount = 0;
            g_particleViewerState->lastFrameTime = currentTime;
        }
    }

    glutPostRedisplay();
}

static void particleViewerReshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void particleViewerSpecialKeyboard(int key, int, int) {
    if (!g_particleViewerState) return;

    switch (key) {
        case GLUT_KEY_UP:
            *g_particleViewerState->gravity += g_particleViewerState->gravityStep;
            particleViewerUpdateWindowTitle();
            break;
        case GLUT_KEY_DOWN:
            *g_particleViewerState->gravity = std::max(0.0f, *g_particleViewerState->gravity - g_particleViewerState->gravityStep);
            particleViewerUpdateWindowTitle();
            break;
    }
}

static void particleViewerKeyboard(unsigned char key, int, int) {
    if (!g_particleViewerState) return;

    switch (key) {
        case 'r':
        case 'R':
            *g_particleViewerState->gravity = 9.8f;
            particleViewerUpdateWindowTitle();
            break;
        case 'o':
        case 'O':
            *g_particleViewerState->bouncyness = std::min(1.0f, *g_particleViewerState->bouncyness + g_particleViewerState->bouncynessStep);
            particleViewerUpdateWindowTitle();
            break;
        case 'i':
        case 'I':
            *g_particleViewerState->bouncyness = std::max(0.0f, *g_particleViewerState->bouncyness - g_particleViewerState->bouncynessStep);
            particleViewerUpdateWindowTitle();
            break;
        case 27:
            std::exit(0);
            break;
    }
}

void particleViewerInit(int& argc, char** argv, ParticleViewerState& state, void (*stepFn)()) {

    g_particleViewerState = &state;     // Copiar la referencia del estado para usarlo en callbacks
    g_particleViewerStepFn = stepFn;    // Copiar la referencia de la función de paso de simulación

    glutInit(&argc, argv);  // Inicializar GLUT
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);   // Doble buffer para evitar parpadeos (dibujar y mostrar)
    glutInitWindowSize(state.width, state.height);  // Tamaño de la ventana
    glutCreateWindow(state.windowTitlePrefix);  // Crear ventana con título inicial

    glutDisplayFunc(particleViewerDisplay);         // Al dibujar                           -> llama a esta función
    glutIdleFunc(particleViewerIdle);               // Cuando no hace nada                  -> llama a esta función (para actualizar la simulación)
    glutReshapeFunc(particleViewerReshape);         // Al cambiar el tamaño de la ventana   -> llama a esta función (para ajustar la vista)
    glutSpecialFunc(particleViewerSpecialKeyboard); // Cuando presiones (arriba/abajo)      -> llama a esta función
    glutKeyboardFunc(particleViewerKeyboard);       // Cuando presiones teclas (R)          -> llama a esta función (para controlar parámetros)
    
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Color de fondo

    particleViewerUpdateWindowTitle();  // Titulo
}
