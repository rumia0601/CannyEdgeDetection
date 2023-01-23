#include <iostream>
#include <time.h>
#include <chrono>
#include <string.h>
#include <math.h>
using namespace std;

///////////////////////CPU_Func/////////////////////////////
void Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len);
void Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian);	
void Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle);
void Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max);
void Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max);
/////////////////////////////////////////////////////////////


////////////////////GPU_Func///////////////////////////////
void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len);
void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian);
void GPU_Intensity_Gradient(int width, int height, uint8_t * gaussian, uint8_t* sobel, uint8_t* angle );
void GPU_Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max);
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max); 
///////////////////////////////////////////////////////////

bool Image_Check(uint8_t *cpu, uint8_t *gpu, int len);

