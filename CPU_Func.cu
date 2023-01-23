#include "Func.h"

void Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) {
    printf("%d\n",start_add);
    for (int i = start_add; i < len; i += 3) {
		int tmp = (buf[i] * 0.114 + buf[i + 1] * 0.587 + buf[i + 2] * 0.299);
		gray[i] = tmp;
		gray[i + 1] = tmp;
		gray[i + 2] = tmp;
	}

}

float conv2d_5x5(float* filter, uint8_t* pixel, int x, int y, int width) {
	float v = 0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			v += pixel[(y + i) * width + x + j] * filter[i * 5 + j];
		}
	}
	return v;
}

void Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian) {
	float filter[25] = {0}; 
	float sigma = 1.0;
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			filter[(i + 2) * 5 + j + 2]
				= (1 / (2 * 3.14* sigma * sigma)) * exp(-(i * i + j * j) / (2 * sigma * sigma));
		}
	}

	//zero padding
	uint8_t* tmp = (uint8_t*)malloc((width+4) * (height+4));
	memset(tmp, (uint8_t)0, (width + 4) * (height + 4));


	for (int i = 2; i < height+2; i++) {
		for (int j = 2; j < width+2; j++) {
			tmp[i * (width + 4) + j] = gray[((i - 2) * width + (j - 2)) * 3];
		}
	}
	
	//GaussianBlur

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uint8_t v = conv2d_5x5(filter,tmp,j, i,width+4);
			gaussian[(i * width + j)*3] = v;
			gaussian[(i * width + j) * 3 +1] = v;
			gaussian[(i * width + j) * 3 +2] = v;
		}
	}
	free(tmp);
}
void conv2d_3x3(int* filter_y, int* filter_x, uint8_t* pixel, int x, int y, int width, int &gx, int &gy) {
	//int gx = 0;
	//int gy = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			gy += (int)pixel[(y + i) * width + x + j] * filter_y[i *3 + j];
			gx += (int)pixel[(y + i) * width + x + j] * filter_x[i * 3 + j];
		}
	}
}

void Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle) {
	int filter_x[9] = {-1,0,1
						,-2,0,2
						,-1,0,1};
	int filter_y[9] = {1,2,1
						,0,0,0
						,-1,-2,-1};
						
	uint8_t* tmp = (uint8_t*)malloc((width + 2) * (height + 2));
	memset(tmp, (uint8_t)0, (width + 2) * (height + 2));

	//zero padding

	for (int i = 1; i < height + 1; i++) {
		for (int j = 1; j < width + 1; j++) {
			tmp[i * (width + 2) + j] = gaussian[((i - 1) * width + (j - 1)) * 3];
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int gx = 0;
			int gy = 0;
			conv2d_3x3(filter_y, filter_x, tmp, j, i, width + 2,gx,gy);
			int t = sqrt(gx * gx + gy * gy);
			uint8_t  v = 0;
			if (t > 255) {
				v = 255;
			}
			else
				v = t;

			sobel[(i * width + j) * 3] = v;
			sobel[(i * width + j) * 3 + 1] = v;
			sobel[(i * width + j) * 3 + 2] = v;
			
			float t_angle = 0;
			if(gy != 0 || gx != 0) 
				t_angle= (float)atan2(gy, gx) * 180.0 / 3.14;
			if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
				angle[i * width + j] = 0;
			else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
				angle[i * width + j] = 45;
			else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
				angle[i * width + j] = 90;
			else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
				angle[i * width + j] = 135;
		}
	}
	free(tmp);
}

void Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max)
{
	uint8_t p1 = 0;
	uint8_t p2 = 0;
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (angle[i * width + j] == 0) {
				p1 = sobel[((i+1) * width + j)*3];
				p2 = sobel[((i-1) * width + j) * 3];
			}
			else if (angle[i * width + j] == 45) {
				p1 = sobel[((i + 1) * width + j-1) * 3];
				p2 = sobel[((i - 1) * width + j+1) * 3];
			}
			else if (angle[i * width + j] == 90) {
				p1 = sobel[((i) * width + j+1) * 3];
				p2 = sobel[((i) * width + j-1) * 3];
			}
			else {
				p1 = sobel[((i + 1) * width + j+1) * 3];
				p2 = sobel[((i - 1) * width + j-1) * 3];
			}
			uint8_t v = sobel[(i * width + j) * 3];
			if(min > v)
				min = v;
			if(max < v)
				max = v;
			if ((v >= p1) && (v >= p2)) {
				suppression_pixel[(i * width + j) * 3] = v;
				suppression_pixel[(i * width + j) * 3 + 1] = v;
				suppression_pixel[(i * width + j) * 3 + 2] = v;
			}
			else {
				suppression_pixel[(i * width + j) * 3] = 0;
				suppression_pixel[(i * width + j) * 3 + 1] = 0;
				suppression_pixel[(i * width + j) * 3 + 2] = 0;
			}
		}
	}
}

void Hysteresis_check(int width, int height, int x, int y, uint8_t * hysteresis, uint8_t *tmp_hysteresis){
	for (int i = y-1; i < y+2; i++) {
		for (int j = x-1; j < x+2; j++) {
			if ((i < height && j < width) && (i >= 0 && j >= 0)) {
				if (tmp_hysteresis[(i * width + j)*3] == 255) {
					hysteresis[(y * width + x)*3] = 255;
					hysteresis[(y * width + x) * 3+1] = 255;
					hysteresis[(y * width + x) * 3+2] = 255;
					return;
				}
			}
		}
	}
}

void Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max) {
	uint8_t diff = max - min;
	uint8_t low_t = min + diff * 0.01;
	uint8_t high_t = min + diff * 0.2;
	uint8_t *tmp_hysteresis = (uint8_t*)malloc(sizeof(uint8_t)*width*height*3);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uint8_t v = suppression_pixel[(i * width + j)*3];
			if (v < low_t) {
				hysteresis[(i * width + j) * 3] = 0;
				hysteresis[(i * width + j) * 3+1] = 0;
				hysteresis[(i * width + j) * 3+2] = 0;
			}
			else if (v < high_t) {
				hysteresis[(i * width + j) * 3] = 123;
				hysteresis[(i * width + j) * 3 + 1] = 123;
				hysteresis[(i * width + j) * 3 + 2] = 123;
			}
			else {
				hysteresis[(i * width + j) * 3] = 255;
				hysteresis[(i * width + j) * 3 + 1] = 255;
				hysteresis[(i * width + j) * 3 + 2] = 255;
			}
		}
	}
//////////////////////Modified in Version3//////////////////////////////
	memcpy(tmp_hysteresis,hysteresis,sizeof(uint8_t)*width*height*3);
    	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if(tmp_hysteresis[(i*width+j)*3] == 123){
                		Hysteresis_check(width,height,j,i,hysteresis,tmp_hysteresis);
			}
		}
	}
//////////////////////////////////////////////////////////////////////
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (hysteresis[(i * width + j) * 3] != 255) {
				hysteresis[(i * width + j) * 3] = 0;
				hysteresis[(i * width + j) * 3+1] = 0;
				hysteresis[(i * width + j) * 3+2] = 0;
			}
		}
	}
	free(tmp_hysteresis);
}


bool Image_Check(uint8_t *cpu, uint8_t *gpu, int len){
    for(int i=0;i<len;i++){
        if(cpu[i] != gpu[i] && cpu[i] + 1 != gpu[i] && cpu[i] - 1 != gpu[i]){ 
            return false;
        }
    }
    return true;
}
