#include "Func.h"
#include <time.h>
/////////////////////////////////////
//
//  GPU Computing Project - Canny Edge Detection
//  GPU_Func.cu외 모든 Code는 수정하지 말 것
//  불가피하게 수정해야 할 경우 조교에게 문의
//
/////////////////////////////////////

int main()
{

	chrono::duration<double> c_end = chrono::duration<double>::zero();
	chrono::duration<double> g_end = chrono::duration<double>::zero();
 	chrono::system_clock::time_point time;
	int score = 0;
	double total_time = 0;

	FILE* fp = fopen("test_file.bmp", "rb");
	FILE* fp2 = fopen("gray_scale.bmp", "wb");
	FILE* fp3 = fopen("gaussian_blur.bmp", "wb");
	FILE* fp4 = fopen("sobel.bmp", "wb");
	FILE* fp5 = fopen("Non-maximum_suppression.bmp", "wb");
	FILE* fp6 = fopen("hysteresis.bmp", "wb");
	uint8_t test[200] = {0};
	fread(test, 200, 1, fp);
	
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	int width = 0;
	int height = 0;
	fseek(fp, 0, SEEK_SET); //go to beg.

	int * sync = NULL;
	cudaMalloc(&sync,sizeof(int));
	cudaFree(sync);

	uint8_t* buf = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* gray = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* gaussian = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* sobel = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* sobel_angle = (uint8_t*)malloc((len - test[10]) / 3); //malloc buffer
	uint8_t* suppression = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* hysteresis = (uint8_t*)malloc(len); //malloc buffer

	uint8_t* g_gray = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* g_gaussian = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* g_sobel = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* g_sobel_angle = (uint8_t*)malloc((len - test[10]) / 3); //malloc buffer
	uint8_t* g_suppression = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* g_hysteresis = (uint8_t*)malloc(len); //malloc buffer


	memset(buf, 0, len);
	memset(gray, 0, len);
	memset(gaussian, 0, len);
	memset(sobel, 0, len);
	memset(sobel_angle, 0, (len - test[10]) / 3);
	memset(suppression, 0, len);
	memset(hysteresis, 0, len);
	memset(g_gray, 0, len);
	memset(g_gaussian, 0, len);
	memset(g_sobel, 0, len);
	memset(g_sobel_angle, 0, (len - test[10]) / 3);
	memset(g_suppression, 0, len);
	memset(g_hysteresis, 0, len);

///////////////////////////////Image Read//////////////////////////////
	fread(buf, len, 1, fp); //read into buffer
	len -= 2;
	for (int i = 0; i < test[10]; i++) {
		gray[i] = buf[i];
		gaussian[i] = buf[i];
		sobel[i] = buf[i];
		suppression[i] = buf[i];
		hysteresis[i] = buf[i];
		g_gray[i] = buf[i];
		g_gaussian[i] = buf[i];
		g_sobel[i] = buf[i];
		g_suppression[i] = buf[i];
		g_hysteresis[i] = buf[i];
	}
	for (int i = 18; i < 22; i++)
		width += test[i] * pow(256, i-18);
	for (int i = 22; i < 26; i++)
		height += test[i] * pow(256, i-22);
///////////////////////////////////////////////////////////////////////



////////////////////////////GrayScale(10)//////////////////////////////////
    time = chrono::system_clock::now();
	Grayscale(buf, gray, test[10], len);
    c_end = chrono::system_clock::now() - time;

    time = chrono::system_clock::now();
    GPU_Grayscale(buf, g_gray, test[10], len);
    g_end = chrono::system_clock::now() - time;

    printf("Gray_Scale Time\t\t\t=\tCPU(%lf)  GPU(%lf)",c_end,g_end);
    if(Image_Check(gray,g_gray,len) && c_end > g_end){
        printf("\tGray_Scale + 10(%lf)\n",g_end);
		score += 10;
		total_time += g_end.count();
    }
    else
        printf("\n");

	fwrite(gray, len+2, 1, fp2); //draw image
////////////////////////////Noise_Reduction(10)///////////////////////////////////////

    time = chrono::system_clock::now();
    Noise_Reduction(width,height,gray+test[10], gaussian+test[10]);
    c_end = chrono::system_clock::now() - time;

    time = chrono::system_clock::now();
    GPU_Noise_Reduction(width,height,gray+test[10],g_gaussian+test[10]);
    g_end = chrono::system_clock::now() - time;
 
    printf("Noise_Reduction Time\t\t=\tCPU(%lf)  GPU(%lf)",c_end,g_end);
    if(Image_Check(gaussian,g_gaussian,len) && c_end > g_end){
        printf("\tNoise_Reduction + 10(%lf)\n",g_end);
		score += 10;
		total_time += g_end.count();
    }
    else
        printf("\n");
	fwrite(gaussian, len+2, 1, fp3); //draw image

//////////////////////////Intensity_Gradient(10)////////////////////////////////////
    
    time = chrono::system_clock::now();
	Intensity_Gradient(width, height, gaussian + test[10], sobel + test[10], sobel_angle);
    c_end = chrono::system_clock::now() - time;
    
    time = chrono::system_clock::now();
    GPU_Intensity_Gradient(width,height,gaussian + test[10],g_sobel + test[10],g_sobel_angle);
    g_end = chrono::system_clock::now() - time;
	
    printf("Intensity_Gradient Time\t\t=\tCPU(%lf)  GPU(%lf)",c_end,g_end);
    if(Image_Check(sobel,g_sobel,len) && Image_Check(sobel_angle,g_sobel_angle,(len - test[10]) / 3)&& c_end > g_end){
        printf("\tIntensity_Gradient + 10(%lf)\n",g_end);
		score += 10;
		total_time += g_end.count();
    }
    else
        printf("\n");

    fwrite(sobel, len+2, 1, fp4); //draw image

//////////////////////////Non-maximum_Suppression(10)//////////////////////////////////////
	uint8_t min=255;
	uint8_t max = 0;
	uint8_t g_min = 255;
	uint8_t g_max = 0;
    
    time = chrono::system_clock::now();
	Non_maximum_Suppression(width, height, sobel_angle, sobel + test[10], suppression+test[10],min,max);
    c_end = chrono::system_clock::now() - time;
    
	
    time = chrono::system_clock::now();
    GPU_Non_maximum_Suppression(width,height,sobel_angle,sobel + test[10],g_suppression+test[10],g_min,g_max);
    g_end = chrono::system_clock::now() - time;
	
    printf("Non-maximum_Suppression Time\t=\tCPU(%lf)  GPU(%lf)",c_end,g_end);
    if(Image_Check(suppression,g_suppression,len) && c_end > g_end && g_min == min && g_max == max){
        printf("\tNon-maximum_Suppression + 10(%lf)\n",g_end);
		score += 10;
		total_time += g_end.count();
    }
    else
        printf("\n");

    fwrite(suppression, len+2, 1, fp5); //draw image

//////////////////////////Hysteresis Thresholding(10) ////////////////////////////

    time = chrono::system_clock::now();
	Hysteresis_Thresholding(width, height, suppression + test[10], hysteresis+test[10], min, max);
    c_end = chrono::system_clock::now() - time;
    
    time = chrono::system_clock::now();
    GPU_Hysteresis_Thresholding(width,height,suppression + test[10], g_hysteresis+test[10],min, max);
    g_end = chrono::system_clock::now() - time;
    
    printf("Hysteresis_Thresholding Time\t=\tCPU(%lf)  GPU(%lf)",c_end,g_end);
    if(Image_Check(hysteresis,g_hysteresis,len) && c_end > g_end){
        printf("\tHysteresis Thresholding + 10(%lf)\n",g_end);
		score += 10;
		total_time += g_end.count();
    }
    else
        printf("\n");
	
    fwrite(hysteresis, len+2, 1, fp6); //draw image
		
    printf("total_score is %d\n",score);
    if(score == 50){
      printf("execution time is %lf\n",total_time);
    }
//////////////////////////////////////////////////////////////////
	fclose(fp);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
	fclose(fp5);
	fclose(fp6);
	free(buf);
	free(gray);
	free(gaussian);
	free(sobel);
	free(sobel_angle);
	free(suppression);
	free(hysteresis);
	free(g_gray);
	free(g_gaussian);
	free(g_sobel);
	free(g_sobel_angle);
	free(g_suppression);
	free(g_hysteresis);
	return 0;
}
