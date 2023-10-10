#include "Func.h"
#define BLOCKSIZE 1024
/////////////////////////////////////////////////////////////////////////
// 1. �Լ��� Colab ȯ�濡�� �����ؾ� �մϴ�.
// 2. �����Ӱ� �����ϼŵ� ������ ��� �Լ����� GPU�� Ȱ���ؾ� �մϴ�.
// 3. CPU_Func.cu�� �ִ� Image_Check�Լ����� True�� Return�Ǿ�� �ϸ�, CPU�ڵ忡 ���� �ӵ��� ����� �մϴ�.
/////////////////////////////////////////////////////////////////////////


__global__ void Grayscale_DEV(uint8_t* buf_dev, uint8_t* gray_dev, uint8_t start_add_dev, int len) //grayscale�� kernel �Լ�
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + start_add_dev;//thread�� �����ּ�
	if (i > len) return;//len �̻��� ��� Ż��
	if (i % 3 != 0) return;//i�� 3�� ����϶��� ����

	//CPU�� ���� ����
	int tmp = buf_dev[i] * 0.114 + buf_dev[i + 1] * 0.587 + buf_dev[i + 2] * 0.299;
	gray_dev[i] = tmp;
	gray_dev[i + 1] = tmp;
	gray_dev[i + 2] = tmp;
}
void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len)
{
	uint8_t* bufDEV = NULL;//buf ����� dev ������
	uint8_t* grayDEV = NULL;//gray ����� dev ������

	//�޸� �Ҵ� �� ������ ����
	cudaMalloc((void**)&bufDEV, len + 2);
	cudaMalloc((void**)&grayDEV, len);
	cudaMemcpy(bufDEV, buf, len + 2, cudaMemcpyHostToDevice);
	cudaMemcpy(grayDEV, gray, len, cudaMemcpyHostToDevice);

	//GPU grid �� thread size ����
	dim3 dimGrid(len / BLOCKSIZE + 1, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	Grayscale_DEV << <dimGrid, dimBlock >> > (bufDEV, grayDEV, start_add, len);//Kernel �Լ� ����
	cudaMemcpy(gray, grayDEV, len, cudaMemcpyDeviceToHost);//gray�� gpu �޸� ������ ����

	//�޸� �Ҵ� ����
	cudaFree(bufDEV);
	cudaFree(grayDEV);
}


__global__ void zero_padding(int width, uint8_t* tmp, uint8_t* gray)//���� �е� gpu Ŀ���Լ�
{
	int i = blockIdx.x + 2;
	int j = threadIdx.x + 2;

	//cpu�� ������ ����
	tmp[i * (width + 4) + j] = gray[((i - 2) * width + (j - 2)) * 3];
}

__global__ void gaussian_blur_DEV(float* filter, uint8_t* tmp, int height, int width, uint8_t* gaussian)
{//����þ� ���� ���� gpu Ŀ���Լ�
	int y = blockIdx.x;
	int x = threadIdx.x;

	//cpu�� ������ ����
	float v = 0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			v += tmp[(y + i) * width + x + j] * filter[i * 5 + j];
		}
	}

	gaussian[(y * (width - 4) + x) * 3] = v;
	gaussian[(y * (width - 4) + x) * 3 + 1] = v;
	gaussian[(y * (width - 4) + x) * 3 + 2] = v;

}

void GPU_Noise_Reduction(int width, int height, uint8_t* gray, uint8_t* gaussian)
{
	//����þ� ����
	float filter[25] = { 0 };
	float sigma = 1.0;
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			filter[(i + 2) * 5 + j + 2]
				= (1 / (2 * 3.14 * sigma * sigma)) * exp(-(i * i + j * j) / (2 * sigma * sigma));
		}
	}

	//����þ� ���� gpu �޸𸮷� ����
	float* filterDEV = NULL;
	cudaMalloc((void**)&filterDEV, 25 * sizeof(float));
	cudaMemcpy(filterDEV, filter, 25 * sizeof(float), cudaMemcpyHostToDevice);

	//���� �е��� ������ tmp �޸�
	uint8_t* tmp = (uint8_t*)malloc((width + 4) * (height + 4));
	memset(tmp, (uint8_t)0, (width + 4) * (height + 4));

	//gpu �޸� �ּ� �J�� �� ������ ����
	uint8_t* tmpDEV = NULL;
	uint8_t* grayDEV = NULL;
	uint8_t* gaussianDEV = NULL;
	cudaMalloc((void**)&tmpDEV, (width + 4) * (height + 4));
	cudaMalloc((void**)&grayDEV, 2400052);
	cudaMalloc((void**)&gaussianDEV, 2400052);

	cudaMemcpy(tmpDEV, tmp, (width + 4) * (height + 4), cudaMemcpyHostToDevice);
	cudaMemcpy(grayDEV, gray, 2400052, cudaMemcpyHostToDevice);
	cudaMemcpy(gaussianDEV, gray, 2400052, cudaMemcpyHostToDevice);
	zero_padding << <800, 1000 >> > (width, tmpDEV, grayDEV);//���� �е�

	gaussian_blur_DEV << <800, 1000 >> > (filterDEV, tmpDEV, height, width + 4, gaussianDEV);//����þ� ���� ����
	cudaMemcpy(gaussian, gaussianDEV, 2400052, cudaMemcpyDeviceToHost);//��� ������ CPU �̵�
	//�޸� �Ҵ�����
	cudaFree(tmpDEV);
	cudaFree(grayDEV);
	cudaFree(gaussianDEV);
	free(tmp);
}

__global__ void zero_padding1(int width, uint8_t* tmp, uint8_t* gaussian)//sobel ���� ������ ���� �����е�
{
	int i = blockIdx.x + 1;
	int j = threadIdx.x + 1;
	//cpu ����� ����
	tmp[i * (width + 2) + j] = gaussian[((i - 1) * width + (j - 1)) * 3];
}

__global__ void get_gradient(int* filter_x, int* filter_y, uint8_t* tmp, int* gx, int* gy, int width)
{//��pixel������ gradient�� ���ϴ� Ŀ���Լ�
	int i = blockIdx.x;
	int j = threadIdx.x;

	//cpu�� ������ ����
	int sum_x = 0;
	int sum_y = 0;
	for (int n = 0; n < 3; n++)
	{
		for (int m = 0; m < 3; m++)
		{
			sum_y += (int)tmp[(i + n) * width + j + m] * filter_y[n * 3 + m];
			sum_x += (int)tmp[(i + n) * width + j + m] * filter_x[n * 3 + m];
		}
	}
	gx[i * (width - 2) + j] = sum_x;
	gy[i * (width - 2) + j] = sum_y;
}
__global__ void sobel_operation(uint8_t* sobel, int* v, int width)//����� gradient�� sobel �迭�� �����ϴ� Ŀ���Լ�
{
	int i = blockIdx.x;
	int j = threadIdx.x;

	sobel[(i * width + j) * 3] = v[i * width + j];
	sobel[(i * width + j) * 3 + 1] = v[i * width + j];
	sobel[(i * width + j) * 3 + 2] = v[i * width + j];

}
__global__ void get_angle(uint8_t* angle, float* t_angle, int width)
{//�� �ȼ����� ���� ���� ������ ����ȭ �ϴ� Ŀ���Լ�
	int i = blockIdx.x;
	int j = threadIdx.x;
	//cpu ����� ����
	if ((t_angle[i * width + j] > -22.5 && t_angle[i * width + j] <= 22.5) || (t_angle[i * width + j] > 157.5 || t_angle[i * width + j] <= -157.5))
		angle[i * width + j] = 0;
	else if ((t_angle[i * width + j] > 22.5 && t_angle[i * width + j] <= 67.5) || (t_angle[i * width + j] > -157.5 && t_angle[i * width + j] <= -112.5))
		angle[i * width + j] = 45;
	else if ((t_angle[i * width + j] > 67.5 && t_angle[i * width + j] <= 112.5) || (t_angle[i * width + j] > -112.5 && t_angle[i * width + j] <= -67.5))
		angle[i * width + j] = 90;
	else if ((t_angle[i * width + j] > 112.5 && t_angle[i * width + j] <= 157.5) || (t_angle[i * width + j] > -67.5 && t_angle[i * width + j] <= -22.5))
		angle[i * width + j] = 135;
}

void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t* angle)
{
	//sobel ����
	int filter_x[9] = { -1,0,1
						   ,-2,0,2
						   ,-1,0,1 };
	int filter_y[9] = { 1,2,1
						,0,0,0
						,-1,-2,-1 };

	//�޸� �Ҵ� �� ������ ����
	uint8_t* tmp = (uint8_t*)malloc((width + 2) * (height + 2));
	memset(tmp, (uint8_t)0, (width + 2) * (height + 2));
	uint8_t* tmpDEV = NULL;
	uint8_t* gaussianDEV = NULL;
	uint8_t* sobelDEV = NULL;//sobel�� gpu �ּ�
	uint8_t* angleDEV = NULL;//angle�� gpu �ּ�
	int* fxDEV = NULL;//filter_x�� gpu �ּ�
	int* fyDEV = NULL;//filter_y�� gpu �ּ�
	cudaMalloc((void**)&tmpDEV, (width + 2) * (height + 2));
	cudaMalloc((void**)&gaussianDEV, 2400052);
	cudaMalloc((void**)&sobelDEV, 2400052);
	cudaMalloc((void**)&angleDEV, width * height);
	cudaMalloc((void**)&fxDEV, 9 * sizeof(int));
	cudaMalloc((void**)&fyDEV, 9 * sizeof(int));
	cudaMemcpy(tmpDEV, tmp, (width + 2) * (height + 2), cudaMemcpyHostToDevice);
	cudaMemcpy(gaussianDEV, gaussian, 2400052, cudaMemcpyHostToDevice);

	zero_padding1 << <800, 1000 >> > (width, tmpDEV, gaussianDEV);//����þ� �̹��� ������ ���� �е�

	int* gx = (int*)malloc(height * width * sizeof(int));//x�� ����
	int* gy = (int*)malloc(height * width * sizeof(int));//y�� ����


	int* v = (int*)malloc(height * width * sizeof(int)); //���� ����
	float* t_angle = (float*)malloc(height * width * sizeof(float));//����
	//�� �������� gpu �ּ� �Ҵ�
	int* gxDEV = NULL;
	int* gyDEV = NULL;
	int* vDEV = NULL;
	float* t_angleDEV = NULL;

	//cuda �޸� �Ҵ�
	cudaMalloc((void**)&gxDEV, height * width * sizeof(int));
	cudaMalloc((void**)&gyDEV, height * width * sizeof(int));
	cudaMalloc((void**)&vDEV, height * width * sizeof(int));
	cudaMalloc((void**)&t_angleDEV, height * width * sizeof(float));

	//������ ����
	cudaMemcpy(fxDEV, filter_x, 9 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(fyDEV, filter_y, 9 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gxDEV, gx, height * width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gyDEV, gy, height * width * sizeof(int), cudaMemcpyHostToDevice);

	get_gradient << <800, 1000 >> > (fxDEV, fyDEV, tmpDEV, gxDEV, gyDEV, width + 2);  //gx gy ���ϱ�
	cudaMemcpy(gx, gxDEV, height * width * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(gy, gyDEV, height * width * sizeof(int), cudaMemcpyDeviceToHost);

	//gradient


	for (int i = 0; i < height; i++)//gx�� gy�� gradient�� angle ������ ���ϴ� ����
	{
		for (int j = 0; j < width; j++) {
			t_angle[i * width + j] = 0;
			v[i * width + j] = sqrt(gx[i * width + j] * gx[i * width + j] + gy[i * width + j] * gy[i * width + j]);
			if (v[i * width + j] > 255) v[i * width + j] = 255;
			t_angle[i * width + j] = (float)atan2(gy[i * width + j], gx[i * width + j]) * 180.0 / 3.14;
		}
	}

	cudaMemcpy(vDEV, v, height * width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(t_angleDEV, t_angle, height * width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sobelDEV, sobel, 2400052, cudaMemcpyHostToDevice);
	cudaMemcpy(angleDEV, angle, height * width * sizeof(int), cudaMemcpyHostToDevice);
	sobel_operation << <800, 1000 >> > (sobelDEV, vDEV, width);//sobel ������ ���ϴ� ����
	get_angle << <800, 1000 >> > (angleDEV, t_angleDEV, width);//���� angle �����͸� ����ȭ�ϴ� ����
	cudaMemcpy(sobel, sobelDEV, 2400052, cudaMemcpyDeviceToHost);
	cudaMemcpy(angle, angleDEV, height * width, cudaMemcpyDeviceToHost);

	//�޸� �Ҵ� ����
	cudaFree(gxDEV);
	cudaFree(gyDEV);
	cudaFree(vDEV);
	cudaFree(t_angleDEV);
	cudaFree(tmpDEV);
	cudaFree(gaussianDEV);
	cudaFree(sobelDEV);
	cudaFree(angleDEV);
	cudaFree(fxDEV);
	cudaFree(fyDEV);
	free(gx);
	free(gy);
	free(v);
	free(t_angle);
}

__global__ void NMS(int width, int height, uint8_t* angle, uint8_t* sobel, uint8_t* suppression_pixel, uint8_t* min, uint8_t* max)
{/*
	int i = blockIdx.x + 1;
	int j = threadIdx.x + 1;
	int WIDTH = width;


	uint8_t ANGLE = angle[i * width + j];

	uint8_t p1 = 0;
	uint8_t p2 = 0;

	if (ANGLE == 0) {
		p1 = sobel[((i + 1) * WIDTH + j) * 3];
		p2 = sobel[((i - 1) * WIDTH + j) * 3];
	}
	else if (ANGLE == 45) {
		p1 = sobel[((i + 1) * WIDTH + j - 1) * 3];
		p2 = sobel[((i - 1) * WIDTH + j + 1) * 3];
	}
	else if (ANGLE == 90) {
		p1 = sobel[((i)*WIDTH + j + 1) * 3];
		p2 = sobel[((i)*WIDTH + j - 1) * 3];
	}
	else {
		p1 = sobel[((i + 1) * WIDTH + j + 1) * 3];
		p2 = sobel[((i - 1) * WIDTH + j - 1) * 3];
	}

	uint8_t v = sobel[(i * WIDTH + j) * 3];

	if (min[0] > v)
		min[0] = v;
	if (max[0] < v)
		max[0] = v;
	if ((v >= p1) && (v >= p2)) {
		suppression_pixel[(i * width + j) * 3] = v;
		suppression_pixel[(i * width + j) * 3 + 1] = v;
		suppression_pixel[(i * width + j) * 3 + 2] = v;
	}
	else {
		suppression_pixel[(i * width + j) * 3] = 0;
		suppression_pixel[(i * width + j) * 3 + 1] = 0;
		suppression_pixel[(i * width + j) * 3 + 2] = 0;
	}*/

}

void GPU_Non_maximum_Suppression(int width, int height, uint8_t* angle, uint8_t* sobel, uint8_t* suppression_pixel, uint8_t* min, uint8_t* max)
{
	/*
	uint8_t* angleDEV = NULL;
	uint8_t* sobelDEV = NULL;
	uint8_t* suppression_pixelDEV = NULL;
	uint8_t MIN[1] = { 255 };
	uint8_t MAX[1] = { 0 };
	uint8_t* MINDEV = NULL, * MAXDEV = NULL;

	cudaMalloc((void**)&angleDEV, width * height);
	cudaMalloc((void**)&sobelDEV, width * height);
	cudaMalloc((void**)&suppression_pixelDEV, 2400052);
	cudaMalloc((void**)&MINDEV, 1);
	cudaMalloc((void**)&MAXDEV, 1);

	cudaMemcpy(angleDEV, angle, width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(sobelDEV, sobel, width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(suppression_pixelDEV, suppression_pixel, width * height * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(MINDEV, MIN, 1, cudaMemcpyHostToDevice);
	cudaMemcpy(MAXDEV, MAX, 1, cudaMemcpyHostToDevice);


	//NMS << <799, 999 >> > (width, height, angleDEV, sobelDEV, suppression_pixelDEV, MINDEV, MAXDEV);

	cudaMemcpy(suppression_pixel, suppression_pixelDEV, width * height * 3, cudaMemcpyDeviceToHost);
	*/
}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t* suppression_pixel, uint8_t* hysteresis, uint8_t min, uint8_t max) {}