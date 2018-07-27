#include <stdio.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.hpp>
#include <vector>
using namespace std;
using namespace cv;

typedef unsigned char uchar;

typedef struct _point
{
	int x;
	int y;
} mPoint;

int getMaxIndex(int *arr, int count, bool isMax)
{
	int temp = arr[0];
	int index = 0;
	for (int i = 1; i < count; i++)
	{
		if (isMax)
		{
			if (temp < arr[i])
			{
				temp = arr[i];
				index = i;
			}
		}
		else
		{
			if (temp > arr[i]) {
				temp = arr[i];
			}
		}
	}
	return index;
}

int getMaxCountVal(std::vector<int> vec)
{
	const int arrSize = 1000;
	int statArr[arrSize] = { 0 };
	for (int i = 0; i < vec.size(); ++i)
	{
		statArr[vec[i]]++;
	}

	int refVal = getMaxIndex(statArr, arrSize, true);

	return refVal;
}

uchar** binarizeImg(uchar** ptrGrayImg, unsigned int rows, unsigned int cols, unsigned thr, int maxVal, bool bInverse)
{
	/***Binarize a gray image pointed by ptrGrayImg***
	*@rows:the rows of gray image
	*@cols:the colums of gray image
	*@thr: the threshold used for binarization
	*@maxVal:binary image's max value
	*@bInverse: the flag that indicate whether inverse the foreground and background
	*/
	assert(ptrGrayImg != NULL && rows * cols != 0);
	unsigned int  minVal = 0;
	if (maxVal > 1)
		maxVal = 255;

	if (bInverse)
	{
		unsigned int tmp = minVal;
		minVal = maxVal;
		maxVal = tmp;
	}

	//2D array used to store binarized image
	uchar** p = (uchar**)malloc(rows * sizeof(uchar*));
	assert(p != NULL);

	for (int i = 0; i < rows; ++i)
	{
		p[i] = (uchar*)malloc(cols * sizeof(uchar));
		assert(p[i] != NULL);
	}

	//binarize
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			if (ptrGrayImg[i][j] < thr)
				p[i][j] = minVal;
			else
				p[i][j] = maxVal;
			if (j == 0 || j == 1 || j == cols || j == cols - 1 || j == cols -2)
				p[i][j] = 0;
		}
	}

	return p;
}

uchar** mediaBlur3(uchar** ptrImg, unsigned int rows, unsigned int cols)
{
	/***Blur a gray image pointed by ptrImg by a 3*3 kernel's media value***
	*@ptrImg:a gray image
	*@rows:the rows of gray image
	*@cols:the cols of gray image
	*/
	assert(ptrImg != NULL && rows * cols != 0);

	//2D array used to store blurred image
	uchar** pBlur = (uchar**)malloc(rows * sizeof(uchar*));
	assert(pBlur != NULL);

	for (int i = 0; i < rows; ++i)
	{
		pBlur[i] = (uchar*)malloc(cols * sizeof(uchar));
		assert(pBlur[i] != NULL);
	}
		
	//intialize elements pointed by pBlur
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			pBlur[i][j] = ptrImg[i][j];
		}
	}

	//blur gray image
	for (int i = 1; i < rows - 1; ++i)
	{
		for (int j = 1; j < cols - 1; ++j)
		{
			std::vector<int> pixelVec(9);
			int count = 0;
			for (int m = - 1; m <=  1; ++m)
			{
				for (int n = - 1; n <= 1; ++n)
				{
					pixelVec[count++] = ptrImg[i+m][j+n];
				}
			}

			sort(pixelVec.begin(), pixelVec.end());
			pBlur[i][j] = pixelVec[4];
		}
	}

	return pBlur;
}

uchar** MatToArr(cv::Mat img)
{
	/***convert a Mat to a 2D array***/
	assert(!img.empty());

	uchar **dataPtr = NULL;
	int row = img.rows;
	int col = img.cols;

	dataPtr = (uchar **)malloc(row * sizeof(uchar *));
	assert(dataPtr != NULL);

	for (int i = 0; i < row; i++)
	{
		dataPtr[i] = (uchar *)malloc(col * sizeof(uchar));
		assert(dataPtr[i] != NULL);
	}
		
	//copy pixels's elements
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			dataPtr[i][j] = img.at<uchar>(i, j);
		}
	}

	return dataPtr;
}

uchar** copyImg(uchar **ptrImg, unsigned int rows, unsigned int cols)
{
	/***copy a image pointed by ptrImg***/
	assert(ptrImg != NULL);

	uchar **ptrCopy = (uchar **)malloc(rows * sizeof(uchar *));
	assert(ptrCopy != NULL);

	for (int i = 0; i < rows; ++i)
	{
		ptrCopy[i] = (uchar *)malloc(cols * sizeof(uchar));
		assert(ptrCopy[i] != NULL);
	}

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			ptrCopy[i][j] = ptrImg[i][j];
		}
	}

	return ptrCopy;
}

cv::Mat arryToMat(uchar **ptr, unsigned int rows, unsigned int cols)
{
	assert(ptr != NULL && rows * cols != 0);

	cv::Mat MatImg = Mat(rows, cols, CV_8UC1);
	uchar *pTmp = NULL;

	for (int i = 0; i < rows; ++i)
	{
		pTmp = MatImg.ptr<uchar>(i);
		for (int j = 0; j < cols; ++j)
		{
			pTmp[j] = ptr[i][j];
		}
	}

	return MatImg;
}

void binaryImgNormalize(uchar** ptrImg, unsigned int rows, unsigned cols)
{
	/***nomalize all of the pixel's value ***/
	assert(ptrImg != NULL && rows * cols != 0);
	for (int i = 0; i < rows; ++i)
	{
		uchar* p = ptrImg[i];
		for (int j = 0; j < cols; ++j)
		{
			p[j] /= 255;
		}
	}
}

void binaryImgScale(uchar** ptrImg, unsigned int rows, unsigned cols)
{
	assert(ptrImg != NULL && rows * cols != 0);
	for (int i = 0; i < rows; ++i)
	{
		uchar* p = ptrImg[i];
		for (int j = 0; j < cols; ++j)
		{
			p[j] *= 255;
		}
	}
}



uchar** dilateImg(uchar** ptrImg, unsigned int rows, unsigned int cols, unsigned int width, unsigned int height)
{
	/***dilate a binarized image ***
	*@ptrImg:a pointer to  binarized image 
	*@rows: the rows of image
	*@cols: the colums of image
	*@width: the width of dilate kernel
	*@height: the height of dilate kernel
	*/
	assert(ptrImg != NULL && rows * cols != 0);
	uchar** ptrNewImg = (uchar **)malloc(rows * sizeof(uchar *));
	for (int i = 0; i < rows; ++i)
		ptrNewImg[i] = (uchar *)malloc(cols * sizeof(uchar));


	for (int i = 0; i < rows; ++i)
	{
		uchar* p = ptrNewImg[i];
		for (int j = 0; j < cols; ++j)
		{
			p[j] = 0;
		}
	}

	for (int i = 0; i < rows; ++i)
	{
		uchar* psrc = ptrImg[i];
		uchar* pnew = ptrNewImg[i];
		for (int j = 0; j < cols;++j)
		{
			if (psrc[j] == 255)
			{
				for (int m = 0; m < width; ++m)
				{
					for (int n = 0; n < height; ++n)
					{
						if (i + m < rows && j + n < cols)
						{
							pnew = ptrNewImg[i + m];
							pnew[j + n] = 255;
							
						}				
					}
				}
			}
		}
	}

	return ptrNewImg;
}

int calcThreshold(uchar **ptr, int rows, int cols)
{
	/***compute the threshold value of a gray image use the method of OTSU***/
	assert(ptr != NULL && rows * cols != 0);

	int rowsImg = rows;
	int colsImg = cols;
	double hist[256];
	double var[256];
	int valThresh = 0;
	int numPixels = rowsImg * colsImg;

	for (int i = 0; i < 256; i++)
	{
		hist[i] = 0.0;
		var[i] = 0.0;
	}

	for (int i = 0; i < rowsImg; i++)
	{
		for (int j = 0; j < colsImg; j++)
		{
			unsigned char pixVal = ptr[i][j];
			hist[pixVal]++;
		}
	}

	double probBackground = 0.0;
	double probForeground = 0.0;
	double valBackAveGray = 0.0;
	double valForeAveGray = 0.0;
	double valGlobAveGray = 0.0;
	double data1 = 0.0;
	double data2 = 0.0;

	for (int i = 0; i < 256; i++)
	{
		hist[i] /= numPixels;
		valGlobAveGray += (i * hist[i]);
	}

	for (int i = 0; i < 256; i++)
	{
		probBackground += hist[i];
		probForeground = 1 - probBackground;
		data1 += i * hist[i];
		data2 = valGlobAveGray - data1;
		valBackAveGray = data1 / probBackground;
		valForeAveGray = data2 / probForeground;
		double val = probBackground * probForeground * pow((valBackAveGray - valForeAveGray), 2);
		var[i] = val;
	}

	double tmp = 0.0;
	for (int i = 0; i < 256; i++)
	{
		if (var[i] > tmp)
		{
			tmp = var[i];
			valThresh = i;
		}
	}

	return valThresh;
}

void adaptiveThreshold(Mat &img, double th)
{
	Mat im;
	im = img.clone();
	int i, j, k;
	//行列
	int row = img.rows;
	int col = img.cols;
	//8邻域、均值
	uchar n[9];
	double m;
	//遍历图像
	uchar *p0, *pp, *p1, *p;
	for (i = 1; i < row - 1; ++i)
	{
		//读取图像数据
		p0 = im.ptr<uchar>(i - 1);
		pp = im.ptr<uchar>(i);
		p1 = im.ptr<uchar>(i + 1);
		//修改原图
		p = img.ptr<uchar>(i);
		for (j = 1; j < col - 1; ++j)
		{
			m = 0;
			//3x3 八邻域
			n[0] = p0[j - 1];
			n[1] = p0[j];
			n[2] = p0[j + 1];
			n[3] = pp[j - 1];
			n[4] = pp[j];
			n[5] = pp[j + 1];
			n[6] = p1[j - 1];
			n[7] = p1[j];
			n[8] = p1[j + 1];
			//计算均值
			for (k = 0; k < 9; k++)
			{
				m += n[k];
			}
			m = m / 9 ;

			if (n[4] > m && n[4] > th * 0.8)
				p[j] = 255;
			else
				p[j] = 0;
		}
	}
	im.release();
}

uchar** makeImgThinner(uchar** ptrImg, unsigned int rows, unsigned int cols, const int maxIterations = -1)
{
	/***Extract a binarized image's skeleton***/
	assert(ptrImg != NULL && rows * cols != 0);

	int width = cols;
	int height = rows;
	uchar** ptrCopy = copyImg(ptrImg, rows, cols);
	assert(ptrCopy != NULL);
	int count = 0;

	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations)
			break;

		std::vector<uchar*> pDelPoints;
		for (int i = 0; i < height; ++i)
		{
			uchar* p = ptrCopy[i];
			assert(p != NULL);
			for (int j = 0; j < width; ++j)
			{
				uchar p1 = p[j];
				if (p1 != 1)
					continue;

				uchar p4 = (j == width - 1) ? 0 : ptrCopy[i][j + 1];
				uchar p8 = (j == 0) ? 0 : ptrCopy[i][j + 1];
				uchar p2 = (i == 0) ? 0 : ptrCopy[i - 1][j];
				uchar p3 = (i == 0 || j == width - 1) ? 0 : ptrCopy[i - 1][j + 1];
				uchar p9 = (i == 0 || j == 0) ? 0 : ptrCopy[i - 1][j - 1];
				uchar p6 = (i == height - 1) ? 0 : ptrCopy[i + 1][j];
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : ptrCopy[i + 1][j + 1];
				uchar p7 = (i == height - 1 || j == 0) ? 0 : ptrCopy[i + 1][j - 1];

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						pDelPoints.push_back(p + j);
					}
				}
			}
		}

		for (std::vector<uchar*>::iterator it = pDelPoints.begin(); it != pDelPoints.end(); ++it)
		{
			**it = 0;
		}

		if (pDelPoints.empty())
		{
			break;
		}
		else
		{
			pDelPoints.clear();
		}


		for (int i = 0; i < height; ++i)
		{
			uchar * p = ptrCopy[i];
			for (int j = 0; j < width; ++j)
			{
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : ptrCopy[i][j+1];
				uchar p8 = (j == 0) ? 0 : ptrCopy[i][j + 1];
				uchar p2 = (i == 0) ? 0 : ptrCopy[i-1][j];
				uchar p3 = (i == 0 || j == width - 1) ? 0 : ptrCopy[i - 1][j+1];
				uchar p9 = (i == 0 || j == 0) ? 0 : ptrCopy[i - 1][j - 1];
				uchar p6 = (i == height - 1) ? 0 : ptrCopy[i + 1][j];
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : ptrCopy[i + 1][j + 1];
				uchar p7 = (i == height - 1 || j == 0) ? 0 : ptrCopy[i + 1][j - 1];

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						pDelPoints.push_back(p + j);
					}
				}
			}
		}

		for (std::vector<uchar *>::iterator it = pDelPoints.begin(); it != pDelPoints.end(); ++it)
		{
			**it = 0;
		}

		if (pDelPoints.empty())
		{
			break;
		}
		else
		{
			pDelPoints.clear();
		}
	}

	return ptrCopy;
}


void dilateTest(uchar *imageBuffer, uchar *outBuffer, int imageWidth, int imageHeight)
{
	uchar *dilateBuffer = (uchar *)malloc((imageWidth + 1)*(1 + imageHeight));
	memset(dilateBuffer, 0, (imageHeight + 1)*(imageWidth + 1));

	for (int i = 0; i < imageHeight; i++)
	{
		for (int j = 0; j < imageWidth; j++)
		{
			dilateBuffer[i*(imageWidth + 1) + j + 1] = imageBuffer[i*imageWidth + j];
		}
	}

	uchar *srcImage = dilateBuffer;

	for (int i = 0; i < imageWidth; i++)
	{
		for (int j = 0; j < imageHeight; j++)
		{
			uchar tempNum = 0;

			srcImage = (dilateBuffer + (i*(imageWidth + 1) + j));
			for (int m = 0; m<3; m++)
			{
				for (int n = 0; n < 3; n++)
				{
					if (tempNum < srcImage[n])
					{
						tempNum = srcImage[n];
					}
				}
				srcImage = (srcImage + m*(imageWidth + 1));
			}

			outBuffer[i*imageWidth + j] = tempNum;
		}
	}
}


uchar** preproAndThin(uchar** ptrGray, unsigned int rows, unsigned int cols)
{
	assert(ptrGray != NULL);

	int imgRows = rows;
	int imgCols = cols;

	uchar** pCopyImg = ptrGray;
	uchar** pBlur = mediaBlur3(pCopyImg, imgRows, imgCols);
	int thresh = calcThreshold(pCopyImg, imgRows, imgCols);

	uchar** pBinarize = binarizeImg(pCopyImg, imgRows, imgCols, thresh, 255, 0);
	uchar** pDilate = dilateImg(pBinarize, imgRows, imgCols, 2, 3);
	
	binaryImgNormalize(pBinarize, imgRows, imgCols);

	uchar** ptrThinner = makeImgThinner(pBinarize, imgRows, imgCols);
	binaryImgScale(ptrThinner, imgRows, imgCols);

	//cv::Mat thinnerMat = arryToMat(ptrThinner, imgRows, imgCols);
	 
	/*cv::imshow("thinner", thinnerMat);
	cv::waitKey(0);*/

	free(pDilate);
	free(pBinarize);
	free(pBlur);
	free(pCopyImg);

	return ptrThinner;

}


bool findNextPoint(vector<mPoint> &_neighbor_points, uchar** pImg,
	unsigned int rows, unsigned int cols, mPoint _inpoint, int flag, mPoint& _outpoint, int &_outflag)
{
	assert(pImg != NULL && rows * cols != 0);

	int i = flag;
	int count = 1;
	bool success = false;

	while (count <= 7)
	{
		mPoint tmppoint;
		tmppoint.x = _inpoint.x + _neighbor_points[i].x;
		tmppoint.y = _inpoint.y + _neighbor_points[i].y;
		
		if (tmppoint.x > 0 && tmppoint.y > 0 && tmppoint.x < cols &&tmppoint.y < rows)
		{
			if (pImg[tmppoint.y][tmppoint.x] == 255)
			{
				_outpoint = tmppoint;
				_outflag = i;
				success = true;
				pImg[tmppoint.y][tmppoint.x] = 0;
				break;
			}
		}
		if (count % 2)
		{
			i += count;
			if (i > 7)
			{
				i -= 8;
			}
		}
		else
		{
			i += -count;
			if (i < 0)
			{
				i += 8;
			}
		}
		count++;
	}
	return success;
}

//search the first foreground pixel
bool findFirstPoint(uchar** pImg, unsigned int rows, unsigned int cols, mPoint &outputPoint)
{
	assert(pImg != NULL && rows * cols != 0);

	bool success = false;
	for (int i = 0; i < rows; i++)
	{
		uchar* data = pImg[i];
		for (int j = 0; j < cols; j++)
		{
			if (data[j] == 255)
			{
				success = true;
				outputPoint.x = j;
				outputPoint.y = i;
				data[j] = 0;
				break;
			}
		}
		if (success)
			break;
	}

	return success;
}


//find curve lines
void findLines(uchar** pImg, unsigned int rows, unsigned int cols, vector<deque<mPoint>> &outputLines)
{
	mPoint p1, p2, p3, p4, p5, p6, p7, p8;
	p1.x = -1; p1.y = -1;
	p2.x = 0; p2.y = -1;
	p3.x = 1; p3.y = -1;
	p4.x = 1; p4.y = 0;
	p5.x = 1; p5.y = 1;
	p6.x = 0; p6.y = 1;
	p7.x = -1; p7.y = 1;
	p8.x = -1; p8.y = 0;
	vector<mPoint> neighborPoints = { p1, p2, p3, p4, p5, p6, p7, p8 };

	mPoint first_point;
	while (findFirstPoint(pImg, rows, cols, first_point))
	{
		deque<mPoint> line;
		line.push_back(first_point);
		//the first foreground pixel may lies in the end point or middle position
		mPoint this_point = first_point;
		int this_flag = 0;
		mPoint next_point;
		int next_flag;
		//find in one direction
		while (findNextPoint(neighborPoints, pImg, rows, cols, this_point, this_flag, next_point, next_flag))
		{
			line.push_back(next_point);
			this_point = next_point;
			this_flag = next_flag;
		}
		//search in the other direction
		this_point = first_point;
		this_flag = 0;
		while (findNextPoint(neighborPoints, pImg, rows, cols, this_point, this_flag, next_point, next_flag))
		{
			line.push_front(next_point);
			this_point = next_point;
			this_flag = next_flag;
		}
		if (line.size() > 10)
		{
			outputLines.push_back(line);
		}
	}
}



Scalar random_color(RNG& _rng)
{
	int icolor = (unsigned)_rng;
	return Scalar(icolor & 0xFF, (icolor >> 8) & 0xFF, (icolor >> 16) & 0xFF);
}


deque<mPoint> mergeTwoLines(deque<mPoint> line1, deque<mPoint> line2)
{
	deque<mPoint> line;
	for (int i = 0; i < line2.size(); ++i)
	{
		line.push_back(line2[i]);
	}

	for (int i = 0; i < line1.size(); ++i)
	{
		line.push_back(line1[i]);
	}

	return line;
}

#define SPACE_DELTA  20
#define HIGH_DELTA 1

vector<deque<mPoint>> correctLines(vector<deque<mPoint>> lines, unsigned int cols)
{
	///replace the y_val of points with the most frequently occurring point.y
	std::vector<std::vector<int>> ylines;
	std::vector<int> maxcountVals;
	for (int i = 0; i < lines.size(); ++i)
	{
		std::deque<mPoint> pointQue = lines[i];
		std::vector<int> yVec(pointQue.size());
		for (int j = 0; j < pointQue.size(); ++j)
		{
			yVec[j] = pointQue[j].y;
		}

		ylines.push_back(yVec);

		maxcountVals.push_back(getMaxCountVal(ylines[i]));

	}

	for (int i = 0; i < lines.size(); ++i)
	{
		if (lines[i].size() > 30)
		{
			for (int j = 0; j < lines[i].size(); ++j)
			{
				if ((lines[i][j].y - maxcountVals[i]) <= 1)
				{
					lines[i][j].y = maxcountVals[i];
				}
			}
		}
	}
	///

	vector<deque<mPoint>> correctedLines;
	vector<int> indexVec;

	for (int i = 0; i < lines.size(); ++i) 
	{
		int i_endPoint1y = lines[i].front().y;
		int i_endPoint2y = lines[i].back().y;
		int i_endPoint1x = lines[i].front().x;
		int i_endPoint2x = lines[i].back().x;
		bool flag1, flag2, flag3, flag4;
	
		for (int j = i+1; j < lines.size(); ++j)
		{
			if (j != i)
			{
				int j_endPoint1y = lines[j].front().y;
				int j_endPoint2y = lines[j].back().y;
				int j_endPoint1x = lines[j].front().x;
				int j_endPoint2x = lines[j].back().x;

				flag1 = (abs(i_endPoint1y - j_endPoint1y) <= HIGH_DELTA) && (abs(i_endPoint1x - j_endPoint1x) <= SPACE_DELTA);
				flag2 = (abs(i_endPoint1y - j_endPoint2y) <= HIGH_DELTA) && (abs(i_endPoint1x - j_endPoint2x) <= SPACE_DELTA);
				flag3 = (abs(i_endPoint2y - j_endPoint1y) <= HIGH_DELTA) && (abs(i_endPoint2x - j_endPoint1x) <= SPACE_DELTA);
				flag4 = (abs(i_endPoint2y - j_endPoint2y) <= HIGH_DELTA) && (abs(i_endPoint2x - j_endPoint2x) <= SPACE_DELTA);
				
				if (flag1 || flag2 || flag3 || flag4)
				{
					if (flag1 && (i_endPoint1x > (5 * cols) / 6 || i_endPoint1x < cols / 6))
						continue;
					if (flag2 && (i_endPoint1x >(5 * cols) / 6 || i_endPoint1x < cols / 6))
						continue;
					if (flag3 && (i_endPoint2x >(5 * cols) / 6 || i_endPoint2x < cols / 6))
						continue;
					if (flag4 && (i_endPoint2x >(5 * cols) / 6 || i_endPoint2x < cols / 6))
						continue;
					indexVec.push_back(i);
					indexVec.push_back(j);
				}
			}
		}	

	}

	vector<int> singleIndex;
	for (int i = 0; i < lines.size(); ++i)
	{
		vector<int>::iterator it;
		it = find(indexVec.begin(), indexVec.end(), i);

		if (it != indexVec.end())
			continue;
		singleIndex.push_back(i);
	}

	for (int i = 0; i < singleIndex.size(); ++i)
	{
		correctedLines.push_back(lines[singleIndex[i]]);
	}

	for (int i = 0; i < indexVec.size(); )
	{
		deque<mPoint> mergedLine;
		mergedLine = mergeTwoLines(lines[indexVec[i]], lines[indexVec[i+1]]);
		correctedLines.push_back(mergedLine);
		i += 2;
	}
	
	return correctedLines;
}


void yMonotonous(std::vector<int> yIndexVec, bool& bYInc, bool& bYDec)
{
	assert(!yIndexVec.empty());
	bYInc = false;
	bYDec = false;
	int incCount = 0;
	int decCount = 0;

	for (int i = 0; i < yIndexVec.size()-1; ++i)
	{
		int j = i + 1;
		if (yIndexVec[j] - yIndexVec[i] > 0)
			incCount++;
		if (yIndexVec[j] - yIndexVec[i] < 0)
			decCount++;
	}
	
	if (incCount > 10)
		bYInc = true;
	if (decCount > 10)
		bYDec = true;
}

void xyIndexTrend(deque<mPoint> line, int index, unsigned int pointsNum, bool& bXsame, bool& bYInc, bool& bYDec)
{

	assert(pointsNum > 1);
	int count = 0;
	bYInc = true;
	bYDec = true;

	std::vector<int> yIndexVec;
	std::vector<int> xIndexVec;
	for (int i = 0; i < line.size(); ++i)
	{
		yIndexVec.push_back(line[i].y);
		xIndexVec.push_back(line[i].x);
	}

	/* whether continuous n pionts's x index are same*/
	for (int i = index; i < index + pointsNum - 1; ++i)
	{
		int j = i + 1;
		if (j < line.size() && (xIndexVec[j] - xIndexVec[i] == 0))
		{
			count++;
		}
	}


	bXsame = (count == (pointsNum - 1));

	/*whether y index is increase or decrease*/
	if (index + pointsNum - 1 < line.size())
	{
		int yDiff = yIndexVec[index + pointsNum - 1] - yIndexVec[index];

		if (abs(yDiff) <= 2)
		{
			bYInc = false;
			bYDec = false;
		}
		else
		{
			int Inc = (yDiff >= 0) ? 1 : -1;
			for (int i = index; i < index + pointsNum; ++i)
			{
				int j = i + 1;
				if (j < line.size() && Inc == 1)
				{
					if (yIndexVec[j] - yIndexVec[i] < 0)
					{
						bYInc = false;
						break;
					}
				}
				else if (j < line.size() && Inc == -1)
				{
					if (yIndexVec[j] - yIndexVec[i] > 0)
					{
						bYDec = false;
						break;
					}
				}

			}
		}
	}
	
	
}


double calcSlop(mPoint point1, mPoint point2)
{
	double deltaX = point2.x - point1.x;
	double deltaY = point2.y - point1.y;

	double slop;
	if (deltaX == 0)
		slop = 1000000;
	else
		slop = deltaY / deltaX;

	return slop;
}

bool lineMonotonous(deque<mPoint> line, std::vector<double>& slopsVec)
{
	/*judge a line's trend: make sure the line keep increase and 
	* decrease in the either endpoint position and keep flat in 
	* the middle position
	*/

	if (line.size() < 20)
		return false;

	const unsigned segNum = 8;
	unsigned nLen = line.size();
	unsigned nSegLen = nLen / segNum;
	int posCnt = 0;
	int negCnt = 0;
	int fltCnt = 0;
	bool bInc = false;
	bool bDec = false;
	bool bFlt = false;
	bool bEndHor = false;

	//the sample points index
	mPoint arr[segNum + 1] = { 0 };
	for (int i = 0; i < segNum + 1; ++i)
	{
		int index = i * nSegLen;
		while (index >= nLen)
		{
			--index;
		}		
		arr[i] = line[index];
	}
	
	//the slop between sample points
	//std::vector<double> slopsVec;
	for (int i = 0; i < segNum; ++i)
	{
		if (i < 4)
		{
			slopsVec.push_back(calcSlop(arr[0], arr[i + 1]));
		}
		else
		{
			slopsVec.push_back(calcSlop(arr[i], arr[segNum]));
		}
	}
	slopsVec.push_back(calcSlop(arr[3], arr[4]));
	slopsVec.push_back(calcSlop(arr[4], arr[5]));
	slopsVec.push_back(calcSlop(arr[3], arr[5]));
	slopsVec.push_back(calcSlop(arr[0], arr[8]));
	
	//nomalize slop vector
	std::vector<double> tmp;
	tmp = slopsVec;
	for (int i = 0; i < tmp.size(); ++i)
	{
		tmp[i] = abs(tmp[i]);
	}

	double maxVal = *(std::max_element(tmp.begin(), tmp.end()));
	for (int i = 0; i < slopsVec.size(); ++i)
		slopsVec[i] /= maxVal;

	//make a decision according to slop's value
	for (int i = 0; i < slopsVec.size(); ++i)
	{

		if (abs(slopsVec[i]) > 0.1 && slopsVec[i] > 0)
			posCnt++;
		else if (abs(slopsVec[i]) > 0.1 && slopsVec[i] < 0)
			negCnt++;
		else
			fltCnt++;
	}

	bInc = posCnt >= 3 ? true : false;
	bDec = negCnt >= 3 ? true : false;
	bFlt = fltCnt >= 2 ? true : false;

	if (abs(slopsVec[slopsVec.size() - 1]) < 0.1)
		bEndHor = true;

	bool bRes = bInc && bDec && bFlt && bEndHor;
	if (!bRes)
		slopsVec.clear();

	return (bRes);
}


bool isDefectLine(deque<mPoint> line, std::vector<double>& lineSlop)
{
	//int ascent = 0;
	//int descend = 0;
	//int flatLand = 0;
	//bool bShort = false;
	//bool bRes = false;

	//std::vector<int> yIndexVec;
	//std::vector<int> xIndexVec;
	//for (int i = 0; i < line.size(); ++i)
	//{
	//	yIndexVec.push_back(line[i].y);
	//	xIndexVec.push_back(line[i].x);
	//}

	//unsigned int numEle = yIndexVec.size();
	//if (numEle < 20)
	//{
	//	bShort = true;
	//}


	//for (int i = 0; i < numEle; ++i)
	//{
	//	if (i + 2 < numEle)
	//	{
	//		/*ascent、descend and flatten judgement*/
	//		if (yIndexVec[i + 1] > yIndexVec[i] && yIndexVec[i + 2] > yIndexVec[i + 1])
	//		{
	//			ascent++;//continue n points's y index increase
	//		}
	//		else if (yIndexVec[i + 1] < yIndexVec[i] && yIndexVec[i + 2] < yIndexVec[i + 1])
	//		{
	//			descend++;//continue n point's y index decrease
	//		}
	//		else if ((yIndexVec[i + 1] - yIndexVec[i]) * (yIndexVec[i + 2] - yIndexVec[i + 1]) <= 0)
	//		{
	//			flatLand++;
	//		}


	//		/* monotonous judgement */
	//		int diff1 = xIndexVec[i + 1] - xIndexVec[i];
	//		int diff2 = xIndexVec[i + 2] - xIndexVec[i + 1];
	//		

	//		if (diff1 * diff2 < 0)
	//		{
	//			//revesre or loop
	//			bMono = false;
	//			cout << "reverse loop" << "i=" << i << endl;
	//			break;
	//		}
	//		else if (diff1 * diff2 == 0)
	//		{
	//			//go head to the y direction:continuous five points's x index are same
	//			if ((i + 3) < numEle && diff1 == 0 && diff2 == 0 && (xIndexVec[i + 3] - xIndexVec[i + 2]) == 0)//1 "I" shape
	//			{
	//				bMono = false;
	//				cout << "1 " << "i=" << i << endl;
	//				break;
	//			}
	//			else if (diff1 != 0 && diff2 == 0)//2 "L"shape   
	//			{
	//				int j = i + 2;
	//				if (j + 1 < numEle && xIndexVec[j + 1] - xIndexVec[j] == 0)
	//				{
	//					bMono = false;
	//					cout << "2," << "i=" << i << endl;
	//					break;
	//				}
	//			}
	//			else if(diff1 == 0 && diff2 != 0)//3 "L" shape
	//			{
	//				int j = i + 2;
	//				if (j +2 < numEle)
	//				{
	//					int delta1 = xIndexVec[j + 1] - xIndexVec[j];
	//					int delta2 = xIndexVec[j + 2] - xIndexVec[j + 1];
	//					if (delta1 == 0 && delta2 == 0)
	//					{
	//						bMono = false;
	//						cout << "3," << "i=" << i << endl;
	//						break;
	//					}
	//				}
	//			}
	//		}
	//	}

	//}


	bool bMono = !lineMonotonous(line, lineSlop);

	return (bMono);
}

std::vector<mPoint> seekSuspectDefectPoints(std::vector<deque<mPoint> > lines)
{
	int lineNums = lines.size();
	std::vector<int> lineLens;

	for (int i = 0; i < lineNums; ++i)
	{
		lineLens.push_back(lines[i].size());
	}

	sort(lineLens.begin(), lineLens.end());

	int maxLen = lineLens[lineNums - 1];
	mPoint p1, p2;
	for (int i = 0; i < lines.size(); ++i)
	{
		if (lines[i].size() == maxLen)
		{
			p1 = lines[i].front();
			p2 = lines[i].back();
			break;
		}
	}

	if (p1.x > p2.x)
	{
		mPoint pTmp = p1;
		p1 = p2;
		p2 = pTmp;
	}

	std::vector<mPoint> suspectPoints;
	
	for (int i = 0; i < lines.size(); ++i)
	{
		mPoint point1, point2;
		point1 = lines[i].front();
		point2 = lines[i].back();

		if (point1.y < point2.y)
		{
			mPoint pTmp = point1;
			point1 = point2;
			point2 = pTmp;
		}
		

		if (lines[i].size() < 0.75 * maxLen && (point2.x > p1.x && point2.x < p2.x))
			suspectPoints.push_back(point2);
	}

	return suspectPoints;
	
}

std::vector<int> suspectLines(std::vector<deque<mPoint>> lines)
{
	std::vector<int> suspectIndexs;
	std::vector<vector<double> > linesSlop;//regular line's slops vector(non-empty value)
	
	for (int i = 0; i < lines.size(); ++i)
	{
		std::vector<double> lineSlop;
		if (isDefectLine(lines[i], lineSlop))
			suspectIndexs.push_back(i);

		linesSlop.push_back(lineSlop);
	}

	return suspectIndexs;
}


std::vector<vector<cv::Point>> toCvPoints(std::vector<deque<mPoint>> lines)
{
	std::vector<vector<cv::Point>> cvlinesVec;
	for (int i = 0; i < lines.size(); ++i)
	{
		std::vector<cv::Point> points;
		for (int j = 0; j < lines[i].size(); ++j)
		{
			cv::Point p;
			p.x = lines[i][j].x;
			p.y = lines[i][j].y;
			points.push_back(p);
		}

		cvlinesVec.push_back(points);

	}

	return cvlinesVec;
}

int pointsDis(mPoint point1, mPoint point2)
{
	return (std::sqrt(point1.x * point2.x + point1.y * point2.y));
}

bool twoLinesDis(std::deque<mPoint> line1, std::deque<mPoint> line2, std::vector<pair<int,int>>& crossPointPairs)
{
	bool bCross = false;
	for (int i = 0; i < line1.size(); ++i)
	{
		for (int j = 0; j < line2.size(); ++j)
		{
			int pointsDis = std::sqrt(line1[i].x *line2[j].x + line1[i].y * line2[j].y);
			if (pointsDis < 3)
			{
				bCross = true;
				std::pair<int, int> pointPair;
				pointPair.first = i;
				pointPair.second = j;
				crossPointPairs.push_back(pointPair);
			}
		}
	}

	return bCross;
}

bool detectCrossLines(vector<deque<mPoint>> lines, std::vector<pair<int, int>>& crossLinePairs,
	std::vector<vector<pair<int, int>>>& pointPairsVec)
{
	/*judge whether there are cross or nearly cross lines*/
	bool bIsCross = false;
	unsigned int numLines = lines.size();

	for (int i = 0; i < numLines - 1; ++i)
	{
		for (int j = i + 1; j < numLines; ++j)
		{
			std::vector<pair<int, int>> pointPairs;
			if (twoLinesDis(lines[i], lines[j], pointPairs))
			{
				cout << "cross" << endl;
				bIsCross = true;
				std::pair<int, int> linePair;
				linePair.first = i;
				linePair.second = j;
				crossLinePairs.push_back(linePair);
				pointPairsVec.push_back(pointPairs);
			}

		}
		
	}

	return bIsCross;
}

int main()
{
	/*cv::Mat img = cv::imread("E:\\pictures\\coil\\coil.bmp", 0);
	cv::Mat imgcolor = cv::imread("E:\\pictures\\coil\\coil.bmp");*/

	cv::Mat img = cv::imread("E:\\pictures\\coil\\roi\\26-6.bmp", 0);
	cv::Mat imgcolor = cv::imread("E:\\pictures\\coil\\roi\\26-6.bmp");// draw result on imgcolor

	int row = img.rows;
	int col = img.cols;

	/************************************************************/
	uchar **ptr =  MatToArr(img);
	uchar **ptrcopy = copyImg(ptr, row, col);
	uchar **pThin = preproAndThin(ptrcopy, row, col);
	cv::Mat thinMat = arryToMat(pThin, row, col);

	vector<deque<mPoint>> lines;
	findLines(pThin, row, col, lines);
	vector<deque<mPoint>> newLines = correctLines(lines, col);

	std::vector<mPoint> suspPoints = seekSuspectDefectPoints(lines);
	std::vector<int> defectLinesIndex = suspectLines(lines);


	std::vector<pair<int, int>> crossLinePairs;
	std::vector<vector<pair<int, int>>> crossPointPairsVec;
	//bool bCross = detectCrossLines(lines, crossLinePairs, crossPointPairsVec);

	/***********************************************************/


//test code begin
	 //code of test suspect lines
	std::vector<std::deque<mPoint>> indexedLines;
	for (int i = 0; i < defectLinesIndex.size(); ++i)
	{
		indexedLines.push_back(lines[defectLinesIndex[i]]);
	}
	std::vector<vector<cv::Point>> cvlinesVec = toCvPoints(indexedLines);
	for (int i = 0; i < cvlinesVec.size(); ++i)
	{
		for (int j = 0; j < cvlinesVec[i].size(); ++j)
		{
			cv::circle(imgcolor, cvlinesVec[i][j], 1, cv::Scalar(0, 0, 255), 1);
		}
	}

	

	/****************************************************************************/
	
	//std::vector<vector<cv::Point>> linesVec;
	//for (int i = 0; i < lines.size(); ++i)
	//{
	//	std::vector<cv::Point> points;
	//	for (int j = 0; j < lines[i].size(); ++j)
	//	{
	//		cv::Point p;
	//		p.x = lines[i][j].x;
	//		p.y = lines[i][j].y;
	//		points.push_back(p);
	//	}

	//	linesVec.push_back(points);

	//}


	////draw lines
	//Mat draw_img = img.clone();
	//RNG rng(123);
	//Scalar color;
	//for (int i = 0; i < lines.size(); i++)
	//{
	//	color = random_color(rng);
	//	for (int j = 0; j < lines[i].size(); j++)
	//	{
	//		imgcolor.at<Vec3b>(linesVec[i][j]) = Vec3b(color[0], color[1], color[2]);
	//	}
	//}


	//draw suspect points
	std::vector<cv::Point> endPoints;
	for (int i = 0; i < suspPoints.size(); ++i)
	{
		cv::Point p;
		p.x = suspPoints[i].x;
		p.y = suspPoints[i].y;
		endPoints.push_back(p);
	}

	for (int i = 0; i < endPoints.size(); ++i)
	{
		cv::circle(imgcolor, endPoints[i], 1, cv::Scalar(0, 255, 0), 2);
	}
	/*******************************************************************************/
	
	cv::imshow("thin", thinMat*255);

	imshow("draw_img", imgcolor);
	cv::imwrite("E:\\pictures\\coil\\fitcoillines.bmp", imgcolor);
	
	//cv::imshow("src", img);
	cv::waitKey(0);

//test code end
	free(ptr);
	free(pThin);

	return 0;
}