using namespace std;
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>
using namespace cv;
enum SYMB
{
	QUAD,
	TRIA,
	ASTER,
};

#define NX 4

void report(string  headerName,vector<string> type_name,vector<float> values)
{
string text = headerName;
int fontFace = CV_FONT_HERSHEY_COMPLEX_SMALL;//FONT_HERSHEY_SCRIPT_SIMPLEX;
double fontScale = 1;
int thickness = 1;
int SIZE_W = 800;
int SIZE_H  = 600;
	
Mat img(SIZE_H, SIZE_W, CV_8UC3, Scalar::all(255));
int baseline=0;
Size textSize = getTextSize(text, fontFace,fontScale, thickness, &baseline);
baseline += thickness;

int OFFS = textSize.height*2;	
float DX = (SIZE_W-2*OFFS)/ NX;

	
line(img, Point(OFFS, SIZE_H-OFFS),Point(OFFS, OFFS),Scalar(0, 0, 0));
	
for(int k=0;k<DX;k++)
{
	float x = DX*k+8;
	float y = 0;
	line(img, Point(OFFS+x, SIZE_H-(OFFS+2+y)),Point(OFFS+x, SIZE_H-(OFFS-2+y)),Scalar(0, 0, 0));
	char stext[0x20];
	sprintf(stext,"%d",k+2);
	putText(img, stext, Point(x+OFFS, SIZE_H-(2+y)), fontFace, fontScale,Scalar(0, 0, 0), thickness, 8);
	//~ line(img, Point(OFFS, SIZE_H-OFFS),Point(OFFS, OFFS),Scalar(0, 0, 0));
}	
line(img, Point(OFFS, SIZE_H-OFFS),Point(SIZE_W-OFFS, SIZE_H-OFFS),Scalar(0, 0, 0));
int  Types     = type_name.size();
int  ValuesIn =  values.size()/Types;

float fm =0;	
for(int t=0;t<Types;t++)
{
	for(int k=0;k<ValuesIn;k++)
	{
		fm = std::max(fm,values[k+t*ValuesIn]);
	}
}

fm = int(fm*20+0.5);
#define NY (fm)
float DY = (SIZE_H-2*OFFS) / NY;


for(int k=0;k<NY;k++)
{
	float y = DY*k;
	float x = 0;
	line(img, Point(OFFS+2+x, SIZE_H-(OFFS+y)),Point(OFFS-2+x, SIZE_H-(OFFS+y)),Scalar(0, 0, 0));
	char stext[0x20];
	sprintf(stext,"%d",k*5);
	putText(img, stext, Point(x, SIZE_H-(OFFS/2+y)), fontFace, fontScale,Scalar(0, 0, 0), thickness, 8);
	//~ line(img, Point(OFFS, SIZE_H-OFFS),Point(OFFS, OFFS),Scalar(0, 0, 0));
}	
	{
Size textSize = getTextSize(headerName, fontFace,fontScale, thickness, &baseline);
putText(img,headerName , Point(SIZE_W/2-textSize.width/2,(1)*OFFS), fontFace, fontScale,Scalar(0, 0, 0), thickness, 8);
	}
Scalar Colors[] = {Scalar(0, 0, 0),Scalar(80, 80, 80),Scalar(160, 160, 160),Scalar(0, 255, 0),Scalar(255, 0, 0),Scalar(0, 0, 255),Scalar(255, 255, 0),Scalar(255, 0, 255),Scalar(0, 196, 200),Scalar(128,0, 64)}; 
	

for(int t=0;t<Types;t++)
{
	float DY =  (SIZE_H-2*OFFS)/fm*20;
	Size textSize = getTextSize(type_name[t], fontFace,fontScale, thickness, &baseline);
	putText(img, type_name[t], Point(SIZE_W-textSize.width,(t+1)*OFFS), fontFace, fontScale,Colors[t], thickness, 8);
	for(int k=0;k<ValuesIn-1;k++)
	{
		float x0 = (k+0)*DX+8;
		float x1 = (k+1)*DX+8;
		float y0 =values[k+t*ValuesIn]*DY;
		float y1 =values[k+1+t*ValuesIn]*DY;
		line(img, Point(OFFS+x0, SIZE_H-(OFFS+y0)),Point(OFFS+x1, SIZE_H-(OFFS+y1)),Colors[t]);
	}
}



// center the text
//~ Point textOrg((img.cols - textSize.width)/2,(img.rows + textSize.height)/2);

// draw the box
//~ rectangle(img, textOrg + Point(0, baseline),textOrg + Point(textSize.width, -textSize.height),Scalar(0,0,255));
// ... and the baseline first
//~ line(img, textOrg + Point(0, thickness),textOrg + Point(textSize.width, thickness),Scalar(0, 0, 255));

// then put the text itself
//~ putText(img, text, textOrg, fontFace, fontScale,Scalar(0, 0, 0), thickness, 8);

vector<int> compression_params;
compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
char filename[0x100];
sprintf(filename,"%s.png",headerName.c_str());
try {
        imwrite(filename, img, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return ;
    }
    
//~ namedWindow("Display Image", WINDOW_AUTOSIZE );
//~ imshow("Display Image", img);
//~ waitKey(0);

}