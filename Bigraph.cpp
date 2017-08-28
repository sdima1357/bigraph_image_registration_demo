/*
        Samsonov Dima Israel 8/2017    sdima1357@gmail.com
	
	
	ported to openCV env from my old vi3dim (vi3dim.com) image registration code, toward open source...
	
	bigraph vertices registration demo code
	
	
*/
#include <stdio.h>
#include <iomanip>
#include <iostream>

#include "options.h"
#include <opencv2/opencv.hpp>
using namespace cv;


#define PRINT(x) std::cout<<""<<std::setw(15)<<(x)<<" " #x "\n"

#define     MAX_LEVEL 8

#define     NUM_EDGES_AROUND_VERTEX 10	

#define     LEDGE_SX 4
#define     LEDGE_SY 3
#define     LEDGE_SIZE (LEDGE_SX*LEDGE_SY)

// LUT_BITS 13-17 
#define     LUT_BITS 16
#define     LUT_SIZE (1<<LUT_BITS)

// NUM_VOTES_OF_EDGE 1-5
#define     NUM_VOTES_OF_EDGE           1							

class HiCl
{
	public:
		 int64 e1;
	string Name;
	HiCl(const char*name):Name(name)
	{

	    e1 = cv::getTickCount();
	    
	}
	~HiCl()
	{
		    int64 e2 = cv::getTickCount();
		    double t = 1000*double(e2 - e1)/getTickFrequency();
		    std::cout<<Name<<" "<<t<<"ms\n";		
	}
};

inline uint32_t hashLUT(const uint32_t inp)
{
	return (inp*7+(inp>>LUT_BITS)*13)&(LUT_SIZE-1);
}

inline uint32_t cntBits(uint32_t x)
 {
   x  = x - ((x >> 1u) & 0x55555555u);
   x  = (x & 0x33333333u) + ((x >> 2u) & 0x33333333u);
   x  = x + (x >> 4u);
   x &= 0xF0F0F0Fu;
   return (x * 0x01010101u) >> 24u;
}

void boxFilter3(Mat& image) 
{
	const int X = image.size().width;
	const int Y = image.size().height;
	
	const float fact = 1.0f/9.0f;
	const int AX = (X+63)&~63;
	float * __restrict__ tb = new float[AX*3];
	memset(tb,0,AX*3*sizeof(float));
	float * __restrict__  t0 = tb;
	float * __restrict__ t1 = t0+AX;
	float * __restrict__ t2 = t1+AX;
	
	for(int y=0;y<Y;y++)
	{
		float * __restrict__ pimage = image.ptr<float>(y);
		
		t2[0]= pimage[0]*2+pimage[1];
		for(int x = 1;x<X-1;x++)
		{
			t2[x]  = (pimage[x-1]+pimage[x]+pimage[x+1]);
		}
		t2[X-1]= pimage[X-1]*2+pimage[X-2];
		
		if(y==1) 
		{
			float *__restrict__  pimageA =image.ptr<float>(0);
			for(int x = 0;x<X;x++)   
			{
				pimageA[x] = (t2[x]+t1[x]+t1[x])*fact;
			}
		}
		else if(y>1) 
		{
			float *__restrict__  pimageA =image.ptr<float>(y-1);
			for(int x = 0;x<X;x++)   
			{
				pimageA[x] = (t0[x]+t1[x]+t2[x])*fact;
			}
		}	
		std::swap(t0,t1);
		std::swap(t1,t2);
	}
	{
		float * pimageA =image.ptr<float>(Y-1);
		for(int x = 0;x<X;x++)
		{
			pimageA[x] = (t1[x]+t1[x]+t0[x])*fact;
		}
	}
	delete  [] tb;
} 

struct FeatVertex
{
	Vec2f coord;
	int     type;
	int     level;
	int     index;	
};
struct FeatEdge
{
	int        first;
	int        second;
	int32_t  mask;
	int32_t  fmask;
	int32_t  smask;
	int        level;	
	int8_t    val[LEDGE_SX*LEDGE_SY];
};

template<class T> inline T clamp( T v, const T lo,const T hi )
{
	return std::max(std::min(v,hi),lo);
}

template<class AV> class FindNeig
{
	typedef vector<AV*>  VERT; 
	typedef VERT * PVERT;	
	deque<int> lst;
	PVERT* STABLE;
	Vec2f shift;
	Vec2f shift0;
	float dScale;
	int    cellX;
	int    cellY;
	public:
	FindNeig(vector<FeatVertex> & vertices,int W,int H, float numVInCell)
	{
		STABLE  = NULL;
		if(!vertices.size())
		{
			return ;
		}
		shift[0] = 0;
		shift[1] = 0;
		float cellSize = 64;
		if(numVInCell<0.0f)
		{
			cellSize = -numVInCell;
		}
		else
		{
			float area = W*H;
			float cells      = vertices.size()/numVInCell;
			cellSize = sqrt(1)*sqrt(area)/sqrt(cells);
		}
			
		cellX         = ceilf(W/cellSize);
		cellY         = ceilf(H/cellSize);
		dScale  = 1.0f/cellSize;
		
		STABLE =new PVERT[cellX*cellY];	
		for(int i=0;i<cellX*cellY;i++)
		{
			STABLE[i] =NULL; 
		}
		for(int i=0;i<vertices.size();i++)
		{
			Vec2f tr = (vertices[i].coord+shift)*dScale;
			int indX =tr[0];
			int indY =tr[1];
			int ind = clamp(indY*cellX+indX,0,cellX*cellY-1);
			if(STABLE[ind]==NULL)
			{
				STABLE[ind] = new VERT();
				lst.push_back(ind);
			}
			STABLE[ind]->push_back(&vertices[i]);
		}
		shift0 = shift;
		shift-=Vec2f(0.5f,0.5f)/dScale;
		
	}
	struct  mVert8
	{
		AV* vt;
		float       currMin;
		float       dummy;
	};
	inline float sdot(Vec2f a)
	{
		return a[0]*a[0]+a[1]*a[1];
	}
	inline int  findNeigAn4(const Vec2f coord,mVert8*  res ,const int n)
	{
		Vec2f tr = (coord+shift)*dScale;//-Vec2f(0.5f,0.5f);
		int indX = tr[0];
		int indY = tr[1];
		for(int k=0;k<n+1;k++)
		{
			res[k].currMin = 100000000.0f;
			res[k].vt         =  NULL;
		}
		mVert8 * pr = &res[n];
		for(int yy=0;yy<2;yy++)
		{
			int indYY = indY+yy;
			if(indYY>=0&&indYY<cellY)
			{	
				for(int xx=0;xx<2;xx++)
				{

					int indXX = indX+xx;
					if(indXX>=0&&indXX<cellX)
					{
						PVERT ST = STABLE[indYY*cellX+indXX];
						if(ST)
						{
							for(auto k=ST->begin();k!=ST->end();k++)
							{
								
								pr->vt          =*k;
								pr->currMin = sdot(coord-pr->vt->coord);
								for(int s=n;s>0;s--)
								{
									if(res[s].currMin<res[s-1].currMin)
									{
										swap(res[s],res[s-1]);
									}
									else break;
								}
							}
						}
					}
				}
			}
		}
		if(res[n].vt==NULL) 
		{
			for(int yy=-1;yy<3;yy++)
			{
				int indYY = indY+yy;
				if(indYY>=0&&indYY<cellY)
				{	
					for(int xx=-1;xx<3;xx++)
					{
						if(!((yy==0||yy==1)&&(xx==0||xx==1)))
						{
							int indXX = indX+xx;
							if(indXX>=0&&indXX<cellX)
							{
								PVERT ST = STABLE[indYY*cellX+indXX];
								if(ST)
								{
									for(auto k=ST->begin();k!=ST->end();k++)
									{
										pr->vt          =*k;
										pr->currMin = sdot(coord-pr->vt->coord);
										for(int s=n;s>0;s--)
										{
											if(res[s].currMin<res[s-1].currMin)
											{
												swap(res[s],res[s-1]);
											}
											else break;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		int     cntr = n;
		while(cntr>=0&&res[cntr].vt==NULL)
		{	
			cntr--;
		}
		return cntr+1;
	}
	~FindNeig()
	{
		for(deque<int>::iterator i =  lst.begin();i!=lst.end();i++)
		{
			delete STABLE[*i]; 
			STABLE[*i] = NULL;
		}
		lst.clear();
		delete [] STABLE;
	}
	
};


class Detector
{
	Mat orig;
	Mat featuresImage;
	public:
	Mat lowPassedImage;
	int level;
	
	Detector(){};
	
	void setImage(Mat& src_image);
	void setImage2(Detector& upperLevelDet);

	void detect(float thresh,vector<FeatVertex>& verts,vector<FeatEdge>& edges)
	{
		const int W = orig.size().width;
		const int H = orig.size().height;
		
		vector<FeatVertex> vertices_local;
#define S 2		
		const float mscale = 1.0f/9.0f;
		for(int y=S;y<H-S;y++)
		{
			float  * pb   = featuresImage.ptr<float>(y);
			const float  * pbs   = lowPassedImage.ptr<float>(y);
			for(int x=S;x<W-S;x+=2)
			{
				float DT = -2.0f*pbs[x];
				float DT1 = -2.0f*pbs[x+1];
				float DX = pbs[x+S]+pbs[x-S];
				float DY = pbs[x+S*W]+pbs[x-S*W];
				float DXY = pbs[x-S-S*W]+pbs[x+S+S*W] - pbs[x+S-S*W]-pbs[x-S+S*W];
				float DX1 = pbs[x+S+1]+pbs[x-S+1];
				float DY1 = pbs[x+S*W+1]+pbs[x-S*W+1];
				float DXY1 = pbs[x-S-S*W+1]+pbs[x+S+S*W+1] - pbs[x+S-S*W+1]-pbs[x-S+S*W+1];
				pb[x]=((DT+DX)*(DT+DY)-mscale*DXY*DXY);
				pb[x+1]=((DT1+DX1)*(DT1+DY1)-mscale*DXY1*DXY1);
			}
		}
		boxFilter3(featuresImage);
		boxFilter3(featuresImage);
		
		for(int y=2;y<H-2;y++)
		{
			const float * porig        =  orig.ptr<float>(y);
			const float * plowPass  =  lowPassedImage.ptr<float>(y);
			const float * pb            =  featuresImage.ptr<float>(y);
			for(int x=2;x<W-2;x++)
			{
				float pbval = pb[x]; 
				if(
					pbval>thresh&&
					pbval>pb[x+1]&&
					pbval>pb[x-1]&&
					pbval>pb[x+W]&&
					pbval>pb[x-W]&&
					pbval>pb[-W-1+x]&&	
					pbval>pb[-W+1+x]&&	
					pbval>pb[W-1+x]&&	
					pbval>pb[W+1+x]
				)
				{
					// find 2'nd order subpixel maximum
					float dx = (pb[x+1] - pb[x-1])   * 0.5f;
					float dy = (pb[x+W] - pb[x-W]) * 0.5f;
					float dxx = (pb[x+1] + pb[x-1] - 2.0f * pbval);
					float dyy = (pb[x+W] + pb[x-W] - 2.0f * pbval);
					float dxy = (pb[x+1+W] - pb[x+1-W] - pb[x-1+W] +pb[x-1-W]) *0.25f;
					float det = (dxx*dyy-dxy*dxy);
					
					det = 1.0f/det;
					FeatVertex tt;
				
					tt.coord = Vec2f(x,y)-Vec2f((dyy*dx - dxy*dy),(dxx*dy - dxy*dx))*det;
					tt.type   = porig[x]>plowPass[x];
					tt.level   = level; 
					vertices_local.push_back(tt);
				}
			}
		}
		// make bigraph edges
		vector<FeatVertex> Red;
		vector<FeatVertex> Green;
		for(int k=0;k<vertices_local.size();k++)
		{
			FeatVertex& curr = vertices_local[k];
			curr.index = k; 
			if(curr.type)
			{
				Red.push_back(curr);
			}
			else
			{
				Green.push_back(curr);
			}
		}
		
		if(Green.size()>=1)
		{
			FindNeig<FeatVertex>* fs = new FindNeig<FeatVertex>(Green,W,H,NUM_EDGES_AROUND_VERTEX);
			
			FindNeig<FeatVertex>::mVert8 buff[NUM_EDGES_AROUND_VERTEX+1]; 
			
			float *porigS1       =  lowPassedImage.ptr<float>(0);; 
			
			for(int i=0;i<Red.size();i++)
			{
				int ind0 = Red[i].index;
				Vec2f & coord = Red[i].coord;
				// find closest neig onto buffer :)
				int cnt = fs->findNeigAn4(coord,buff,NUM_EDGES_AROUND_VERTEX);
				
				for(int k=0;k<cnt;k++)
				{
					int ind1 = buff[k].vt->index;
					Vec2f dir  = (buff[k].vt->coord - coord)/2.0f;//*0.5f;  // make edge direction 
					Vec2f dirsw = Vec2f(-dir[1],dir[0])/2.0f;   // make ortodirection 
					Vec2f center = (buff[k].vt->coord+coord)*0.5f;
					uint32_t num =0;
					int n=0;
					float val[LEDGE_SX*LEDGE_SY];
					for(int y=0;y<LEDGE_SY;y++)
					{
						for(int x=0;x<LEDGE_SX;x++)
						{
							Vec2f coord = center+(x-(LEDGE_SX-1)*0.5f)*dir+(y-(LEDGE_SY-1)*0.5f)*dirsw;
							// bilinear sample of smoothed image on coord
							float X = clamp(coord[0],0.0f,float(W-1.01f));
							float Y = clamp(coord[1],0.0f,float(H-1.01f));
							int lx = int(X);
							int ly = int(Y);
							float sx = X-lx;
							float sy = Y-ly;
							float* pp = porigS1+ly*W+lx;
							float val0 = (1.0f-sx)*pp[0]+sx*pp[1];
							float val1 = (1.0f-sx)*pp[W]+sx*pp[W+1];
							val[n] = ((1.0f-sy)*val0+val1*sy);//-meanY;
							n++;
						}
					}
					// make long bitfield feature ~24 bit
					int rcnt =0;
					for(int y=0;y<LEDGE_SY;y++)
					{
						for(int x=0;x<LEDGE_SX;x++)
						{
							num|=(val[y*LEDGE_SX+x]>val[((y+0)%LEDGE_SY)*LEDGE_SX+((x+3)%LEDGE_SX)])<<rcnt;
							rcnt++;
							num|=(val[y*LEDGE_SX+x]>val[((y+1)%LEDGE_SY)*LEDGE_SX+((x+0)%LEDGE_SX)])<<rcnt;
							rcnt++;
						}
					}
					// make short bitfield feature ~6 bit
					int sm = 0;
					int kcnt = 0;
					{
						int y = (LEDGE_SY-1)/2;
						for(int x=0;x<LEDGE_SX;x++)
						{
							for(int z=x+1;z<LEDGE_SX;z++)
							{
								sm|=(val[y*LEDGE_SX+x]>val[y*LEDGE_SX+((z)%LEDGE_SX)])<<kcnt;
								kcnt++;
							}
						}
					}						
					float summ =0;
					for(int k=0;k<LEDGE_SX*LEDGE_SY;k++)
					{
						summ+=val[k];
					}
					summ/=LEDGE_SX*LEDGE_SY; 
					
					FeatEdge c;
					c.mask    = (rcnt>LUT_BITS ? hashLUT(num) : num);
					c.first      = ind0;
					c.second = ind1;
					c.fmask   = num;
					c.smask   = sm;
					c.level      = level;
					for(int k=0;k<LEDGE_SX*LEDGE_SY;k++)
					{
						c.val[k] = (val[k]-summ)/2;
					}
					edges.push_back(c);						
				}
			}
			delete fs;
		}
		// correct of position for 0 base interpolation
		for(int k=0;k<vertices_local.size();k++)
		{
			FeatVertex & curr = vertices_local[k];
			curr.coord[0] = (curr.coord[0]+0.5f)*(1<<level)-0.5f;
			curr.coord[1] = (curr.coord[1]+0.5f)*(1<<level)-0.5f;
			verts.push_back(curr);
		}
		PRINT(vertices_local.size());
	}
};

void Detector::setImage(Mat& src_image)
{
	level = 0;
	orig= src_image.clone();
	
	lowPassedImage= orig.clone();
	// gauss smooth approximation 3 times box3x3
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	
	featuresImage.create(lowPassedImage.size(), lowPassedImage.type());
	
}
void Detector::setImage2(Detector& upperLevelDet)
{
	level = upperLevelDet.level+1;
	Size NewSize;
	NewSize.width   = ((upperLevelDet.lowPassedImage.size().width)/2)&~1;
	NewSize.height  = ((upperLevelDet.lowPassedImage.size().height)/2)&~1;
	//~ cout<<NewSize.width<<" "<<NewSize.height<<"\n";
	
	orig.create(NewSize,upperLevelDet.lowPassedImage.type());
	const int oldW = upperLevelDet.lowPassedImage.size().width;
	
	for(int y=0;y<NewSize.height;y++)
	{
		float * porig        =  orig.ptr<float>(y);
		const float * plp        =  upperLevelDet.lowPassedImage.ptr<float>(y*2);
		for(int x=0;x<NewSize.width;x++)
		{
			porig[x] = (plp[x*2]+plp[x*2+1]+plp[x*2+oldW]+plp[x*2+oldW+1])*0.25f;
		}
	}
	lowPassedImage= orig.clone();
	// gauss smooth approximation 3 times box3x3
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	
	featuresImage.create(lowPassedImage.size(), lowPassedImage.type());
}
class LevelsStore
{
	public:
	vector<FeatVertex>      levelsV[MAX_LEVEL];
	vector<FeatEdge>        levelsE[MAX_LEVEL];
	LevelsStore(vector<FeatVertex> &verts,vector<FeatEdge>&edges,int makeOneLevel);
	~LevelsStore()
	{
		for(int i=0;i<MAX_LEVEL;i++)
		{
			levelsV[i].clear();
			levelsE[i].clear();
		}
	}
};

LevelsStore::LevelsStore(vector<FeatVertex> &verts,vector<FeatEdge>&edges,int makeOneLevel)
{
	for(int i=0;i<MAX_LEVEL;i++)
	{
		levelsV[i].clear();
		levelsE[i].clear();
	}
	          
	for(int i=0;i<verts.size();i++)
	{
		int lev = verts[i].level;
		if(lev>=0&&lev<MAX_LEVEL)
		{
			levelsV[lev].push_back(verts[i]);
		}
		//else  ??
	}

	for(int i=0;i<edges.size();i++)
	{
		int leve = edges[i].level;
		if(leve>=0&&leve<MAX_LEVEL)
		{
			levelsE[leve].push_back(edges[i]);
		}
		//else  ??
	}
	if(makeOneLevel>0)
	{
		HiCl cl("make One Level");
		int base = levelsV[0].size();
		int startLevel = 0;
		for(int lev = 1;lev<MAX_LEVEL;lev++)
		{
			for(int k=0;k<levelsV[lev].size();k++)
			{
				levelsV[0].push_back(levelsV[lev][k]);
			}
			for(int k=0;k<levelsE[lev].size();k++)
			{
				FeatEdge edge = levelsE[lev][k];
				edge.first     +=base;
				edge.second+=base;
				levelsE[0].push_back(edge);
			}
			base+=levelsV[lev].size();
		}
	}
	if(1)  // if want compare multiple times 
	{
		HiCl cl("sort for better cache access");  
		#define BTS 8	
		for(int leve=0;leve<MAX_LEVEL;leve++)
		{
			int size = levelsE[leve].size();
			if(size)
			{
				vector<vector<FeatEdge> >groups(1<<BTS);
				for(int k=0;k<size;k++)
				{
					int ind = (levelsE[leve][k].mask>>(16-BTS))&((1<<BTS)-1);
					groups[ind].push_back(levelsE[leve][k]);
				}
				vector<FeatEdge> ncand;
				for(int g=0;g<(1<<BTS);g++)
				{
					for(int k=0;k<groups[g].size();k++)
					{
						ncand.push_back(groups[g][k]);
					}
				}
				std::swap(levelsE[leve],ncand);
			}
		}
	}
	
}
struct rMis
{
	int distance;
	int first;
	int second;
};

int main(int argc, char** argv )
{

	string inputFileNameA; 
	string inputFileNameB; 
	int makeOneLevel;
	float threshold;
	int    paint;
	int   startLevel;

        COptions Options(argc,argv);
	Options.Parse("-a"    ,inputFileNameA    ," Input File  A ",string("00000001.jpg"));
	Options.Parse("-b"    ,inputFileNameB    ," Input File  B ",string("00000010.jpg"));
	Options.Parse("-t"    	,threshold    ," detection threshold ", 10.0f );
	Options.Parse("-p"    ,paint    ," paint ", 1 );
	Options.Parse("-l"     ,startLevel    ,"Lowest resolution level", 1 );
	Options.Parse("-m"   ,makeOneLevel    , "make one Level", -1);
	 
	if(!Options.IsOk()){Options.Usage("Parse Error");return 0;}
	
    //~ printf("OpenCV: %s", cv::getBuildInformation().c_str());
    Mat imageA;
    Mat imageB;
    
    imageA = imread( inputFileNameA.c_str(), 1 );
    if ( !imageA.data )
    {
        printf("No image A data \n");
        return -1;
    }
    imageB = imread( inputFileNameB.c_str(), 1 );
    if ( !imageB.data )
    {
        printf("No image B data \n");
        return -1;
    }
    Mat gray;
    cvtColor(imageA, gray, COLOR_BGR2GRAY);
    Mat imgFloat;
    Mat imgFloat1;
    
    gray.convertTo(imgFloat, CV_32FC1);
    gray.convertTo(imgFloat1, CV_32FC1);
    
    Mat imgFloatB;
    
    //~ imgFloatB= imgFloat.clone();


    cvtColor(imageB, gray, COLOR_BGR2GRAY);
    //~ 
    gray.convertTo(imgFloatB, CV_32FC1);
    
    startLevel = clamp(startLevel,0,MAX_LEVEL-1);
    
	Detector* dtArr[MAX_LEVEL];
	for(int k=0;k<MAX_LEVEL;k++)
	{
		dtArr[k] =  new Detector; 
	}
	LevelsStore* LS[2];
	for(int tt=0;tt<2;tt++)
	{
		vector<FeatVertex> verts;
		vector<FeatEdge>   edges;
		{    
			HiCl cl("set up images and detect"); 
			for(int k=0;k<MAX_LEVEL;k++)
			{
				if(k==0)
				{
					dtArr[0]->setImage(tt==0?imgFloat:imgFloatB);	
				}
				else
				{
					dtArr[k]->setImage2(*dtArr[k-1]);
				}
			}
			//~     
			for(int k=startLevel;k<MAX_LEVEL;k++)
			{
				
				dtArr[k]->detect(threshold,verts,edges);
			}
		}
		LS[tt] = new LevelsStore(verts,edges,makeOneLevel);
		cout<<"verts "<<verts.size()<<"\n";
		cout<<"edges "<<edges.size()<<"\n";
	}
    // compare

//~ struct countIndex
//~ {
//~ uint count:  10;
//~ uint index:   22;	
//~ };
    
#define SBB                     22							
#define SB 			(1<<SBB)
#define SBM 			(SB-1)
    

    
    vector< vector<int> > LutOfRightEdges(LUT_SIZE);
    
    for(int LV=MAX_LEVEL-1;LV>=0;LV--) 
    {
			vector<FeatEdge>&  RightEdge = LS[1]->levelsE[LV];
			vector<FeatEdge>&  LeftEdge   = LS[0]->levelsE[LV];
			if(RightEdge.size()>=2)
			{
				PRINT(RightEdge.size());
				PRINT(LeftEdge.size());
				HiCl cl("bind Level"); 
				int rcount = 0; 
				int rcountOK = 0; 
				int size = RightEdge.size();
				const int sflg          = (makeOneLevel>0)?makeOneLevel:1;
				const int rmask      = size>(2048*sflg)?(LUT_SIZE-1):0;
				const int smaxSize = 128*sflg;
				const int smaxSizeS = 128*sflg;
				const int smaxDiff  = rmask?(LEDGE_SIZE*4):(LEDGE_SIZE*4);
				const int MIN_VOTES  = rmask?(4*NUM_VOTES_OF_EDGE-1):(4*NUM_VOTES_OF_EDGE-1);
				int smask = 0xff;  //0 or 0xff
				
				
				//~ vector<vector<countIndex > > voting_result(LS[0]->levelsV[LV].size());
				vector<vector<int > > voting_result(LS[0]->levelsV[LV].size());
				
				if(rmask)   // search long mask
				{
					for(int i=0;i<RightEdge.size();i++)
					{
						LutOfRightEdges[RightEdge[i].mask].push_back(i);
					}
					{
						for(auto curr=LeftEdge.begin();curr!=LeftEdge.end();curr++)
						{
							auto & bindN = LutOfRightEdges[curr->mask];
							int size = bindN.size();
							if(size&&size<smaxSize)
							{
								rcount++;
								int indF = curr->first;
								int indS = curr->second;
								rMis TS[NUM_VOTES_OF_EDGE];
								for(int k=0;k<NUM_VOTES_OF_EDGE;k++)
								{
									TS[k].distance = smaxDiff;
								}
								for(int k=0;k<size;k++)
								{
									const FeatEdge& cand = RightEdge[bindN[k]];
									if(cand.fmask==curr->fmask)
									{	
										int summ = 0;
										for(int p=0;p<LEDGE_SIZE;p++) 
										{
											summ+=std::abs(curr->val[p] - cand.val[p]);
										}
										if(summ<TS[NUM_VOTES_OF_EDGE-1].distance) 
										{
											TS[NUM_VOTES_OF_EDGE-1].first        = cand.first;
											TS[NUM_VOTES_OF_EDGE-1].second   = cand.second;
											TS[NUM_VOTES_OF_EDGE-1].distance = summ; 
											for(int s=NUM_VOTES_OF_EDGE-1;s>0;s--)
											{
												if(TS[s].distance<TS[s-1].distance)
												{
													swap(TS[s],TS[s-1]);
												}
												else break;
											}
										} 
									}
								}
								for(int k=0;k<NUM_VOTES_OF_EDGE;k++) 
								{
									if(TS[k].distance<smaxDiff)
									{
										int weight = NUM_VOTES_OF_EDGE-k;
										rcountOK++;
										
										int indHF = TS[k].first;
										int indHS = TS[k].second;
										{
											int  bFound = 0;
											for(int s=0;s<voting_result[indF].size();s++)
											{
												int sflag = ((voting_result[indF][s]&SBM)==indHF);
												voting_result[indF][s]+=weight*SB*sflag;
												bFound |= sflag;
											}
											if(!bFound)
											{
												voting_result[indF].push_back(indHF|(weight*SB));
											}
										}
										{
											int bFound = 0;
											for(int s=0;s<voting_result[indS].size();s++)
											{
												int sflag =((voting_result[indS][s]&SBM)==indHS);
												voting_result[indS][s]+=(weight*SB)*sflag;
												bFound |= sflag;
											}
											if(!bFound)
											{
												voting_result[indS].push_back(indHS|(weight*SB));
											}
										}
									}
								}
							}
										
						}
					}
				}
				else if(smask)  // search short mask
				{
					for(int i=0;i<RightEdge.size();i++)
					{
						LutOfRightEdges[RightEdge[i].smask&smask].push_back(i);
					}
					{
						for(auto curr=LeftEdge.begin();curr!=LeftEdge.end();curr++)
						{
							auto & bindN = LutOfRightEdges[curr->smask&smask];
							int size = bindN.size();
							if(size&&size<smaxSizeS)
							{
								rcount++;
								
								int indF = curr->first;
								int indS = curr->second;
								//~ int rmin = smaxDiff;
								rMis TS[NUM_VOTES_OF_EDGE];
								for(int k=0;k<NUM_VOTES_OF_EDGE;k++)
								{
									TS[k].distance = smaxDiff;
								}
								for(int k=0;k<size;k++)
								{
									const FeatEdge& cand = RightEdge[bindN[k]];
									if(cntBits(curr->fmask^cand.fmask)<6)
									{	
										int summ = 0;
										for(int p=0;p<LEDGE_SIZE;p++) 
										{
											summ+=std::abs(curr->val[p] - cand.val[p]);
										}
										if(summ<TS[NUM_VOTES_OF_EDGE-1].distance) 
										{
											TS[NUM_VOTES_OF_EDGE-1].first        = cand.first;
											TS[NUM_VOTES_OF_EDGE-1].second   = cand.second;
											TS[NUM_VOTES_OF_EDGE-1].distance = summ; 
											for(int s=NUM_VOTES_OF_EDGE-1;s>0;s--)
											{
												if(TS[s].distance<TS[s-1].distance)
												{
													swap(TS[s],TS[s-1]);
												}
												else break;
											}
										} 
									}
								}
								for(int k=0;k<NUM_VOTES_OF_EDGE;k++)
								{
									if(TS[k].distance<smaxDiff)
									{
										int weight = NUM_VOTES_OF_EDGE-k;
										rcountOK++;
										int indHF = TS[k].first;
										int indHS = TS[k].second;
										{
											int  bFound = 0;
											for(int s=0;s<voting_result[indF].size();s++)
											{
												int sflag = ((voting_result[indF][s]&SBM)==indHF);
												voting_result[indF][s]+=weight*SB*sflag;
												bFound |=sflag;
											}
											if(!bFound)
											{
												voting_result[indF].push_back(indHF|(weight*SB));
											}
										}
										{
											int bFound = 0;
											for(int s=0;s<voting_result[indS].size();s++)
											{
												int sflag =((voting_result[indS][s]&SBM)==indHS);
												voting_result[indS][s]+=(weight*SB)*sflag;
												bFound |= sflag;
											}
											if(!bFound)
											{
												voting_result[indS].push_back(indHS|(weight*SB));
											}
										}
									}
								}
							}
										
						}
					}
				}
				else // full search due small numbers
				{
					{
						
						for(auto curr = LeftEdge.begin();curr!=LeftEdge.end();curr++)
						{
							rcount++;
							
							int indF = curr->first;
							int indS = curr->second;
							rMis TS[NUM_VOTES_OF_EDGE];
							for(int k=0;k<NUM_VOTES_OF_EDGE;k++)
							{
								TS[k].distance = smaxDiff;
							}
							for(auto cand = RightEdge.begin();cand != RightEdge.end();cand++)
							{
								if(cntBits(curr->fmask^cand->fmask)<10)
								{
									if(cntBits(curr->smask^cand->smask)<4)
									{
										int summ = 0;
										for(int p=0;p<LEDGE_SIZE;p++) 
										{
											summ+=std::abs(curr->val[p] - cand->val[p]);
										}
										if(summ<TS[NUM_VOTES_OF_EDGE-1].distance) 
										{
											TS[NUM_VOTES_OF_EDGE-1].first        = cand->first;
											TS[NUM_VOTES_OF_EDGE-1].second   = cand->second;
											TS[NUM_VOTES_OF_EDGE-1].distance = summ; 
											for(int s=NUM_VOTES_OF_EDGE-1;s>0;s--)
											{
												if(TS[s].distance<TS[s-1].distance)
												{
													swap(TS[s],TS[s-1]);
												}
												else break;
											}
										} 
									}
								}
							}
							for(int k=0;k<NUM_VOTES_OF_EDGE;k++)
							{
								if(TS[k].distance<smaxDiff)
								{
									int weight = NUM_VOTES_OF_EDGE-k;
									rcountOK++;
									int indHF = TS[k].first;
									int indHS = TS[k].second;
									{
										int  bFound = 0;
										for(int s=0;s<voting_result[indF].size();s++)
										{
											int sflag = ((voting_result[indF][s]&SBM)==indHF);
											voting_result[indF][s]+=weight*SB*sflag;
											bFound |=sflag;
										}
										if(!bFound)
										{
											voting_result[indF].push_back(indHF|(weight*SB));
										}
									}
									{
										int bFound = 0;
										for(int s=0;s<voting_result[indS].size();s++)
										{
											int sflag =((voting_result[indS][s]&SBM)==indHS);
											voting_result[indS][s]+=(weight*SB)*sflag;
											bFound |= sflag;
										}
										if(!bFound)
										{
											voting_result[indS].push_back(indHS|(weight*SB));
										}
									}
								}
							}
										
						}
					}
				}
				
				int kcntm = 0;
				int kcnt3 = 0;
				int kcnt4 = 0;
				int color = 0xffff;
				for(int k=0;k<voting_result.size();k++)
				{
						int smax =MIN_VOTES*SB;
						for(int s=0;s<voting_result[k].size();s++)
						{
							smax = std::max(smax,voting_result[k][s]);
						}
						if(smax>MIN_VOTES*SB)
						{
							kcntm++;
							kcnt3+=(smax>>SBB)>(MIN_VOTES+1);
							kcnt4+=(smax>>SBB)>(MIN_VOTES+2);
							Vec2f crd1 = LS[0]->levelsV[LV][k].coord;
							Vec2f crd2 = LS[1]->levelsV[LV][smax&SBM].coord;
							if(paint)
							{
								Scalar clr= (smax>>SBB)>(MIN_VOTES)?Scalar(0,255,0):Scalar(0,0,255);
								circle(imageB,Point(crd2),1<<(LV+1),clr);
								line(imageB,Point(crd2),Point(crd1),clr);
							}
						}
				}
				//~ PRINT(LeftEdge.size());
				PRINT(rcount);
				PRINT(rcountOK);
				PRINT(voting_result.size());
				PRINT(kcntm);
				PRINT(kcnt3);
				PRINT(kcnt4);
				if(rmask)
				{
					for(int i=0;i<RightEdge.size();i++)
					{
						LutOfRightEdges[RightEdge[i].mask].clear();
					}
				}
				else if(smask)
				{
					for(int i=0;i<RightEdge.size();i++)
					{
						LutOfRightEdges[RightEdge[i].smask&smask].clear();
					}
				}
			}
	    
	    
    }
	for(int k=0;k<MAX_LEVEL;k++)
	{
		delete dtArr[k]; 
	}
	for(int tt=0;tt<2;tt++)
	{
	       delete LS[tt];
	}
	if(paint)
	{
		namedWindow("Display Image", WINDOW_AUTOSIZE );
		imshow("Display Image", imageB);
		waitKey(0);
	}
	
//~ #define INPLACE_BOX_TEST       
#ifdef INPLACE_BOX_TEST   
	   {
		   HiCl cl("cv::boxFilter");
		   for(int k=0;k<1000;k++)
		   {
			boxFilter(imgFloat,imgFloat,-1,cv::Size(3,3));
		   }
	   }
	   {
		    Mat gray;
		    imgFloat.convertTo(gray, CV_8UC1);
		    
		    namedWindow("Display Image", WINDOW_AUTOSIZE );
		    imshow("Display Image", gray);
		    waitKey(0);
	   }
	   {
		   HiCl cl("boxFilter3");
		   for(int k=0;k<1000;k++)
		   {
			boxFilter3(imgFloat1);
		   }
	   }
	   {
		    Mat gray;
		    imgFloat1.convertTo(gray, CV_8UC1);
		    namedWindow("Display Image", WINDOW_AUTOSIZE );
		    imshow("Display Image", gray);
		    waitKey(0);
	   }
#endif    
    
    return 0;
}
