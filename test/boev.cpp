#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include "boev.h" 

inline uint32_t hashLUT(const uint32_t inp)
{
	//~ uint64_t s=inp^0x6666666666666666ul;
	//~ uint32_t k = s+(s>>(LUT_BITS));
	return ((inp)+(inp>>LUT_BITS))&(LUT_SIZE-1);
	//~ return (inp^(inp>>LUT_BITS)^(inp>>(2*LUT_BITS)))&(LUT_SIZE-1);
}

void reduceHalf5(Mat& d0,Mat& d1)
{
	const int W2 = d0.size().width;
	const int H2 = d0.size().height;
	
	const float reduceFact = 1.5f;
	const int W = int(W2/reduceFact)&~1;
	const int H  = int(H2/reduceFact)&~1; 
	Size NewSize;
	NewSize.width   = W;
	NewSize.height  = H;
	//~ if(NewSize.width<8||NewSize.height<8)
	//~ cout<<NewSize.width<<" "<<NewSize.height<<"\n";
	
	d1.create(NewSize,d0.type());
	
	
	float oldXS =   (0.5f)*reduceFact-0.5f;
	float oldXS1 = (1.5f)*reduceFact-0.5f;
	float oldX =oldXS;
	int      xxb = int(oldX);
	const float dxx = oldX-xxb;
	float oldX1 = oldXS1;
	int xx1b = int(oldX1);
	const float dxx1 = oldX1-xx1b;
	for(int y=0;y<H;y++)   
	{ 
		{
			float oldY = (y+0.5f)*reduceFact-0.5f;
			int       yy = int(oldY);
			float  dyy =oldY-yy;
			float * p0 =  d0.ptr<float>(yy);
			float * d1p = d1.ptr<float>(y);
			for(int x = 0;x<W/2;x++)
			{
				{
					int xx= xxb+x*3;
					float v0 = (1.0f-dxx)*p0[xx]+dxx*p0[xx+1];
					float v1 = (1.0f-dxx)*p0[xx+W2]+dxx*p0[xx+1+W2];
					d1p[x*2] = (1.0f-dyy)*v0+dyy*v1;
				}
				{
					int xx1= xx1b+x*3;
					float v0 = (1.0f-dxx1)*p0[xx1]+dxx1*p0[xx1+1];
					float v1 = (1.0f-dxx1)*p0[xx1+W2]+dxx1*p0[xx1+1+W2];
					d1p[x*2+1] = (1.0f-dyy)*v0+dyy*v1;
				}
				
			}
		}
		
	}
#ifdef TEST_IMAGE_REDUCE	
	static int imageN = 0;
	imageN++;
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	compression_params.push_back(IMWRITE_PNG_STRATEGY);
	compression_params.push_back(IMWRITE_PNG_STRATEGY_DEFAULT);
	 
	char filename[0x100];
	sprintf(filename,"%d.png",imageN);
	try 
	{
		imwrite(filename, d1, compression_params);
	}
	catch (runtime_error& ex) 
	{
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
	//~ return ;
	}
	
#endif	
	//~ PRINT(W2);
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



	void Detector::detect(float thresh,vector<FeatVertex>& verts,vector<FeatEdge>& edges,int L_numEdgesAround,int level,float level_scale)
	{
		const int W = orig.size().width;
		const int H = orig.size().height;
		if(W<16||H<16) return;
		vector<FeatVertex> vertices_local;
#define S 2		
		const float mscale = 1.0f/9.0f;
		for(int y=0;y<H;y++)
		{
			float  * pb   = featuresImage.ptr<float>(y);
			for(int x=0;x<W;x++)
			{
				pb[x] = 0.0f;
			}
		}
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
		//~ boxFilter3(featuresImage);
		
		for(int y=4;y<H-4;y++)
		{
			const float * porig        =  orig.ptr<float>(y);
			const float * plowPass  =  lowPassedImage.ptr<float>(y);
			const float * pb            =  featuresImage.ptr<float>(y);
			for(int x=4;x<W-4;x++)
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
				        Vec2f rv = Vec2f((dyy*dx - dxy*dy),(dxx*dy - dxy*dx))*det;
					if(rv[0]*rv[0]+rv[1]*rv[1]<1.4f)
					{
						tt.coord = Vec2f(x,y)-rv;
					}
					else
					{
						tt.coord = Vec2f(x,y);
					}
					tt.type   = porig[x]>plowPass[x];
					tt.val     =  plowPass[x];
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
			FindNeig<FeatVertex>* fs = new FindNeig<FeatVertex>(Green,W,H,L_numEdgesAround);
			
			FindNeig<FeatVertex>::mVert8 buff[L_numEdgesAround+1]; 
			
			float *porigS1       =  lowPassedImage.ptr<float>(0);; 
			
			for(int i=0;i<Red.size();i++)
			{
				int ind0 = Red[i].index;
				Vec2f & coord = Red[i].coord;
				// find closest neig onto buffer :)
				int cnt = fs->findNeigAn4(coord,buff,L_numEdgesAround);
				
				for(int k=0;k<cnt;k++)
				{
					int ind1 = buff[k].vt->index;
					const float kt = std::sqrt(2.0f); 
					Vec2f dir  = (buff[k].vt->coord - coord)/kt/2.0f;//*0.5f;
					Vec2f dirsw = Vec2f(-dir[1],dir[0])*kt/4.0f; 
					
					//~ Vec2f dir  = (buff[k].vt->coord - coord)/2.0f;//*0.5f;  // make edge direction 
					//~ Vec2f dirsw = Vec2f(-dir[1],dir[0])/2.0f;   // make ortodirection 
					Vec2f center = (buff[k].vt->coord+coord)*0.5f;
					uint64_t num =0;
					int n=0;
					float val[LEDGE_SIZE];
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
								//~ num|=(val[y*LEDGE_SX+x]+0.0f>val[((y+x+1)%LEDGE_SY)*LEDGE_SX+((x+y+2)%LEDGE_SX)])<<rcnt;
								//~ rcnt++;
								//~ num|=(val[y*LEDGE_SX+x]+0.0f>val[((y+1)%LEDGE_SY)*LEDGE_SX+((x+1)%LEDGE_SX)])<<rcnt;
								//~ rcnt++;
							num+=(val[y*LEDGE_SX+x]>val[((y+0)%LEDGE_SY)*LEDGE_SX+((x+3)%LEDGE_SX)])<<rcnt;
							rcnt++;
							num+=(val[y*LEDGE_SX+x]>val[((y+1)%LEDGE_SY)*LEDGE_SX+((x+0)%LEDGE_SX)])<<rcnt;
							rcnt++;
						}
					}
					
					num+=(buff[k].vt->val>Red[i].val)<<(rcnt&31ul);
					rcnt++;
					//~ num+=(val[n-1]>Red[i].val)<<(rcnt&31ul);
					//~ rcnt++;
					//~ num+=(val[0]>buff[k].vt->val)<<(rcnt&31ul);
					//~ rcnt++;
					num+=(val[0]>val[n-1])<<(rcnt&31ul);
					rcnt++;
					//~ num+=(val[0]>val[n/2])<<(rcnt&31ul);
					//~ rcnt++;
					//~ num+=(val[n-1]>val[n/2])<<(rcnt&31ul);
					//~ rcnt++;
					
					uint64_t lmask =0;
					uint64_t lmask_cnt =0;
					
					lmask^=(buff[k].vt->val>Red[i].val)<<((lmask_cnt*7)&63ul);
					//~ lmask^=(buff[k].vt->val>Red[i].val)<<((lmask_cnt*lmask_cnt+1)&63ul);;
					lmask_cnt++;
					for(int k0=0;k0<LEDGE_SIZE;k0++)
					{
						for(int k1=k0+1;k1<LEDGE_SIZE;k1++)
						{
							uint64_t cn = val[k0]>val[k1];
							lmask^= (cn)<<((lmask_cnt*7)&63ul);
							//~ lmask^= (cn)<<((lmask_cnt*lmask_cnt+1)&63ul);
							lmask_cnt++;
						}
					}
					
					float summ =0;
					for(int k=0;k<LEDGE_SIZE;k++)
					{
						summ+=val[k];
					}
					summ/=LEDGE_SIZE; 
					
					FeatEdge c;
					c.mask    = (rcnt>LUT_BITS ? hashLUT(num) : num);
					c.first      = ind0;
					c.second = ind1;
					c.fmask   = num;
					c.lmask   = lmask;
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
			curr.coord[0] = (curr.coord[0]+0.5f)*level_scale-0.5f;
			curr.coord[1] = (curr.coord[1]+0.5f)*level_scale-0.5f;
			verts.push_back(curr);
		}
		//~ PRINT(vertices_local.size());
	}
 
void Detector::setImage(Mat& src_image)
{
	orig= src_image.clone();
	
	lowPassedImage= orig.clone();
	// gauss smooth approximation 4 times box3x3
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	
	featuresImage.create(lowPassedImage.size(), lowPassedImage.type());
	
}
void Detector::setImage2(Detector& upperLevelDet)
{
	Size NewSize;
	NewSize.width   = ((upperLevelDet.lowPassedImage.size().width)/2)&~1;
	NewSize.height  = ((upperLevelDet.lowPassedImage.size().height)/2)&~1;
	//~ if(NewSize.width<8||NewSize.height<8)
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
	// gauss smooth approximation 4 times box3x3
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	boxFilter3(lowPassedImage);
	
	featuresImage.create(lowPassedImage.size(), lowPassedImage.type());
}
 
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
		//~ HiCl cl("make One Level");
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
		//~ HiCl cl("sort for better cache access");  
		#define BTS 7	
		for(int leve=0;leve<MAX_LEVEL;leve++)
		{
			int size = levelsE[leve].size();
			if(size)
			{
				vector<vector<FeatEdge> >groups(1<<BTS);
				for(int k=0;k<size;k++)
				{
					int ind = (levelsE[leve][k].mask>>(LUT_BITS-BTS))&((1<<BTS)-1);
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

void BoevDetector::detect(Mat & gray,vector<FeatVertex> & vertices,vector<FeatEdge>& edges)
{
	Mat imgFloat;
	gray.convertTo(imgFloat, CV_32FC1);	
	vertices.clear();
	edges.clear();
	dtArr[0]->setImage(imgFloat);	
	levelScale[0] = 1.0f;
	
	if(makeHalfLevels)
	{
		Mat imgFloatH;
		reduceHalf5(dtArr[0]->lowPassedImage,imgFloatH);
		dtArr[1]->setImage(imgFloatH);
		levelScale[1] = 1.5f;
		for(int k=2;k<MAX_LEVEL;k++)
		{
			dtArr[k]->setImage2(*dtArr[k-2]);
			levelScale[k] = levelScale[k-2]*2.0f;
		}
		//~     
	}
	else
	{
		for(int k=1;k<MAX_LEVEL;k++)
		{
			dtArr[k]->setImage2(*dtArr[k-1]);
			levelScale[k] = levelScale[k-1]*2.0f;
		}
		
	}
	for(int level=startLevel;level<MAX_LEVEL;level++)
	{
		dtArr[level]->detect(threshold,vertices,edges,L_numEdgesAround,level,levelScale[level]);
	}
}

void BoevMatcher::setTrain(vector<FeatVertex> & vertices,vector<FeatEdge>& edges)
{
	if(LS_TRAIN)
	{
		delete LS_TRAIN;
	}
	LS_TRAIN = new LevelsStore(vertices,edges,makeOneLevel);
}
void BoevMatcher::matchQuery(vector<FeatVertex> & vertices,vector<FeatEdge>& edges,vector<DMatch>& matches)
{
	matches.clear();
	if(LS_QUERY)
	{
		delete LS_QUERY;
	}
	LS_QUERY = new LevelsStore(vertices,edges,makeOneLevel);
#define SBB                     22							
#define SB 			(1<<SBB)
#define SBM 			(SB-1)
    

    for(int LV=makeOneLevel?0:(MAX_LEVEL-1);LV>=0;LV--) 
    {
	vector<FeatEdge>&  RightEdge = LS_TRAIN->levelsE[LV];
	vector<FeatEdge>&  LeftEdge   = LS_QUERY->levelsE[LV];
	if(RightEdge.size()>=2)
	{
		PRINT(RightEdge.size());
		PRINT(LeftEdge.size());
		HiCl cl("bind Level"); 
		int rcount = 0; 
		int rcountOK = 0; 
		const int rmask          = (LUT_SIZE-1);
		const int smaxSize     = 1<<7;
		const int smaxDiff      = LEDGE_SIZE*10;
		const int MIN_VOTES  = NUM_VOTES_OF_EDGE_MIN;//4*NUM_VOTES_OF_EDGE-4;
		
		vector<vector<int > > voting_result(LS_QUERY->levelsV[LV].size());
		
		for(int i=0;i<RightEdge.size();i++)
		{
			LutOfRightEdges[RightEdge[i].mask].push_back(i);
		}
		for(auto curr=LeftEdge.begin();curr!=LeftEdge.end();curr++)
		{
			auto & bindN = LutOfRightEdges[curr->mask];
			const int size = bindN.size();
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
					//~ if(__builtin_popcount(curr->fmask^cand.fmask)<5)
					if(cand.fmask==curr->fmask)
					{	
						int ks = __builtin_popcountl(curr->lmask^cand.lmask);
						if(ks<7)
						{
							int summ = ks*LEDGE_SIZE;
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
		int kcnt0 = 0;
		int kcnt1 = 0;
		for(int k=0;k<voting_result.size();k++)
		{
				int smax =MIN_VOTES*SB;
				for(int s=0;s<voting_result[k].size();s++)
				{
					smax = std::max(smax,voting_result[k][s]);
				}
				if(smax>MIN_VOTES*SB)
				{
					kcnt0++;
					kcnt1+=(smax>>SBB)>(MIN_VOTES);
					Vec2f crd1 = LS_QUERY->levelsV[LV][k].coord;
					Vec2f crd2 = LS_TRAIN->levelsV[LV][smax&SBM].coord;
					DMatch dm;
					dm.trainIdx = smax&SBM;
					dm.queryIdx = k;
					dm.distance = 1.0f/(smax>>SBB);
					matches.push_back(dm);
				}
		}
		PRINT(voting_result.size());
		PRINT(kcnt0);
		PRINT(kcnt1);
		for(int i=0;i<RightEdge.size();i++)
		{
			LutOfRightEdges[RightEdge[i].mask].clear();
		}
	}
    }
}
