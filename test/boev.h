#ifndef BOEV_H
#define BOEV_H
#include <iomanip>
#define PRINT(x) std::cout<<""<<std::setw(15)<<(x)<<" " #x "\n"
#define     MAX_LEVEL 16

#define     NUM_EDGES_AROUND_VERTEX 48	

#define     LEDGE_SX 4
#define     LEDGE_SY 3
#define     LEDGE_SIZE (LEDGE_SX*LEDGE_SY)

// LUT_BITS 13-17 
#define     LUT_BITS (19)
#define     LUT_SIZE (1<<LUT_BITS)

// NUM_VOTES_OF_EDGE 1-5
#define     NUM_VOTES_OF_EDGE           2							
class HiCl
{
	public:
		 int64 e1;
	string Name;
	HiCl(const char*name):Name(name)
	{

	    e1 = cv::getTickCount();
	    
	}
	float time_ms()
	{
		return float(1000*double(cv::getTickCount() - e1)/getTickFrequency());
	}
	~HiCl()
	{
		    int64 e2 = cv::getTickCount();
		    double t = 1000*double(e2 - e1)/getTickFrequency();
		    std::cout<<Name<<" "<<t<<"ms\n";		
	}
};

struct FeatVertex
{
	Vec2f coord;
	float  val;
	int     type;
	int     level;
	int     index;	
};
struct FeatEdge
{
	int        first;
	int        second;
	int32_t  mask;
	uint32_t  fmask;
	uint64_t  lmask;
	int        level;	
	int8_t    val[LEDGE_SX*LEDGE_SY];
};
class Detector
{
	Mat orig;
	Mat featuresImage;
	public:
	Mat lowPassedImage;
	
	Detector(){};
	
	void setImage(Mat& src_image);
	void setImage2(Detector& upperLevelDet);

	void detect(float thresh,vector<FeatVertex>& verts,vector<FeatEdge>& edges,int L_numEdgesAround,int level,float level_scale);
};
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

class BoevDetector
{
	int   L_numEdgesAround;
	int    startLevel;
	float   threshold;
	bool makeHalfLevels;
	Detector* dtArr[MAX_LEVEL];
	float levelScale[MAX_LEVEL];

public:
	BoevDetector(bool _makeHalfLevels = false,int _L_numEdgesAround=NUM_EDGES_AROUND_VERTEX,int _startLevel=1,float _threshold=10.0f)
	{
		L_numEdgesAround = _L_numEdgesAround;
		startLevel = _startLevel;
		threshold  = _threshold;
		makeHalfLevels = _makeHalfLevels;
		for(int k=0;k<MAX_LEVEL;k++)
		{
			dtArr[k] =  new Detector; 
		}
	}
	~BoevDetector()
	{
		for(int k=0;k<MAX_LEVEL;k++)
		{
			delete dtArr[k]; 
		}
	}
	void convertFeatVertex2KeyPoints(vector<FeatVertex> & vertices,vector<KeyPoint> & keyPoints)
	{
		keyPoints.resize(vertices.size());
		for(int i=0;i<vertices.size();i++)
		{
			KeyPoint kp;
			kp.angle = -1;
			kp.octave = vertices[i].level;
			kp.pt = vertices[i].coord;
			kp.response = 1.0f;
			kp.size = 2*levelScale[kp.octave];
			keyPoints[i] = kp;
		}
	}
	void detect(Mat & image,vector<FeatVertex> & vertices,vector<FeatEdge>& edges);
};
class BoevMatcher
{
		LevelsStore* LS_TRAIN;
		LevelsStore* LS_QUERY;
		vector< vector<int> > LutOfRightEdges;
		int makeOneLevel;
	public:
		BoevMatcher(int _makeOneLevel=1)
		{
			LS_TRAIN = NULL;
			LS_QUERY=NULL;
			makeOneLevel = _makeOneLevel;
			LutOfRightEdges.resize(LUT_SIZE);
		}
		~BoevMatcher()
		{
			delete LS_TRAIN;
			delete LS_QUERY;
		}
		void setTrain(vector<FeatVertex> & vertices,vector<FeatEdge>& edges);
		void matchQuery(vector<FeatVertex> & vertices,vector<FeatEdge>& edges,vector<DMatch>& matches);
};
#endif