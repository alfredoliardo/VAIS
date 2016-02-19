#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

const int frame_width = 800;
const int frame_height = 400;
const double match_threshold = 0.178;

bool sortByDistance(DMatch &match1,DMatch &match2){return match1.distance<match2.distance;}
bool isBadMatch(DMatch &match){return match.distance > match_threshold;}

int main(int argc,char *argv[])
{
    if(argc<2)
    {
        cout <<"Errore: specificare file video"<<endl;
        return -1;
    }
    VideoCapture cap(argv[1]);
    assert(cap.isOpened());
    cap.set(CV_CAP_PROP_FRAME_WIDTH,frame_width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,frame_height);
    const int frames_count = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "Video: "<<argv[1]<<"\nframes totali: "<< frames_count << endl;
    namedWindow("VAIS");
    /**Stabilizzazione*/
    ///Stima del moto
    KeyPointsFilter filter;
    Mat prev_frame, prev_gray;
    vector<KeyPoint> prev_features;
    cap >> prev_frame;
    cvtColor(prev_frame,prev_gray,COLOR_BGR2GRAY);
    FAST(prev_gray,prev_features,15);
    filter.retainBest(prev_features,30);
    for(int k = 0;k < frames_count;k++)
    {
        Mat cur_frame,cur_gray;
        vector<KeyPoint> cur_features;
        cap >> cur_frame;
        if(!cur_frame.data)break;
        cvtColor(cur_frame,cur_gray,COLOR_BGR2GRAY);
        FAST(cur_gray,cur_features,15);
        filter.retainBest(cur_features,30);
        // computing descriptors

        SurfDescriptorExtractor extractor;
        Mat descriptors1, descriptors2;
        extractor.compute(prev_gray, prev_features, descriptors1);
        extractor.compute(cur_gray, cur_features, descriptors2);
        // matching descriptors
        BFMatcher matcher(NORM_L2);
        vector< DMatch > matches;
        matcher.match( descriptors1, descriptors2, matches );
        sort(matches.begin(),matches.end(),sortByDistance);
        vector<DMatch>::iterator validEnd = remove_if( matches.begin(), matches.end(),isBadMatch);
        matches.erase( validEnd , matches.end() );
        for(int i = 0;i<matches.size();i++)cout << matches.at(i).distance << endl;
        Mat img_matches;
        drawMatches(prev_frame, prev_features, cur_frame, cur_features, matches, img_matches,Scalar(255,0,0),Scalar(0,255,0),vector<char>(),2);
        cur_frame.copyTo(prev_frame);
        cur_gray.copyTo(prev_gray);

        imshow("VAIS", img_matches);
        waitKey(20);
    }
    return 0;
}
