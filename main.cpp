#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
// This video stablisation smooths the global trajectory using a sliding average window

const int SMOOTHING_RADIUS = 30; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

using namespace std;
using namespace cv;

const int frame_width = 800;
const int frame_height = 400;
const double match_threshold = 0.178;
const int fast_accuracy = 20;
const int fast_filter = 100;

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};
struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    double x;
    double y;
    double a; // angle
};

int main(int argc,char *argv[])
{
    if(argc < 2) {
        cout << "./VAIS [video.avi]" << endl;
        return 0;
    }

    VideoCapture cap(argv[1]);
    VideoCapture cap(fileName);
    assert(cap.isOpened());
    cap.set(CV_CAP_PROP_FRAME_WIDTH,frame_width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,frame_height);
    const int frames_count = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "Video: "<<argv[1]<<"\nframes totali: "<< frames_count << endl;
    namedWindow("VAIS",CV_WINDOW_KEEPRATIO);
    resizeWindow("VAIS",frame_width*2,frame_height);
    /**Stabilizzazione*/
    ///Stima del moto
    KeyPointsFilter filter;
    vector<TransformParam>prev_to_cur_transform;
    Mat prev_frame, prev_gray;
    vector<KeyPoint> prev_features;
    cap >> prev_frame;
    cvtColor(prev_frame,prev_gray,COLOR_BGR2GRAY);
    //FAST(prev_gray,prev_features,fast_accuracy);
    //filter.retainBest(prev_features,fast_filter);
    for(int k = 0;k < frames_count-1;k++)
    {
        Mat cur_frame,cur_gray;
        vector<KeyPoint> cur_features;
        vector<Point2f> prev_points, prev_good_points;
        vector<Point2f> cur_points,cur_good_points;
        vector<uchar> status;
        vector<float>errors;
        cap >> cur_frame;
        if(!cur_frame.data)break;
        cvtColor(cur_frame,cur_gray,COLOR_BGR2GRAY);
        goodFeaturesToTrack(prev_gray, prev_points, 200 , 0.01, 30);//corner detection
        //filter.retainBest(cur_features,fast_filter);
        //KeyPoint::convert(prev_features,prev_points);
        //KeyPoint::convert(cur_features,cur_points);
        calcOpticalFlowPyrLK(prev_gray,cur_gray,prev_points,cur_points,status,errors);
        for(size_t i =0;i< status.size();i++)
        {
            prev_good_points.push_back(prev_points[i]);
            cur_good_points.push_back(cur_points[i]);
        }
        Mat T = estimateRigidTransform(prev_good_points,cur_good_points,false);
        // decompose T
        double dx = T.at<double>(0,2);
        double dy = T.at<double>(1,2);
        double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

        prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        cur_frame.copyTo(prev_frame);
        cur_gray.copyTo(prev_gray);
        cout << "Calcolo Trasformazione -opticalFlow("<<cur_good_points.size()<<") tra frame("<<k-1<<"-"<<k<<")-("<< k * 100 / frames_count <<"%)" <<endl;
    }
        // Step 2 - Accumulate the transformations to get the image trajectory

    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;

    vector <Trajectory> trajectory; // trajectory at all frames

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        trajectory.push_back(Trajectory(x,y,a));
    }

    // Step 3 - Smooth out the trajectory using an averaging window
    vector <Trajectory> smoothed_trajectory; // trajectory at all frames

    for(size_t i=0; i < trajectory.size(); i++) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_a = 0;
        int count = 0;

        for(int j=-SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++) {
        for(size_t j=-SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++) {
            if(i+j >= 0 && i+j < trajectory.size()) {
                sum_x += trajectory[i+j].x;
                sum_y += trajectory[i+j].y;
                sum_a += trajectory[i+j].a;

                count++;
            }
        }

        double avg_a = sum_a / count;
        double avg_x = sum_x / count;
        double avg_y = sum_y / count;

        smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    vector <TransformParam> new_prev_to_cur_transform;

    // Accumulated frame to frame transform
    a = 0;
    x = 0;
    y = 0;

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        // target - current
        double diff_x = smoothed_trajectory[i].x - x;
        double diff_y = smoothed_trajectory[i].y - y;
        double diff_a = smoothed_trajectory[i].a - a;

        double dx = prev_to_cur_transform[i].dx + diff_x;
        double dy = prev_to_cur_transform[i].dy + diff_y;
        double da = prev_to_cur_transform[i].da + diff_a;

        new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
    }

    // Step 5 - Apply the new transformation to the video
    cap.set(CV_CAP_PROP_POS_FRAMES, 0);
    Mat T(2,3,CV_64F);

    int vert_border = HORIZONTAL_BORDER_CROP * prev_frame.rows / prev_frame.cols; // get the aspect ratio correct
    VideoWriter video("stab.avi",cap.get(CV_CAP_PROP_FOURCC),cap.get(CV_CAP_PROP_FPS), Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
    for(int k = 0;k<frames_count-1;k++) { // don't process the very last frame, no valid transform
        Mat cur_frame;
        cap >> cur_frame;

        if(cur_frame.data == NULL) {
            break;
        }

        T.at<double>(0,0) = cos(new_prev_to_cur_transform[k].da);
        T.at<double>(0,1) = -sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,0) = sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,1) = cos(new_prev_to_cur_transform[k].da);

        T.at<double>(0,2) = new_prev_to_cur_transform[k].dx;
        T.at<double>(1,2) = new_prev_to_cur_transform[k].dy;

        Mat cur2;

        warpAffine(cur_frame, cur2, T, cur_frame.size());

        cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

        // Resize cur2 back to cur size, for better side by side comparison
        resize(cur2, cur2, cur_frame.size());
        video.write(cur2);
        // Now draw the original and stablised side by side for coolness
        Mat canvas = Mat::zeros(cur_frame.rows, cur_frame.cols*2+10, cur_frame.type());

        cur_frame.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        if(canvas.cols > 1920) {
            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
        }

        imshow("VAIS", canvas);

        //char str[256];
        //sprintf(str, "images/%08d.jpg", k);
        //imwrite(str, canvas);

        if(waitKey(20)==27)break;
    }
    video.release();
    return 0;
}
