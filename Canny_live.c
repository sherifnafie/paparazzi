#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>

// Rotate the image 90° CCW
static cv::Mat rotate90CCW(const cv::Mat &src) {
    cv::Mat dst;
    cv::rotate(src, dst, cv::ROTATE_90_COUNTERCLOCKWISE);
    return dst;
}

// Minimal “Canny-like” approach:
//  1) Optional blur
//  2) Sobel gradient magnitude
//  3) Threshold to produce edges
static void cannyLikeEdge(const cv::Mat &gray, cv::Mat &edges) {
    edges = cv::Mat::zeros(gray.size(), CV_8U);

    // e.g. int EDGE_THRESH = 80
    const int EDGE_THRESH = 80;

    // We'll skip the border to avoid boundary issues
    for (int y = 1; y < gray.rows - 1; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {
            // 3×3 region
            int v00 = gray.at<uchar>(y-1, x-1);
            int v01 = gray.at<uchar>(y-1, x);
            int v02 = gray.at<uchar>(y-1, x+1);
            int v10 = gray.at<uchar>(y,   x-1);
            int v12 = gray.at<uchar>(y,   x+1);
            int v20 = gray.at<uchar>(y+1, x-1);
            int v21 = gray.at<uchar>(y+1, x);
            int v22 = gray.at<uchar>(y+1, x+1);

            int gx = (-v00 + v02) - 2*v10 + 2*v12 - v20 + v22;
            int gy = ( v00 + 2*v01 + v02 ) - ( v20 + 2*v21 + v22 );
            int mag = std::abs(gx) + std::abs(gy); // cheap approximation

            if (mag > EDGE_THRESH) {
                edges.at<uchar>(y, x) = 255;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <camera_or_stream>\n"
                  << "Examples:\n"
                  << "  " << argv[0] << " 0\n"
                  << "  " << argv[0] << " rtp://127.0.0.1:5000\n";
        return -1;
    }

    std::string source = argv[1];
    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera/stream: " << source << std::endl;
        return -1;
    }

    cv::namedWindow("Triple View", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break; // end of stream or error
        }

        // Left panel: original BGR
        cv::Mat original = frame.clone();

        // Middle panel: rotate 90° CCW
        cv::Mat rotated = rotate90CCW(original);

        // Right panel: canny on the rotated image
        //   1) Convert rotated to grayscale
        cv::Mat rotatedGray;
        cv::cvtColor(rotated, rotatedGray, cv::COLOR_BGR2GRAY);

        //   2) Run canny-like edge
        cv::Mat edges;
        cannyLikeEdge(rotatedGray, edges);

        //   3) We want to draw lines from the bottom to the first edge in each column
        //      plus highlight the best column with a different color
        //   We'll create a color version for display
        cv::Mat edgeDisplay;
        cv::cvtColor(rotatedGray, edgeDisplay, cv::COLOR_GRAY2BGR);

        int h = edgeDisplay.rows;
        int w = edgeDisplay.cols;

        int best_col = -1;
        int best_dist = -1;

        for (int x = 0; x < w; x++) {
            int dist = 0;
            bool foundEdge = false;
            int yEdge = 0;
            for (int y = h - 1; y >= 0; y--) {
                if (edges.at<uchar>(y, x) == 255) {
                    dist = (h - 1) - y;
                    foundEdge = true;
                    yEdge = y;
                    break;
                }
            }
            if (!foundEdge) {
                // no edge => free entirely
                dist = h;
                yEdge = 0;
            }
            // if it's the best so far, record it
            if (dist > best_dist) {
                best_dist = dist;
                best_col = x;
            }
            // Draw a green line from bottom => yEdge
            cv::line(edgeDisplay, cv::Point(x, h-1), cv::Point(x, yEdge),
                     cv::Scalar(0,255,0), 1);
        }

        // highlight best column in red
        if (best_col >= 0 && best_col < w) {
            int dist = 0;
            bool foundEdge = false;
            int yEdge = 0;
            for (int y = h - 1; y >= 0; y--) {
                if (edges.at<uchar>(y, best_col) == 255) {
                    dist = (h - 1) - y;
                    foundEdge = true;
                    yEdge = y;
                    break;
                }
            }
            if (!foundEdge) {
                dist = h;
                yEdge = 0;
            }
            // thicker red line
            cv::line(edgeDisplay, cv::Point(best_col, h-1), 
                                 cv::Point(best_col, yEdge),
                     cv::Scalar(0,0,255), 2);
            // put some text
            std::ostringstream oss;
            oss << "best_col=" << best_col << " dist=" << dist;
            cv::putText(edgeDisplay, oss.str(), cv::Point(30,30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 2);
        }

        // Now we have three images: original, rotated, edgeDisplay
        // We need them to have the same height to combine horizontally
        // Let's pick a common height (the max of the three's heights).
        int h1 = original.rows,   w1 = original.cols;
        int h2 = rotated.rows,    w2 = rotated.cols;
        int h3 = edgeDisplay.rows,w3 = edgeDisplay.cols;
        int maxH = std::max(std::max(h1,h2), h3);

        // Resize them to the same height for a side-by-side. We'll keep aspect ratio.
        auto resizeToHeight = [&](const cv::Mat &src, int newH){
            int oldH = src.rows;
            int oldW = src.cols;
            if (oldH == 0) {
                return src.clone();
            }
            double scale = (double)newH / (double)oldH;
            int newW = (int)(oldW * scale);
            cv::Mat dst; 
            cv::resize(src, dst, cv::Size(newW, newH));
            return dst;
        };

        cv::Mat left  = resizeToHeight(original,   maxH);
        cv::Mat mid   = resizeToHeight(rotated,    maxH);
        cv::Mat right = resizeToHeight(edgeDisplay,maxH);

        // Now we can hconcat them
        std::vector<cv::Mat> toCombine = { left, mid, right };
        cv::Mat combined;
        cv::hconcat(toCombine, combined);

        cv::imshow("Triple View", combined);
        char c = (char)cv::waitKey(10);
        if (c == 27) { // ESC
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}


