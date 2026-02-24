#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("test.png");

    if (img.empty()) {
        cout << "Image not loaded!" << endl;
        return -1;
    }

    Mat gray, blurImg, sobelX, sobelY, sobelCombined, cannyEdges;

    // 1. Grayscale
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 2. Gaussian Blur
    GaussianBlur(gray, blurImg, Size(5, 5), 0);

    // 3. Sobel Edge Detection
    Sobel(blurImg, sobelX, CV_64F, 1, 0, 3);
    Sobel(blurImg, sobelY, CV_64F, 0, 1, 3);
    convertScaleAbs(sobelX, sobelX);
    convertScaleAbs(sobelY, sobelY);
    addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobelCombined);

    // 4. Canny Edge Detection
    Canny(blurImg, cannyEdges, 100, 200);

    // 5. Save Outputs
    imwrite("output_gray.jpg", gray);
    imwrite("output_blur.jpg", blurImg);
    imwrite("output_sobel.jpg", sobelCombined);
    imwrite("output_canny.jpg", cannyEdges);

    // ---------------- GRID VIEW ADDITION ----------------

    // Convert grayscale images to 3-channel BGR
    Mat grayBGR, blurBGR, sobelBGR, cannyBGR;
    cvtColor(gray, grayBGR, COLOR_GRAY2BGR);
    cvtColor(blurImg, blurBGR, COLOR_GRAY2BGR);
    cvtColor(sobelCombined, sobelBGR, COLOR_GRAY2BGR);
    cvtColor(cannyEdges, cannyBGR, COLOR_GRAY2BGR);

    // Resize all images to same size
    Size size(300, 300);
    Mat imgR, grayR, blurR, sobelR, cannyR;

    resize(img, imgR, size);
    resize(grayBGR, grayR, size);
    resize(blurBGR, blurR, size);
    resize(sobelBGR, sobelR, size);
    resize(cannyBGR, cannyR, size);

    // Create grid:
    // [ Original | Grayscale | Blur ]
    // [ Sobel    | Canny     | (empty) ]
    Mat topRow, bottomRow, grid;

    hconcat(vector<Mat>{imgR, grayR, blurR}, topRow);
    hconcat(vector<Mat>{sobelR, cannyR, Mat::zeros(size, CV_8UC3)}, bottomRow);
    vconcat(topRow, bottomRow, grid);

    // Show individual windows (optional, keep if you want)
    imshow("Original", img);
    imshow("Grayscale", gray);
    imshow("Blurred", blurImg);
    imshow("Sobel Combined", sobelCombined);
    imshow("Canny Edges", cannyEdges);

    // Final Grid View
    imshow("Image Processing Toolkit - Grid View", grid);

    waitKey(0);
    return 0;
}