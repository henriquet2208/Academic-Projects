#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cmath>

//create masks from screenshots for the dataset

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

struct Sail {
    vector<vector<Point>> lines;
    string imagePath;
};

int main(int argc, char* argv[])
{

    // Define input and output folder paths
    string inputFolder = "C:/Users/gusta/source/repos/Project2/Project2/fotos_vela_linha_laranja/fotos_vela_linha_laranja";
    string outputFolder = "C:/Users/gusta/source/repos/Project2/Project2/fotos_vela_linha_laranja/fotos_vela_binarias";


    // Ensure the output folder exists
    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }



    // Iterate over all images in the input folder
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        string imagePath = entry.path().string();

   

    int HueValue;
    int SaturationValue;
    int ConPolyValue = 60;
    int ValueValue;

    int NumberOfLines = 3;

    cout << "Processing Image Path: " << imagePath << endl;

    // Read image
    Mat img = imread(imagePath);
    if (img.empty()) {
        cerr << "Error: Image not found!" << endl;
    }

    cvtColor(img, img, COLOR_BGR2RGB);

    // Get image dimensions
    int height = img.rows;
    int width = img.cols;

    // Convert to HSV
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_RGB2HSV);

    // Split HSV channels
    vector<Mat> channels;
    split(imgHSV, channels);
    Mat& hueChannel = channels[0];

    // Modify Hue channel
    hueChannel += Scalar(30);

    // Ensure Hue values stay within 0-179 range
    Mat maskHigher, maskLower;
    compare(hueChannel, Scalar(180), maskHigher, CMP_GE);
    compare(hueChannel, Scalar(0), maskLower, CMP_LT);
    hueChannel.setTo(Scalar(0), maskHigher);
    hueChannel.setTo(Scalar(180), maskLower);

    // Merge back modified HSV
    merge(channels, imgHSV);

    // Blur
    int blur_threshold = 0;
    if (blur_threshold % 2 == 0) blur_threshold++;
    Mat blur;
    GaussianBlur(imgHSV, blur, Size(blur_threshold, blur_threshold), 3, 0);

    // Binarization & Thresholding
    Mat thresh1;
    threshold(blur, thresh1, ConPolyValue, 255, THRESH_BINARY);

    // Convert back to RGB
    cvtColor(thresh1, thresh1, COLOR_HSV2RGB);

    // Create a binary mask for sail line detection
    Mat mask2;
    inRange(thresh1, Scalar(255, 0, 0), Scalar(255, 255, 0), mask2);

    // Dilation to strengthen edges
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilation;
    dilate(mask2, dilation, kernel);

    // Find contours
    vector<vector<Point>> contours;
    Mat hierarchy;
    findContours(dilation, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    // Sort contours by width (largest first)
    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
        return boundingRect(c2).width < boundingRect(c1).width;
        });

    int line = 0;
    vector<vector<Point>> lines;
    // Create a black mask with the same size as img, but in grayscale (1 channel)
    Mat FinalMask = Mat::zeros(img.size(), CV_8UC1);  // Ensure 1 channel
    // Convert to binary: Threshold so pixels > 127 become 255 (white), others become 0 (black)
    Mat FinalMask2;

    for (auto& contour : contours) {
        Rect bbox = boundingRect(contour);

        int width = bbox.width;
        int height = bbox.height;
        int centerX = dilation.cols / 2;
        int rectCenterX = bbox.x + width / 2;
        int differenceX = abs(centerX - rectCenterX);
        // Filter out noise
        //if (width < height || differenceX >= 1000) {
          //  continue;
        //}

        vector<Point> approx;
        approxPolyDP(contour, approx, 0.0001, true);

        //cout << "line: " << line << "approx.size(): " << approx.size() << endl;
        if (line < NumberOfLines && approx.size() > ConPolyValue) {
            lines.push_back(approx);
            line++;

            // Draw contour
            drawContours(FinalMask, vector<vector<Point>>{approx}, -1, Scalar(255, 255, 255), FILLED);
            //cout << "escrevi countors" << endl;
        }
    }
    // Convert to binary: Threshold so pixels > 127 become 255 (white), others become 0 (black)
    threshold(FinalMask, FinalMask2, 127, 255, THRESH_BINARY);

    // Create output image path
    string outputPath = outputFolder + "\\" + entry.path().filename().string();

    // Save the binary mask
    imwrite(outputPath, FinalMask);

    cout << "Processed: " << imagePath << " -> " << outputPath << endl;

}
}
