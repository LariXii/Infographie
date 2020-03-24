#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>


using namespace cv;
using namespace std;

Mat histoToMat(Mat img){
  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true;
  bool accu = false;

  Mat hist;

  /// Compute the histograms:
  calcHist( &img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accu );

  // Draw the histograms for B, G and R
  int hist_w = 512;
  int hist_h = 512;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( 512, 512, CV_8UC1, Scalar(0) );
  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar(255), 2, 8, 0  );
  }

  return histImage;
}

Mat substitution(Mat qrcode, Mat texture){
    int nbColsQR = qrcode.cols;
    int nbRowsQR = qrcode.rows;
    int nbColsTEXT = texture.cols;
    int nbRowsTEXT = texture.rows;

    if(nbColsQR != nbColsTEXT || nbRowsQR != nbRowsTEXT){
        resize(texture,texture,Size(nbRowsQR,nbColsQR)); //save the small background image to variable imgBack_small
    }

    //Postula : Le qrcode est binarisé
    Mat newImg;
    qrcode.copyTo(newImg);
    cout << nbColsQR << endl;
    imshow(" ",texture);
    waitKey(0);
    for(int x = 0 ; x < nbRowsQR ; x++){
        for(int y = 0 ; y < nbColsQR ; y++){
            if(qrcode.at<uchar>(x, y) != 0 ){ //Comparaison entre la valeur du pixel et la valeur du seuil
                newImg.at<uchar>(x, y) = texture.at<uchar>(x, y);
            }
        }
    }

    return newImg;
}


int main()
{
    Mat texture(Size(512,512),CV_8UC1);
    randn(texture, 200, 70); //mean and variance

    normalize(texture, texture, 0.0, 255, NORM_MINMAX, -1, Mat() );

    imwrite("./resources/textures/texture_200_70.jpg", texture);
    imshow( "Image", texture );
    waitKey(0);
    Mat histo = histoToMat(texture);
    imshow( "Histo", histo );                // Show our image inside it.
    waitKey(0);

    //------- Read QR code image -------

    //define the variable imageQR of type string with the path to the QR code image
    string imageQR("./resources/qrcodes/qrtest.png"); // path to the image

    Mat imgQr;
    imgQr = imread(imageQR, IMREAD_GRAYSCALE); // Read the file as a grayscale

    if( imgQr.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    imshow( "Captured QR code", imgQr );
    waitKey(0);

    Mat W_QRcode = substitution(imgQr,texture);
    imwrite("./resources/wqrcodes/wqrcode_200_70_1.jpg", W_QRcode);
    imshow("W_QRcode",W_QRcode);
    waitKey(0);
    return 0;
}
