#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <boost/math/special_functions/trigamma.hpp>


using namespace cv;
using namespace std;

Mat createTexture(int mean, int var, int x = 512 , int y = 512 ){
    Mat texture(Size(512,512),CV_8UC1);
    randn(texture, mean, var); //mean and variance
    normalize(texture, texture, 0.0, 255, NORM_MINMAX, -1, Mat() );
    return texture;
}

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

    for(int x = 0 ; x < nbRowsQR ; x++){
        for(int y = 0 ; y < nbColsQR ; y++){
            if(qrcode.at<uchar>(x, y) != 0 ){
                newImg.at<uchar>(x, y) = texture.at<uchar>(x, y);
            }
        }
    }

    return newImg;
}

void moyenne_block(Mat wqrcode, int l = 8){
    int nbCols = wqrcode.cols;
    int nbRows = wqrcode.rows;

    srand (time(NULL));

    ofstream myfile;
    myfile.open ("t.csv");
    myfile << "Moyenne;Variance\n";

    Mat wqrcode_norm(wqrcode.size(),wqrcode.type());
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(wqrcode, wqrcode_norm, 0, 1, NORM_MINMAX, -1, Mat() );

    for(int x = 0 ; x < nbRows - (l - 1) ; x += l){
        for(int y = 0 ; y < nbCols - (l - 1) ; y += l){


            vector<uchar> v; //Vector de uchar pour stocker les pixels de la fenêtre glissante
            for(int i = 0 ; i < l ; i++){
                for(int j = 0 ; j < l ; j++){
                    //Ajout de la valeur du pixel(x+i,y+j) dans le vector v
                    v.push_back(wqrcode_norm.at<uchar>(x + i, y + j));
                }
            }

            Scalar mean; //Variable pour sotcker la moyenne de la fenêtre glissante
            Scalar deviation; //Variable pour sotcker l'écart type de la fenêtre glissante
            meanStdDev(v,mean,deviation); //Calcul de la moyenne et l'écart type du vector v

            /*double somme_var;
            int v_size = v.size();
            for (int i = 0 ; i < v_size ; ++i){
                somme_var += pow((v[i]-mean.val[0]),2);
            }
            somme_var = somme_var/63;*/

            myfile << mean.val[0] << ";" << pow(deviation.val[0],2) << endl;
        }
    }
    myfile.close();
}

float Statistique_S(Mat qrcode, float alpha, float beta, int l = 8){
    int nbCols = qrcode.cols;
    int nbRows = qrcode.rows;

    vector<double> v_variance;//Vector pour stocker le set des variances des blocks

    for(int x = 0 ; x < nbRows - (l - 1) ; x += l){
        for(int y = 0 ; y < nbCols - (l - 1) ; y += l){

            vector<uchar> v; //Vector de uchar pour stocker les pixels de la fenêtre glissante
            for(int i = 0 ; i < l ; i++){
                for(int j = 0 ; j < l ; j++){
                    //Ajout de la valeur du pixel(x+i,y+j) dans le vector v
                    v.push_back(qrcode.at<uchar>(x + i, y + j));
                }
            }

            Scalar mean; //Variable pour sotcker la moyenne de la fenêtre glissante
            Scalar deviation; //Variable pour sotcker l'écart type de la fenêtre glissante
            meanStdDev(v,mean,deviation); //Calcul de la moyenne et l'écart type du vector v

            v_variance.push_back(deviation.val[0]);
        }
    }
    Scalar mean_v; //Variable pour sotcker la moyenne des variances des blocks
    Scalar deviation_v; //Variable pour sotcker l'écart type des variances des blocks
    meanStdDev(v_variance,mean_v,deviation_v);

    double Bn;
    int size_v = v_variance.size();
    for(int i = 0 ; i < size_v ; ++i){
        Bn += (v_variance[i] - mean_v.val[0]) * (log(v_variance[i]) - log(mean_v.val[0]));
    }
    Bn += Bn/size_v;
    cout << Bn << endl;
    float n = pow(beta,2) * (1+alpha*boost::math::trigamma(alpha));

    float S = (sqrt(size_v)/sqrt(n))*(Bn - beta);
    return S;
}


int main()
{
    Mat texture = createTexture(200,70);

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

    Mat W_QRcode = substitution(imgQr,texture);
    imwrite("./resources/wqrcodes/wqrcode_200_70_1.jpg", W_QRcode);
    imshow("W_QRcode",W_QRcode);
    waitKey(0);

    //moyenne_block(W_QRcode);
    //imshow( "Image", W_QRcode );
    //waitKey(0);
    //Mat histo5 = histoToMat(texture);
    //imshow( "Histo", histo5 );                // Show our image inside it.
    //waitKey(0);

    //cout << "Statistique S : " << Statistique_S(texture,2.4,8.5) << endl;

    /*
    ofstream myfile;
    myfile.open ("example.csv");
    myfile << "This is the first cell in the first column.\n";
    myfile << "a,b,c,\n";
    myfile << "c,s,v,\n";
    myfile << "1,2,3.456\n";
    myfile << "semi;colon";
    myfile.close();
    */

    return 0;
}
