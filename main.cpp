#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <algorithm>
#include <vector>



using namespace cv;
using namespace std;

/**
    Fonction de calcul de la variance et la moyenne des blocs de pixels de la matrice donnée en paramètre.
    Les blocs sont par défaut de 8*8 pixels.
*/
void block_variance(Mat wqrcode, vector<float>& v_variance, vector<float>& v_moyenne, int l = 8){
    int nbCols = wqrcode.cols;
    int nbRows = wqrcode.rows;

    for(int x = 0 ; x < nbRows - (l - 1) ; x += l){
        for(int y = 0 ; y < nbCols - (l - 1) ; y += l){

            vector<uchar> v; //Vector de uchar pour stocker les pixels du bloc
            for(int i = 0 ; i < l ; i++){
                for(int j = 0 ; j < l ; j++){
                    //Ajout de la valeur du pixel(x+i,y+j) dans le vector v
                    v.push_back(wqrcode.at<uchar>(x + i, y + j));
                }
            }

            Scalar mean; //Variable pour sotcker la moyenne du bloc
            Scalar deviation; //Variable pour sotcker l'écart type du bloc
            meanStdDev(v,mean,deviation); //Calcul de la moyenne et l'écart type du bloc de pixels

            //Calcul de la variance
            double somme_var;
            int v_size = v.size();
            for (int i = 0 ; i < v_size ; ++i){
                somme_var += pow((v[i]-mean.val[0]),2);
            }
            somme_var = somme_var/63;

            v_variance.push_back(somme_var); //Sauvegarde de la variance du bloc
            v_moyenne.push_back(mean.val[0]); //Sauvegarde de la moyenne du bloc
        }
    }
}

/**
    Fonction permettant de créer une texture de taille N*M à l'aide d'un couple moyenne et sigma
*/
Mat createTexture(int mean, int var, int x = 512 , int y = 512 ){
    Mat texture(Size(512,512),CV_8UC1); //Création d'une matrice de 512*512 pixels en noir et blanc
    randn(texture, mean, var); //mean and variance
    normalize(texture, texture, 0, 255, NORM_MINMAX, -1, Mat() ); //Normalisation de la matrice entre 0 et 255
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

/**
    Fonction pour ajouter une image de fond à un QR code.
    Principe : un qr code étant binaire nous remplaçons dans l'image du qr code tous les pixels blanc par le pixel
    correspondant dans l'image du fond. Ainsi la lecture du qr code est préservé.
*/
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

/**
    Fonction pour calculer la moyenne des log d'un vector de float.
*/
float log_moyenne(vector<float> v){
    float log_mean;
    int size_v = v.size();
    for(int i = 0 ; i < size_v ; ++i){
        if(v[i] != 0){
            log_mean += log(v[i]);
        }
    }
    return log_mean/size_v;
}

float Statistique_S(Mat wqrcode, float alpha, float beta, int l = 8){

    vector<float> v_variance;
    vector<float> v_moyenne;
    //Récupération des variances et des moyennes des blocs du wqrcode
    block_variance(wqrcode,v_variance,v_moyenne);

    Scalar mean;
    Scalar deviation;
    meanStdDev(v_variance,mean,deviation); //Calucl de la moyenne et de la variance des blocs de variances du wqr code

    float log_mean = log_moyenne(v_variance); //Calcul de la moyenne des logs des variances des blocs
    double Bn;

    //Calcul de la covariance entre X et Z avec Z = log(X)
    int size_v = v_variance.size();
    for(int i = 0 ; i < size_v ; ++i){
        Bn += (v_variance[i] - mean.val[0]) * (log(v_variance[i]) - log_mean);
    }
    Bn += Bn/size_v;

    float n = pow(beta,2) * (1+alpha*boost::math::trigamma(alpha)); //Calcul de n

    float S = (sqrt(size_v)/sqrt(n))*(Bn - beta); //Calcul de la statistique S
    return S;
}

void MaximumLikehood(Mat wqrcode, float& alpha, float& beta){
    vector<float> v_variance;
    vector<float> v_moyenne;
    //Récupération des variances et des moyennes des blocs du wqr code
    block_variance(wqrcode,v_variance,v_moyenne);

    Scalar mean;
    Scalar deviation;
    meanStdDev(v_variance,mean,deviation); //Calucl de la moyenne et de la variance des blocs de variances du wqr code

    float log_mean = log(mean.val[0]); //Calcul du log de la moyenne des variances des blocs du wqr code
    float mean_log = log_moyenne(v_variance); //Calcul de la moyenne du log des variances des blocs du wqr code

    //Calcul de l'estimateur du maximum de likehood.
    float alpha0 = 0.5 / (log_mean - mean_log); //Calcul de alpha 0
    // Calcul de alpha 1
    float alphak = 1 / ((1/alpha0) + (mean_log - log_mean + log(alpha0) - boost::math::digamma(alpha0)) / (pow(alpha0,2) * (1/alpha0 - boost::math::trigamma(alpha0))));

    //L'estimation du maximum de likehood converge rapidement, environ 4 itérations.
    for(int i = 1 ; i < 4 ; ++i){
        alpha0 = alphak;
        alphak = 1 / ((1/alpha0) + (mean_log - log_mean + log(alpha0) - boost::math::digamma(alpha0)) / (pow(alpha0,2) * (1/alpha0 - boost::math::trigamma(alpha0))));
    }

    alpha = alphak;
    beta = mean.val[0] / alpha;
}


int main()
{
    //Mat texture = createTexture(200,70);

    //imwrite("./resources/textures/texture_200_70.jpg", texture);
    //imshow( "Image", texture );
    //waitKey(0);
    //Mat histo = histoToMat(texture);
    //imshow( "Histo", histo );                // Show our image inside it.
    //waitKey(0);

    //------- Read QR code image -------

    //define the variable imageQR of type string with the path to the QR code image
    /*string imageQR("./resources/qrcodes/qrtest.png"); // path to the image

    Mat imgQr;
    imgQr = imread(imageQR, IMREAD_GRAYSCALE); // Read the file as a grayscale

    if( imgQr.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }*/

    ///Mat W_QRcode = substitution(imgQr,texture);
    ///imwrite("./resources/wqrcodes/wqrcode_200_70_1.jpg", W_QRcode);
    ///imshow("W_QRcode",W_QRcode);
    ///waitKey(0);

    //------- Read W-QR code image -------

    /*string imageWQR_1("./resources/wqrcodes/wqrcode_200_70_1.jpg"); // path to the image

    Mat imgWQr_1;
    imgWQr_1 = imread(imageWQR_1, IMREAD_GRAYSCALE); // Read the file as a grayscale

    if( imgWQr_1.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    imshow("W_QRcode",imgWQr_1);
    waitKey(0);

    float alpha_1;
    float beta_1;
    MaximumLikehood(imgWQr_1, alpha_1, beta_1);
    cout << "Alpha : " << alpha_1 << "\nBeta : " << beta_1 << endl;*/

    //Ouverture d'un fichier csv pour suavegarder les valeurs alpha et beta des wqr codes
    ofstream myfile;

    myfile.open ("alpha_beta.csv"); //Ouverture du fichier
    myfile << "Alpha;Beta;Type\n"; //Initialisation des entêtes

    //Ouverture d'un fichier csv pour suavegarder les valeurs de la statistique S des wqr codes
    ofstream stats;

    stats.open ("stats.csv"); //Ouverture du fichier
    stats << "S;Type\n"; //Initialisation des entêtes

    ///Lecture des 10 bon wqr codes

    for(int i = 1 ; i < 11 ; ++i){
        ///------- Read W-QR code image -------
        string imageWQR("./resources/wqrcodes/wrqrcode_200_70_genuine" + to_string(i) + ".jpg"); // path to the image

        Mat imgWQr;
        imgWQr = imread(imageWQR, IMREAD_GRAYSCALE); // Read the file as a grayscale

        if( imgWQr.empty() )// Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        cout << "Ouverture de l'image genuine " << to_string(i) << endl;

        float alpha;
        float beta;
        MaximumLikehood(imgWQr, alpha, beta); //Calcul du alpha et du beta à l'aide de l'estimateur du maximum de likehood
        myfile << alpha << ";" << beta << ";" << "genuine" << endl; //Ecriture des résultats dans le fichier csv alpha_beta.csv

        cout << "Alpha : " << alpha << "\nBeta : " << beta << endl;

        float S = Statistique_S(imgWQr, alpha, beta); //Calcul de la statistique S à l'aide de l'alpha et beta calculé précédemment
        stats << S << ";" << "genuine" << endl; //Ecriture du résultat dans le fichier csv stats.csv

        cout << "Statistique S : " << S << endl;
    }

    ///Lecture des 10 faux wqr codes

    for(int i = 1 ; i < 11 ; ++i){
        ///------- Read W-QR code image -------
        string imageWQR("./resources/wqrcodes/wrqrcode_200_70_fake" + to_string(i) + ".jpg"); // path to the image

        Mat imgWQr;
        imgWQr = imread(imageWQR, IMREAD_GRAYSCALE); // Read the file as a grayscale

        if( imgWQr.empty() )                      // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        cout << "Ouverture de l'image genuine " << to_string(i) << endl;

        float alpha;
        float beta;
        MaximumLikehood(imgWQr, alpha, beta); //Calcul du alpha et du beta à l'aide de l'estimateur du maximum de likehood
        myfile << alpha << ";" << beta << ";" << "fake" << endl; //Ecriture des résultats dans le fichier csv alpha_beta.csv

        cout << "Alpha : " << alpha << "\nBeta : " << beta << endl;

        float S = Statistique_S(imgWQr, alpha, beta); //Calcul de la statistique S à l'aide de l'alpha et beta calculé précédemment
        stats << S << ";" << "fake" << endl; //Ecriture du résultat dans le fichier csv stats.csv

        cout << "Statistique S : " << S << endl;
    }

    myfile.close(); //Fermeture du fichier alpha_beta.csv
    stats.close(); //Fermeture du fichier stats.csv
    return 0;
}
