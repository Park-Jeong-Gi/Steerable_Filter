#pragma warning (disable:4996)

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "cmath"

using namespace cv;
using namespace std;

void DoSteerable(Mat Gaus, Mat pBin, int Theta) {
    // �̺а���� ���� ����� �迭�� ��ȯ
    float* array = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.rows; i++)
        for (int j = 0; j < Gaus.cols; j++) {
            array[Gaus.cols * i + j] = Gaus.at<uchar>(i, j);
        }


    //x�� ���� �̺� - Gx
    float* Mibunx = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.rows; i++)
        for (int j = 0; j < Gaus.cols; j++) {

            if (j == Gaus.cols - 1)    Mibunx[Gaus.cols * i + j] = Mibunx[Gaus.cols * i + j - 1]; // �� �࿡�� ������ �ȼ����� �̺п� ����� ���� �ȼ��� ��� �׳� �� �� �ȼ����� �״�� �Է��Ѵ�
            else
                Mibunx[Gaus.cols * i + j] = (array[Gaus.cols * i + j + 1] - array[Gaus.cols * i + j]);

        }


    //x�� ���� 2�� �̺�
    float* Mibunx2 = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.rows; i++)
        for (int j = 0; j < Gaus.cols; j++) {

            if (j == Gaus.cols - 1)    Mibunx2[Gaus.cols * i + j] = Mibunx2[Gaus.cols * i + j - 1];
            else
                Mibunx2[Gaus.cols * i + j] = (Mibunx[Gaus.cols * i + j + 1] - Mibunx[Gaus.cols * i + j]);

        }


    //y�� ���� �̺� - Gy
    float* Mibuny = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.cols; i++)
        for (int j = 0; j < Gaus.rows; j++) {

            if (j == Gaus.rows - 1)    Mibuny[Gaus.cols * j + i] = Mibuny[Gaus.cols * j + i - Gaus.cols];
            else
                Mibuny[Gaus.cols * j + i] = (array[Gaus.cols * j + i + Gaus.cols] - array[Gaus.cols * j + i]);

        }


    //y�� ���� 2�� �̺�
    float* Mibuny2 = new float[Gaus.rows * Gaus.cols];
    // if(Theta<=90)
    for (int i = 0; i < Gaus.cols; i++)
        for (int j = 0; j < Gaus.rows; j++) {

            if (j == Gaus.rows - 1)    Mibuny2[Gaus.cols * j + i] = Mibuny2[Gaus.cols * j + i - Gaus.cols];
            else
                Mibuny2[Gaus.cols * j + i] = (Mibuny[Gaus.cols * j + i + Gaus.cols] - Mibuny[Gaus.cols * j + i]);

        }


    //xy�� ���� �̺�
    float* Mibunxy = new float[Gaus.rows * Gaus.cols];
    // if(Theta<=90)
    for (int i = 0; i < Gaus.cols; i++)
        for (int j = 0; j < Gaus.rows; j++) {

            if (j == Gaus.rows - 1)    Mibunxy[Gaus.cols * j + i] = Mibunxy[Gaus.cols * j + i - Gaus.cols];
            else
                Mibunxy[Gaus.cols * j + i] = (Mibunx[Gaus.cols * j + i + Gaus.cols] - Mibunx[Gaus.cols * j + i]);

        }


    float* Mibunx2_cal = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.rows; i++)
        for (int j = 0; j < Gaus.cols; j++) {
            Mibunx2_cal[Gaus.cols * i + j] = Mibunx2[Gaus.cols * i + j] / 30; // �� ���� 0~1�� ������ ���� CV_32F�� �� ���̹Ƿ� 1�� �ʰ����� �ʵ��� 30�� �����ش�
            Mibunx2_cal[Gaus.cols * i + j] = Mibunx2_cal[Gaus.cols * i + j] * cos(Theta * 3.14 / 180) * cos(Theta * 3.14 / 180); // ������ �־��� ��
        }


    float* Mibuny2_cal = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.cols; i++)
        for (int j = 0; j < Gaus.rows; j++) {
            Mibuny2_cal[Gaus.cols * j + i] = Mibuny2[Gaus.cols * j + i] / 30;
            Mibuny2_cal[Gaus.cols * j + i] = Mibuny2_cal[Gaus.cols * j + i] * sin(Theta * 3.14 / 180) * sin(Theta * 3.14 / 180);
        }

    float* Mibunxy_cal = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.cols; i++)
        for (int j = 0; j < Gaus.rows; j++) {
            Mibunxy_cal[Gaus.cols * j + i] = Mibunxy[Gaus.cols * j + i] / 30;
            Mibunxy_cal[Gaus.cols * j + i] = -2 * Mibunxy_cal[Gaus.cols * j + i] * sin(Theta * 3.14 / 180) * cos(Theta * 3.14 / 180);
        }


    float* Mibunx_cal = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.rows; i++)
        for (int j = 0; j < Gaus.cols; j++) {
            Mibunx_cal[Gaus.cols * i + j] = Mibunx[Gaus.cols * i + j] / 30; // �� ���� 0~1�� ������ ���� CV_32F�� �� ���̹Ƿ� 1�� �ʰ����� �ʵ��� 30�� �����ش�
            Mibunx_cal[Gaus.cols * i + j] = Mibunx_cal[Gaus.cols * i + j] * cos(Theta * 3.14 / 180); // ������ �־��� ��
        }


    float* Mibuny_cal = new float[Gaus.rows * Gaus.cols];
    for (int i = 0; i < Gaus.cols; i++)
        for (int j = 0; j < Gaus.rows; j++) {
            Mibuny_cal[Gaus.cols * j + i] = Mibuny[Gaus.cols * j + i] / 30;
            Mibuny_cal[Gaus.cols * j + i] = Mibuny_cal[Gaus.cols * j + i] * sin(Theta * 3.14 / 180);
        }




    Mat x(size(Gaus), CV_32F, Mibunx);
    Mat y(size(Gaus), CV_32F, Mibuny);
    Mat x1(size(Gaus), CV_32F, Mibunx_cal);
    Mat y1(size(Gaus), CV_32F, Mibuny_cal);
    Mat x2(size(Gaus), CV_32F, Mibunx2_cal);
    Mat y2(size(Gaus), CV_32F, Mibuny2_cal);
    Mat xy(size(Gaus), CV_32F, Mibunxy_cal);

    for (int i = 0; i < Gaus.rows; i++)
        for (int j = 0; j < Gaus.cols; j++) {
            pBin.at<float>(i, j) = x1.at<float>(i, j) + y1.at<float>(i, j); // �Ѱ��� pBin�� ���� ����
        }
}



void DoROIFiltering(Mat pGray, Mat pSteer, Mat pROI)
{
    // array�� �׷��̽����� ������ �� ���� �������� ���� ������
    double* array = new double[pGray.rows];
    for (int i = 0; i < pGray.rows; i++)
        for (int j = 0; j < pGray.cols; j++) {
            if (j == 0) array[i] = 0;  // array�� �ʱ�ȭ���� �ʾ����Ƿ� j==0�� �� 0���� �ʱ�ȭ
            array[i] = array[i] + pGray.at<uchar>(i, j);  // Mat.at<float>(y,x)  //  (i+1)��° ���� ������Ű�� x�ุ ��ȭ���Ѱ��� ���Ѵ�
        }

    // ���� ������ ���� ��
    int minheight = 1; // minheight���� ���� ���� ���� ���̴� ���� ���° ������
    double mindata = 0; // mindata���� �� ���� ���� �������� ���� - �̴� �� ���� ���� ���ϱ� ���ؼ�
    for (int i = 0; i < pGray.rows; i++)
        if (i == 0) mindata = array[i];
        else if (mindata > array[i]) {
            mindata = array[i];  // i+1��° ���� ���� mindata���� ������ �� ���� ����
            minheight = i + 1;
        }

    printf("������⸦ ���� ��: ������ %d��° ��\n", minheight);



    int midvalue = (int)(pSteer.cols / 2); // ������� ���� x �߰���ǥ

    for (int i = 0; i < pSteer.rows; i++)
        for (int j = 0; j < pSteer.cols; j++)


            pROI.at<float>(i, j) = pSteer.at<float>(i, j);

    for (int i = 0; i < (pSteer.rows + 1 - minheight) * 2 / 3 + minheight - 1; i++) {
        double calc = (double)pSteer.cols / ((double)pSteer.rows - (double)minheight) / 2;  // ����� ���� ����
        int numleft = midvalue - (int)(3 * (calc * (i - minheight + 1)) / 2);
        int numright = midvalue + (int)(3 * (calc * (i - minheight + 1)) / 2);


        for (int j = 0; j < pSteer.cols; j++) {
            if (i <= minheight - 1) pROI.at<float>(i, j) = 0;
            else if (j<numleft || j>numright) pROI.at<float>(i, j) = 0;


        }
    }

}

void draw_line2(Mat Original, Mat Unsigend_ROI)
{
    Mat hough;
    vector<Vec2f> linesL, linesR;
    float resultLine[2];
    HoughLines(Unsigend_ROI, linesL, 1, CV_PI / 180, 100, 0, 0, 0, CV_PI / 2); // threshold �� �������� �ִ��ּҰ� ����
    HoughLines(Unsigend_ROI, linesR, 1, CV_PI / 180, 100, 0, 0, CV_PI / 2, CV_PI);

    for (int i = 0; i < linesL.size(); i++)
    {
        resultLine[0] = linesL[0][0];
        resultLine[1] = linesL[0][1];

        if (linesL[i][1] < resultLine[1])
        {
            resultLine[0] = linesL[i][0];
            resultLine[1] = linesL[i][1];
        }
    }
    float rho = resultLine[0];
    float theta = resultLine[1];

    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    Point pt1(cvRound(x0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
        cvRound(y0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));
    Point pt2(cvRound(x0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
        cvRound(y0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));



    for (int i = 0; i < linesR.size(); i++)
    {
        resultLine[0] = linesR[0][0];
        resultLine[1] = linesR[0][1];

        if (linesR[i][1] > resultLine[1])
        {
            resultLine[0] = linesR[i][0];
            resultLine[1] = linesR[i][1];
        }
    }
    rho = resultLine[0];
    theta = resultLine[1];
    a = cos(theta), b = sin(theta);
    x0 = a * rho, y0 = b * rho;
    Point pt3(cvRound(x0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
        cvRound(y0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));
    Point pt4(cvRound(x0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
        cvRound(y0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));



    float leftLineA = (float)(pt2.y - pt1.y) / (float)(pt2.x - pt1.x); // ����
    float leftLineB = pt2.y - leftLineA * pt2.x; // 

    float rightLineA = (float)(pt4.y - pt3.y) / (float)(pt4.x - pt3.x);
    float rightLineB = pt4.y - rightLineA * pt4.x;

    Point banishP;
    banishP.x = (int)((rightLineB - leftLineB) / (leftLineA - rightLineA));
    banishP.y = (int)(leftLineA * banishP.x + leftLineB);

    line(Original, pt1, banishP, cvScalar(255, 0, 255), 3);
    line(Original, pt4, banishP, cvScalar(255, 0, 255), 3);





}
void draw_line1(Mat Original, Mat Unsigend_ROI)
{
    Mat hough;
    vector<Vec2f> linesL, linesR;
    float resultLine[2];
    HoughLines(Unsigend_ROI, linesL, 1, CV_PI / 180, 150, 0, 0, 0, CV_PI / 2); // threshold �� �������� �ִ��ּҰ� ����
    HoughLines(Unsigend_ROI, linesR, 1, CV_PI / 180, 100, 0, 0, CV_PI / 2, CV_PI);
    for (int i = 0; i < linesL.size(); i++)
    {

        float rho = linesL[i][0];
        float theta = linesL[i][1];
        printf("%.1f\n", theta * 3.14);
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        Point pt1(cvRound(x0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
            cvRound(y0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));
        Point pt2(cvRound(x0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
            cvRound(y0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));

        line(Original, pt1, pt2, cvScalar(255, 0, 0), 1);
    }


    for (int i = 0; i < linesR.size(); i++)
    {

        float rho = linesR[i][0];
        float theta = linesR[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        Point pt3(cvRound(x0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
            cvRound(y0 + (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));
        Point pt4(cvRound(x0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (-b)),
            cvRound(y0 - (Unsigend_ROI.rows + Unsigend_ROI.cols) * (a)));

        line(Original, pt3, pt4, cvScalar(0, 0, 255), 1);
    }








}


int main()
{
    int BTheta = 45;
    //printf("degree ������ �Է��ϼ���: ");
    //scanf("%d", &Theta);


    Mat src1;
    src1 = imread("oo.png", CV_LOAD_IMAGE_COLOR);
    imshow("Orig", src1);



    // ������� ��ȯ
    Mat gray;
    cvtColor(src1, gray, CV_BGR2GRAY);


    // ����þ� �Լ�
    Mat AGaus;
    GaussianBlur(gray, AGaus, Size(9, 9), 1);


    Mat Steer1(size(AGaus), CV_32F, Scalar(0));
    Mat Steer2(size(AGaus), CV_32F, Scalar(0));
    DoSteerable(AGaus, Steer1, BTheta);
    DoSteerable(AGaus, Steer2, -BTheta);

    Mat SteerFinal;
    SteerFinal = Steer1 + Steer2;



    for (int i = 0; i < AGaus.rows; i++)
        for (int j = 0; j < AGaus.cols; j++) {
            if (SteerFinal.at<float>(i, j) < 0.3) SteerFinal.at<float>(i, j) = 0; // ��Ⱚ�� �������� ���ϸ� �ƿ� ��Ӱ� ó��
            if (Steer1.at<float>(i, j) > 0.1 && Steer2.at<float>(i, j) > 0.1) SteerFinal.at<float>(i, j) = SteerFinal.at<float>(i, j) / 20;
            // ��ȣ�� �ٸ� �� ������ ���͸����� �� ���ÿ� ������ �̻��� ���� ���̸� �̴� �������򼺺����� �ǹ�
            // �������򼺺��� ��Ⱚ�� �ٿ��ش�
        }

    imshow("1st_Steerable", SteerFinal);

    Mat ROI(size(AGaus), CV_32F, Scalar(0));
    DoROIFiltering(AGaus, SteerFinal, ROI);
    for (int i = 0; i < ROI.rows; i++)
        for (int j = 0; j < ROI.cols; j++) {
            if (ROI.at<float>(i, j) < 0.3) ROI.at<float>(i, j) = 0; // ��Ⱚ�� �������� ���ϸ� �ƿ� ��Ӱ� ó��
        }





    Mat gray_ROI;
    normalize(ROI, gray_ROI, 0, 255, NORM_MINMAX, CV_8U); // hough transform �˰����� ���� ������Ÿ���� CV_32F���� CV_8U�� ��ȯ
    draw_line1(src1, gray_ROI);
    draw_line2(src1, gray_ROI);





    waitKey(0);
    return 0;

}



/*
for (int i = 0; i < Steer.rows; i++) {
   for (int j = 0; j < Steer.cols; j++) {
      float A = Mibunx2[i * Steer.cols + j] * Mibunx2[i * Steer.cols + j];
      A = A - 2 * Mibunx2[i * Steer.cols + j] * Mibuny2[i * Steer.cols + j];
      A = A + Mibuny2[i * Steer.cols + j] * Mibuny2[i * Steer.cols + j];
      A = A + 4 * Mibunxy[i * Steer.cols + j] * Mibunxy[i * Steer.cols + j];
      A = sqrt(A);
      float Theta_min = atan((Mibunx2[i * Steer.cols + j] - Mibuny2[i * Steer.cols + j] - A) / 2);
      float Theta_max = atan((Mibunx2[i * Steer.cols + j] - Mibuny2[i * Steer.cols + j] + A) / 2);

      //printf("(%.1f, %.1f)", Theta_min, Theta_max);
      //printf("%f  ", Mibunxy[i * Steer.cols + j]);
      // min���� max ���� ���� ������ botts dots�� �ν��ϰ� �����
      if(abs(Theta_min-Theta_max)>1.58) Steer.at<float>(i, j) = 0;

   }
   printf("\n");
}


for (int i = 0; i < Gaus.rows; i++) {
      for (int j = 0; j < Gaus.cols; j++) {
         float A = Mibunx2[i * Gaus.cols + j] * Mibunx2[i * Gaus.cols + j];
         A = A - 2 * Mibunx2[i * Gaus.cols + j] * Mibuny2[i * Gaus.cols + j];
         A = A + Mibuny2[i * Gaus.cols + j] * Mibuny2[i * Gaus.cols + j];
         A = A + 4 * Mibunxy[i * Gaus.cols + j] * Mibunxy[i * Gaus.cols + j];
         A = sqrt(A);
         float Theta_min = 0;
         float Theta_max = 0;
         if (Mibunxy[i * Gaus.cols + j]!=0) Theta_min = atan((Mibunx2[i * Gaus.cols + j] - Mibuny2[i * Gaus.cols + j] - A) / 2 / Mibunxy[i * Gaus.cols + j]);
         if (Mibunxy[i * Gaus.cols + j]!=0) Theta_max = atan((Mibunx2[i * Gaus.cols + j] - Mibuny2[i * Gaus.cols + j] + A) / 2 / Mibunxy[i * Gaus.cols + j]);


         //printf("(%f, %f)", Theta_min, Theta_max);
         //printf("%f  ", Mibunxy[i * Steer.cols + j]);
         // min���� max ���� ���� ������ botts dots�� �ν��ϰ� �����
         if ((abs(Theta_min - Theta_max) < 1.57) && (abs(Theta_min - Theta_max) !=0)) pBin.at<float>(i, j) = 0;

      }
      printf("\n");
   }
*/