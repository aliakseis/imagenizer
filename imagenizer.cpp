// imagenizer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

#include <lbfgs.h>


enum { SCALE = 2 };

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
)
{
    return 0;
}

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
)
{
    auto src = static_cast<cv::Mat*>(instance);

    const cv::Mat x2(src->rows * SCALE,
        src->cols * SCALE,
        CV_64FC1,
        const_cast<void*>(static_cast<const void*>(x)));

    cv::Mat Ax2;
    cv::idct(x2, Ax2);

    double fx = 0;

    for (int y = 0; y < src->rows; ++y)
        for (int x = 0; x < src->cols; ++x)
        {
            double v = 0;
            for (int yy = 0; yy < SCALE; ++yy)
                for (int xx = 0; xx < SCALE; ++xx)
                    v += Ax2.at<double>(y * SCALE + yy, x * SCALE + xx);

            v /= SCALE * SCALE;

            const auto Ax = v - src->at<float>(y, x);

            for (int yy = 0; yy < SCALE; ++yy)
                for (int xx = 0; xx < SCALE; ++xx)
                    Ax2.at<double>(y * SCALE + yy, x * SCALE + xx) = Ax;

            fx += Ax * Ax * SCALE * SCALE;
        }

    cv::Mat AtAxb2(src->rows * SCALE,
        src->cols * SCALE,
        CV_64FC1,
        g);
    cv::dct(Ax2, AtAxb2);
    AtAxb2 *= 2;

    return fx;
};




int main()
{
    /*
    cv::Mat imgIn = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        std::cout << "ERROR : Image cannot be loaded!\n";
        return -1;
    }

    imgIn.convertTo(imgIn, CV_32F);
    */

    cv::Mat imgIn = cv::imread("input.jpg");
    std::vector<cv::Mat> bgr;
    split(imgIn, bgr);

    bgr[1].convertTo(imgIn, CV_32F);


    enum { UPSCALE = 31 };

    const int destW = 44 * UPSCALE;
    const int destH = 16 * UPSCALE;
    
    cv::Mat dest;

    cv::Point2f src_points[] = { {68,25}, {312, 77}, {56, 114}, {298, 171} };

    cv::Point2f dst_points[] = { {0,0}, {destW - 1, 0}, {0, destH - 1}, {destW - 1, destH - 1} };

    auto projective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    cv::warpPerspective(imgIn, dest, projective_matrix, { destW, destH }, cv::INTER_LANCZOS4);

     
    cv::Mat demo0;
    dest.convertTo(demo0, CV_8U);
    cv::imshow("demo0", demo0);


    cv::Mat reducedImg = cv::Mat::zeros(16, 44, CV_32F);

    for (int y = 0; y < dest.rows; ++y) {
        for (int x = 0; x < dest.cols; ++x) {
            if (y % UPSCALE != 0 && y % UPSCALE != (UPSCALE - 1) && x % UPSCALE != 0 && x % UPSCALE != (UPSCALE - 1))
                reducedImg.at<float>(y / UPSCALE, x / UPSCALE) += dest.at<float>(y, x);
            /*
            const auto dy = (y % UPSCALE) - UPSCALE / 2;
            const auto dx = (x % UPSCALE) - UPSCALE / 2;
            const auto coeff = exp(-(dy * dy + dx * dx) / 100.);
            reducedImg.at<float>(y / UPSCALE, x / UPSCALE) += dest.at<float>(y, x) * coeff;
            */
        }
    }

    cv::Mat demo1;
    reducedImg.convertTo(demo1, CV_8U, 1. / (UPSCALE * UPSCALE));
    cv::resize(demo1, demo1, { 44 * 8, 16 * 8 }, 0, 0, cv::INTER_LANCZOS4);

    cv::imshow("demo1", demo1);


    // upscale

    reducedImg -= cv::mean(reducedImg);

    const double param_c = 2.;

    const int numImgPixels = reducedImg.rows * reducedImg.cols * SCALE * SCALE;

    // Initialize solution vector
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(numImgPixels);
    if (x == nullptr) {
        return EXIT_FAILURE;
    }
    for (int i = 0; i < numImgPixels; i++) {
        x[i] = 1;
    }

    // Initialize the parameters for the optimization.
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = param_c; // this tells lbfgs to do OWL-QN
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    int lbfgs_ret = lbfgs(numImgPixels, x, &fx, evaluate, progress, &reducedImg, &param);

    cv::Mat Xat2(reducedImg.rows * SCALE, reducedImg.cols * SCALE, CV_64FC1, x);
    cv::Mat Xa;
    idct(Xat2, Xa);

    lbfgs_free(x);

    /*
    const cv::Mat sharpening_kernel = (cv::Mat_<double>(3, 3)
        << 0, -1, 0,
        -1, 4, -1,
        0, -1, 0);
    cv::Mat sharpened;
    filter2D(Xa, sharpened, -1, sharpening_kernel);
    cv::Mat contrastMask = abs(sharpened);
    contrastMask.convertTo(contrastMask, CV_8U);
    cv::threshold(contrastMask, contrastMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);

    sharpened += Xa;
    sharpened.copyTo(Xa, contrastMask);
    //*/

    normalize(Xa, Xa, 0, 1, cv::NORM_MINMAX);

    /*
    const double gamma = 0.5;
    for (int i = 0; i < Xa.rows; i++)
    {
        for (int j = 0; j < Xa.cols; j++)
        {
            Xa.at<double>(i, j) = pow(Xa.at<double>(i, j), gamma);
        }
    }
    */

    cv::Mat demo;
    Xa.convertTo(demo, CV_8U, 255);
    cv::resize(demo, demo, { 44 * 8, 16 * 8 }, 0, 0, cv::INTER_LANCZOS4);

    cv::imshow("demo", demo);

    cv::waitKey();

    cv::imwrite("output.jpg", demo);
}
