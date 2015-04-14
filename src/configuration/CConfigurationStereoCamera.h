#ifndef CCONFIGURATIONSTEREOCAMERA_H_
#define CCONFIGURATIONSTEREOCAMERA_H_

#include <opencv/cv.h>

class CConfigurationStereoCamera
{

//ds fields
public:

    static struct RIGHT
    {
        //ds camera 0 (right)
        constexpr static double dFx = 469.8960716280004;
        constexpr static double dFy = 468.1438849121085;
        constexpr static double dCx = 375.2015644784387;
        constexpr static double dCy = 233.2718971993311;
        const static cv::Mat matCameraMatrixIntrinsic;
        const static cv::Vec4d vecDistortionCoefficients;
    } RIGHT;

    static struct LEFT
    {
        //ds camera 1 (left)
        constexpr static double dFx = 468.2793078854663;
        constexpr static double dFy = 466.4527618561632;
        constexpr static double dCx = 368.7120388971904;
        constexpr static double dCy = 215.186116509721;
        const static cv::Mat matCameraMatrixIntrinsic;
        const static cv::Vec4d vecDistortionCoefficients;
    } LEFT;

};

//ds camera 0 (right)
const cv::Mat CConfigurationStereoCamera::RIGHT::matCameraMatrixIntrinsic = ( cv::Mat_< double > ( 3, 3 ) << dFx, 0, dCx, 0, dFy, dCy, 0, 0, 1 );
const cv::Vec4d CConfigurationStereoCamera::RIGHT::vecDistortionCoefficients = cv::Vec4d( -0.2786354177535735, 0.07230699700225292, -0.0003789301271210703, 8.083561308516752e-05 );

//ds camera 1 (left)
const cv::Mat CConfigurationStereoCamera::LEFT::matCameraMatrixIntrinsic = ( cv::Mat_< double > ( 3, 3 ) << dFx, 0, dCx, 0, dFy, dCy, 0, 0, 1 );
const cv::Vec4d CConfigurationStereoCamera::LEFT::vecDistortionCoefficients = cv::Vec4d( -0.2796582742482296, 0.0733556480116312, -0.0009278639320681781, -8.855407606759842e-06 );

#endif //#define CCONFIGURATIONSTEREOCAMERA_H_
