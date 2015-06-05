#ifndef CCONFIGURATIONSTEREOCAMERA_H_
#define CCONFIGURATIONSTEREOCAMERA_H_

#include <opencv/cv.h>
#include <Eigen/Core>

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
        const static cv::Mat matIntrinsic;
        const static Eigen::Vector4d vecDistortionCoefficients;
        const static cv::Vec4d vecDistortionCoefficientsCV;
        const static cv::Mat matRotationToLEFT;
        const static cv::Vec3d vecTranslationToLEFT;
        const static cv::Mat matRectification;
        const static cv::Mat matProjection;
        const static Eigen::Vector2d vecPrincipalPoint;
    } RIGHT;

    static struct LEFT
    {
        //ds camera 1 (left)
        const static double dFx;
        const static double dFy;
        const static double dCx;
        const static double dCy;
        const static cv::Mat matIntrinsic;
        const static cv::Mat matIntrinsicNormalized;
        const static Eigen::Vector4d vecDistortionCoefficients;
        const static cv::Vec4d vecDistortionCoefficientsCV;
        const static cv::Mat matRotationToRIGHT;
        const static cv::Vec3d vecTranslationToRIGHT;
        const static cv::Mat matRectification;
        const static cv::Mat matProjection;
        const static Eigen::Vector2d vecPrincipalPoint;
    } LEFT;

};

//ds camera 0 (right)
const cv::Mat CConfigurationStereoCamera::RIGHT::matIntrinsic = ( cv::Mat_< double > ( 3, 3 ) << 469.8960716280004, 0.0, 375.2015644784387, 0.0, 468.1438849121085, 233.2718971993311, 0.0, 0.0, 1.0 );
const Eigen::Vector4d CConfigurationStereoCamera::RIGHT::vecDistortionCoefficients = Eigen::Vector4d( -0.2786354177535735, 0.07230699700225292, -0.0003789301271210703, 8.083561308516752e-05 );
const cv::Vec4d CConfigurationStereoCamera::RIGHT::vecDistortionCoefficientsCV = cv::Vec4d( -0.2786354177535735, 0.07230699700225292, -0.0003789301271210703, 8.083561308516752e-05 );
const cv::Mat CConfigurationStereoCamera::RIGHT::matRotationToLEFT = ( cv::Mat_< double > ( 3, 3 ) << 0.9999783, -0.0056784, -0.0029262, 0.0056693, 0.9999752, -0.0029723, 0.0029431, 0.0029557, 0.9999883 );
const cv::Vec3d CConfigurationStereoCamera::RIGHT::vecTranslationToLEFT = cv::Vec3d( -1.1016e-01, 3.4788e-04, 6.7937e-04 );
const cv::Mat CConfigurationStereoCamera::RIGHT::matRectification = ( cv::Mat_< double > ( 3, 3 ) << 0.999990753148139, 0.0005481502260720453, -0.004265342840574327, -0.0005419629065900193, 0.9999987995316526, 0.001451623732757562, 0.004266133428042508, -0.0014492986522044342, 0.9999898497679818 );
const cv::Mat CConfigurationStereoCamera::RIGHT::matProjection = ( cv::Mat_< double > ( 3, 4 ) << 450.5097158071153, 0.0, 375.9431800842285, -49.63250853439215, 0.0, 450.5097158071153, 222.3379611968994, 0.0, 0.0, 0.0, 1.0, 0.0 );
const Eigen::Vector2d CConfigurationStereoCamera::RIGHT::vecPrincipalPoint = Eigen::Vector2d( 375.2015644784387, 233.2718971993311 );

//ds camera 1 (left)
const double CConfigurationStereoCamera::LEFT::dFx = 468.2793078854663;
const double CConfigurationStereoCamera::LEFT::dFy = 466.4527618561632;
const double CConfigurationStereoCamera::LEFT::dCx = 368.7120388971904;
const double CConfigurationStereoCamera::LEFT::dCy = 215.186116509721;

const cv::Mat CConfigurationStereoCamera::LEFT::matIntrinsic = ( cv::Mat_< double > ( 3, 3 ) << 468.2793078854663, 0.0, 368.7120388971904, 0.0, 466.4527618561632, 215.186116509721, 0.0, 0.0, 1.0 );
const cv::Mat CConfigurationStereoCamera::LEFT::matIntrinsicNormalized = ( cv::Mat_< double > ( 3, 3 ) << 468.2793078854663/752, 0.0, 368.7120388971904/752, 0.0, 466.4527618561632/480, 215.186116509721/480, 0.0, 0.0, 1.0 );
const Eigen::Vector4d CConfigurationStereoCamera::LEFT::vecDistortionCoefficients = Eigen::Vector4d( -0.2796582742482296, 0.0733556480116312, -0.0009278639320681781, -8.855407606759842e-06 );
const cv::Vec4d CConfigurationStereoCamera::LEFT::vecDistortionCoefficientsCV = cv::Vec4d( -0.2796582742482296, 0.0733556480116312, -0.0009278639320681781, -8.855407606759842e-06 );
const cv::Mat CConfigurationStereoCamera::LEFT::matRotationToRIGHT = ( cv::Mat_< double > ( 3, 3 ) << 0.9999809, 0.0056697, 0.0029430, -0.0056780, 0.9999839, 0.0029556, -0.0029263, -0.0029723, 0.9999943 );
const cv::Vec3d CConfigurationStereoCamera::LEFT::vecTranslationToRIGHT = cv::Vec3d( 1.1016e-01, -3.4788e-04, -6.7937e-04 );
const cv::Mat CConfigurationStereoCamera::LEFT::matRectification = ( cv::Mat_< double > ( 3, 3 ) << 0.9999606380973329, -0.005153089493488114, -0.0072227366452659965, 0.005142607646202396, 0.9999856976355481, -0.0014690510348766648, 0.007230203494507346, 0.0014318495095343982, 0.9999728366132802 );
const cv::Mat CConfigurationStereoCamera::LEFT::matProjection = ( cv::Mat_< double > ( 3, 4 ) << 450.5097158071153, 0.0, 375.9431800842285, 0.0, 0.0, 450.5097158071153, 222.3379611968994, 0.0, 0.0, 0.0, 1.0, 0.0 );
const Eigen::Vector2d CConfigurationStereoCamera::LEFT::vecPrincipalPoint = Eigen::Vector2d( 368.7120388971904, 215.186116509721 );

#endif //#define CCONFIGURATIONSTEREOCAMERA_H_
