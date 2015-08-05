#ifndef CCONFIGURATIONCAMERA_H_
#define CCONFIGURATIONCAMERA_H_

#include "vision/CPinholeCamera.h"

class CConfigurationCamera
{

//ds fields
public:

    static struct LEFT
    {
        //ds camera_1 (left)
        static const uint32_t uWidth;
        static const uint32_t uHeight;
        static const double dFx;
        static const double dFy;
        static const double dCx;
        static const double dCy;
        static const double dFocalLengthMeters;
        static const double matIntrinsic[9];
        static const double vecDistortionCoefficients[4];
        static const double matQuaternionToIMU[4];
        static const double vecTranslationToIMU[3];
        static const double matRectification[9];
        static const double matProjection[12];

        static const CPinholeCamera cPinholeCamera;

    } LEFT;

    static struct RIGHT
    {
        //ds camera_0 (right)
        static const uint32_t uWidth;
        static const uint32_t uHeight;
        static const double dFx;
        static const double dFy;
        static const double dCx;
        static const double dCy;
        static const double dFocalLengthMeters;
        static const double matIntrinsic[9];
        static const double vecDistortionCoefficients[4];
        static const double matQuaternionToIMU[4];
        static const double vecTranslationToIMU[3];
        static const double matRectification[9];
        static const double matProjection[12];

        static const CPinholeCamera cPinholeCamera;

    } RIGHT;

    static const double matTransformationIntialAlberto[16];
    static const double matTransformationIntialStandard[16];
    static const double matRotationIMUtoLEFT[9];
    static const double matRotationToIMU[9];

};

//ds camera_1 (left)
const uint32_t CConfigurationCamera::LEFT::uWidth  = 752;
const uint32_t CConfigurationCamera::LEFT::uHeight = 480;
const double CConfigurationCamera::LEFT::dFx = 468.2793078854663;
const double CConfigurationCamera::LEFT::dFy = 466.4527618561632;
const double CConfigurationCamera::LEFT::dCx = 368.7120388971904;
const double CConfigurationCamera::LEFT::dCy = 215.186116509721;
const double CConfigurationCamera::LEFT::dFocalLengthMeters = 0.0028;

const double CConfigurationCamera::LEFT::matIntrinsic[]              = {dFx, 0, dCx, 0, dFy, dCy, 0, 0, 1};
const double CConfigurationCamera::LEFT::vecDistortionCoefficients[] = {-0.2796582742482296, 0.0733556480116312, -0.0009278639320681781, -8.855407606759842e-06};
const double CConfigurationCamera::LEFT::matQuaternionToIMU[]        = {-0.00333631563313, 0.00154028789643, -0.0114620263178, 0.999927556608};
const double CConfigurationCamera::LEFT::vecTranslationToIMU[]       = {0.0666914200614, 0.0038316133947, -0.0101029245794};
const double CConfigurationCamera::LEFT::matRectification[]          = {0.9999606380973329, -0.005153089493488114, -0.0072227366452659965, 0.005142607646202396, 0.9999856976355481, -0.0014690510348766648, 0.007230203494507346, 0.0014318495095343982, 0.9999728366132802};
const double CConfigurationCamera::LEFT::matProjection[]             = {450.5097158071153, 0.0, 375.9431800842285, 0.0, 0.0, 450.5097158071153, 222.3379611968994, 0.0, 0.0, 0.0, 1.0, 0.0};

const CPinholeCamera CConfigurationCamera::LEFT::cPinholeCamera = CPinholeCamera( "LEFT", matIntrinsic, vecDistortionCoefficients, matRectification, matProjection, matQuaternionToIMU, vecTranslationToIMU, uWidth, uHeight, dFocalLengthMeters );



//ds camera_0 (right)
const uint32_t CConfigurationCamera::RIGHT::uWidth  = 752;
const uint32_t CConfigurationCamera::RIGHT::uHeight = 480;
const double CConfigurationCamera::RIGHT::dFx = 469.8960716280004;
const double CConfigurationCamera::RIGHT::dFy = 468.1438849121085;
const double CConfigurationCamera::RIGHT::dCx = 375.2015644784387;
const double CConfigurationCamera::RIGHT::dCy = 233.2718971993311;
const double CConfigurationCamera::RIGHT::dFocalLengthMeters = 0.0028;

const double CConfigurationCamera::RIGHT::matIntrinsic[]              = {dFx, 0, dCx, 0, dFy, dCy, 0, 0, 1};
const double CConfigurationCamera::RIGHT::vecDistortionCoefficients[] = {-0.2786354177535735, 0.07230699700225292, -0.0003789301271210703, 8.083561308516752e-05};
const double CConfigurationCamera::RIGHT::matQuaternionToIMU[]        = {-0.00186686047363, 6.55239757426e-05, -0.00862255915657, 0.999961080249};
const double CConfigurationCamera::RIGHT::vecTranslationToIMU[]       = {-0.0434705406089, 0.00417949317011, -0.00942355850866};
const double CConfigurationCamera::RIGHT::matRectification[]          = {0.999990753148139, 0.0005481502260720453, -0.004265342840574327, -0.0005419629065900193, 0.9999987995316526, 0.001451623732757562, 0.004266133428042508, -0.0014492986522044342, 0.9999898497679818};
const double CConfigurationCamera::RIGHT::matProjection[]             = {450.5097158071153, 0.0, 375.9431800842285, -49.63250853439215, 0.0, 450.5097158071153, 222.3379611968994, 0.0, 0.0, 0.0, 1.0, 0.0};

const CPinholeCamera CConfigurationCamera::RIGHT::cPinholeCamera = CPinholeCamera( "RIGHT", matIntrinsic, vecDistortionCoefficients, matRectification, matProjection, matQuaternionToIMU, vecTranslationToIMU, uWidth, uHeight, dFocalLengthMeters );

const double CConfigurationCamera::matTransformationIntialAlberto[] = { 0.994504,  0.0803288, 0.0671501, -3.14989,
                                                                        0.052872,  0.168259, -0.984324,   0.0468936,
                                                                       -0.0903682, 0.982464,  0.163087,  -1.38018,
                                                                               0,  0,         0,          1};

const double CConfigurationCamera::matTransformationIntialStandard[] = { 1, 0,  0, 0,
                                                                         0, 0, -1, 0,
                                                                         0, 1,  0, 0,
                                                                         0, 0,  0, 1 };

#endif //#define CCONFIGURATIONCAMERA_H_
