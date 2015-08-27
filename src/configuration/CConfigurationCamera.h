#ifndef CCONFIGURATIONCAMERA_H
#define CCONFIGURATIONCAMERA_H

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
    static const double matTransformationIntialWORLDtoLEFT[16];
    static const double matRotationIntrinsicCAMERAtoIMU[9];

};

#endif //#define CCONFIGURATIONCAMERA_H
