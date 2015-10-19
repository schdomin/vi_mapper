#ifndef CCONFIGURATIONCAMERAKITTI_H
#define CCONFIGURATIONCAMERAKITTI_H

#include "vision/CPinholeCamera.h"

class CConfigurationCameraKITTI
{

//ds fields
public:

    static struct LEFT
    {
        //ds camera_1 (left)
        static const uint32_t uWidth;
        static const uint32_t uHeight;

        static const double matProjection[12];

        static const CPinholeCamera cPinholeCamera;

    } LEFT;

    static struct RIGHT
    {
        //ds camera_0 (right)
        static const uint32_t uWidth;
        static const uint32_t uHeight;

        static const double matProjection[12];

        static const CPinholeCamera cPinholeCamera;

    } RIGHT;

    static const double matTransformationIntialWORLDtoLEFT[16];
    static const double matTransformationIntialLEFTtoWORLD[16];

};

#endif //#define CCONFIGURATIONCAMERAKITTI_H
