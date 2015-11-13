#include "CConfigurationCameraKITTI.h"

/* CONFIGURATION for 00,13,14,15 */
/*ds camera_1 (left)
const uint32_t CConfigurationCameraKITTI::LEFT::uWidth  = 1241;
const uint32_t CConfigurationCameraKITTI::LEFT::uHeight = 376;

const double CConfigurationCameraKITTI::LEFT::matProjection[]            = {7.188560000000e+02,
                                                                            0.000000000000e+00,
                                                                            6.071928000000e+02,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            7.188560000000e+02,
                                                                            1.852157000000e+02,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            1.000000000000e+00,
                                                                            0.000000000000e+00};



//ds camera_0 (right)
const uint32_t CConfigurationCameraKITTI::RIGHT::uWidth  = 1241;
const uint32_t CConfigurationCameraKITTI::RIGHT::uHeight = 376;

const double CConfigurationCameraKITTI::RIGHT::matProjection[]           = {7.188560000000e+02,
                                                                            0.000000000000e+00,
                                                                            6.071928000000e+02,
                                                                            -3.861448000000e+02,
                                                                            0.000000000000e+00,
                                                                            7.188560000000e+02,
                                                                            1.852157000000e+02,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            1.000000000000e+00,
                                                                            0.000000000000e+00};*/

/* CONFIGURATION for 11,12 */
//ds camera_1 (left)
const uint32_t CConfigurationCameraKITTI::LEFT::uWidth  = 1226;
const uint32_t CConfigurationCameraKITTI::LEFT::uHeight = 370;

const double CConfigurationCameraKITTI::LEFT::matProjection[]            = {7.070912000000e+02,
                                                                            0.000000000000e+00,
                                                                            6.018873000000e+02,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            7.070912000000e+02,
                                                                            1.831104000000e+02,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            1.000000000000e+00,
                                                                            0.000000000000e+00};



//ds camera_0 (right)
const uint32_t CConfigurationCameraKITTI::RIGHT::uWidth  = 1226;
const uint32_t CConfigurationCameraKITTI::RIGHT::uHeight = 370;

const double CConfigurationCameraKITTI::RIGHT::matProjection[]           = {7.070912000000e+02,
                                                                            0.000000000000e+00,
                                                                            6.018873000000e+02,
                                                                            -3.798145000000e+02,
                                                                            0.000000000000e+00,
                                                                            7.070912000000e+02,
                                                                            1.831104000000e+02,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            0.000000000000e+00,
                                                                            1.000000000000e+00,
                                                                            0.000000000000e+00};

const CPinholeCamera CConfigurationCameraKITTI::LEFT::cPinholeCamera = CPinholeCamera( "LEFT", matProjection, uWidth, uHeight );
const CPinholeCamera CConfigurationCameraKITTI::RIGHT::cPinholeCamera = CPinholeCamera( "RIGHT", matProjection, uWidth, uHeight );

const double CConfigurationCameraKITTI::matTransformationIntialWORLDtoLEFT[] = { 1,  0,  0, 0,
                                                                                 0,  0,  1, 0,
                                                                                 0, -1,  0, 0,
                                                                                 0,  0,  0, 1 };
const double CConfigurationCameraKITTI::matTransformationIntialLEFTtoWORLD[] = { 1,  0,  0, 0,
                                                                                 0,  0, -1, 0,
                                                                                 0,  1,  0, 0,
                                                                                 0,  0,  0, 1 };
