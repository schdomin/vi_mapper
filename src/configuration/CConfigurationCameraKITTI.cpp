#include "CConfigurationCameraKITTI.h"

//ds camera_1 (left)
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
