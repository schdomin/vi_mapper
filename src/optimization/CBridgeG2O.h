#ifndef CBRIDGEG2O_H
#define CBRIDGEG2O_H

#include "vision/CStereoCamera.h"
#include "types/CLandmark.h"


class CBridgeG2O
{
    enum EG2OParameterID
    {
        eWORLD        = 0,
        eCAMERA_LEFT  = 1,
        eCAMERA_RIGHT = 2,
        eOFFSET_IMUtoLEFT = 3
    };

private:

    static constexpr double m_dMaximumReliableDepth       = 7.5;
    static const uint8_t m_uMinimumCalibrationsForDump    = 1;
    static constexpr double m_dMaximumErrorPerCalibration = 10.0; //ds e.g after 3 calibrations an error of 30.0 is allowed

public:

    static void saveXYZAndDisparity( const std::string& p_strOutfile,
                                     const CStereoCamera& p_cStereoCamera,
                                     const std::vector< CLandmark* >& p_vecLandmarks,
                                     const std::vector< CMeasurementPose >& p_vecMeasurements );
    static void saveUVDepthOrDisparity( const std::string& p_strOutfile,
                                        const CStereoCamera& p_cStereoCamera,
                                        const std::vector< CLandmark* >& p_vecLandmarks,
                                        const std::vector< CMeasurementPose >& p_vecMeasurements );

};

#endif //CBRIDGEG2O_H
