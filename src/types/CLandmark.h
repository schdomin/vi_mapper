#ifndef CLANDMARK_H
#define CLANDMARK_H

#include "types/Typedefs.h"

class CLandmark
{

public:

    CLandmark( const UIDLandmark& p_uID,
               const CDescriptor& p_matDescriptor,
               const double& p_dKeyPointSize,
               const CPoint3DInWorldFrame& p_vecPointXYZ,
               const CPoint2DInCameraFrameHomogenized& p_cPositionUVLEFTReference,
               const cv::Point2d& p_ptUVLEFT,
               const cv::Point2d& p_ptUVRIGHT,
               const CPoint3DInCameraFrame& p_vecPointXYZCamera,
               const Eigen::Vector3d& p_vecCameraPosition,
               const Eigen::Matrix3d& p_matKRotation,
               const Eigen::Vector3d& p_vecKTranslation );

    ~CLandmark( );

public:

    const UIDLandmark uID;

    const CDescriptor matDescriptorReference;
    CDescriptor matDescriptorLast;
    const double dKeyPointSize;

    CPoint3DInWorldFrame vecPointXYZCalibrated;
    const CPoint2DInCameraFrameHomogenized vecUVLEFTReference;

    uint8_t uFailedSubsequentTrackings;
    uint32_t uCalibrations;
    double dCurrentAverageSquaredError;

private:

    std::vector< CMeasurementLandmark* > m_vecMeasurements;

    //ds last camera position for calibration
    Eigen::Vector3d m_vecLastCameraPosition;

    //ds LM LS configuration
    const double m_dDistanceDeltaForCalibration = 0.25;
    const uint32_t m_uIterations                = 100;
    const double m_dLevenbergDamping            = 10.0;
    //const double m_dFactorDamping               = 1.05;
    const double m_dConvergenceDelta            = 1e-5;
    const double m_dMaximumError                = 25.0;

public:

    //ds add detection
    void addPosition( const cv::Point2d& p_ptUVLEFT,
                      const cv::Point2d& p_ptUVRIGHT,
                      const CPoint3DInCameraFrame& p_vecPointXYZ,
                      const Eigen::Vector3d& p_vecCameraPosition,
                      const Eigen::Matrix3d& p_matKRotation,
                      const Eigen::Vector3d& p_vecKTranslation );

    const cv::Point2d getLastDetectionLEFT( ) const;

    const CMeasurementLandmark* getLastMeasurement( ) const;

private:

    //ds calibrate 3d point
    const CPoint3DInWorldFrame _getOptimizedLandmarkLMA( const CPoint3DInWorldFrame& p_vecInitialGuess );

};

#endif //CLANDMARK_H
