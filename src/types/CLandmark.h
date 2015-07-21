#ifndef CLANDMARK_H
#define CLANDMARK_H

#include "types/Typedefs.h"

class CLandmark
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CLandmark( const UIDLandmark& p_uID,
               const CDescriptor& p_matDescriptorLEFT,
               const CDescriptor& p_matDescriptorRIGHT,
               const double& p_dKeyPointSize,
               const CPoint3DInWorldFrame& p_vecPointXYZ,
               const CPoint2DInCameraFrameHomogenized& p_cPositionUVLEFTReference,
               const cv::Point2d& p_ptUVLEFT,
               const cv::Point2d& p_ptUVRIGHT,
               const CPoint3DInCameraFrame& p_vecPointXYZCamera,
               const Eigen::Vector3d& p_vecCameraPosition,
               //const Eigen::Matrix3d& p_matKRotation,
               //const Eigen::Vector3d& p_vecKTranslation,
               const MatrixProjection& p_matProjectionWORLDtoLEFT,
               const uint64_t& p_uFrame );

    ~CLandmark( );

public:

    const UIDLandmark uID;

    const CDescriptor matDescriptorReference;
    CDescriptor matDescriptorLastLEFT;
    CDescriptor matDescriptorLastRIGHT;
    const double dKeyPointSize;

    CPoint3DInWorldFrame vecPointXYZInitial;
    CPoint3DInWorldFrame vecPointXYZCalibrated;
    const CPoint2DInCameraFrameHomogenized vecUVLEFTReference;

    uint8_t uFailedSubsequentTrackings;
    uint32_t uCalibrations;
    double dCurrentAverageSquaredError;
    CPoint3DInWorldFrame vecMeanMeasurement;

    bool bIsCurrentlyVisible;

private:

    //ds all measurements of this landmark
    std::vector< CMeasurementLandmark* > m_vecMeasurements;

    //ds last camera position for calibration
    Eigen::Vector3d m_vecLastCameraPosition;

    //ds LM LS configuration
    const double m_dDistanceDeltaForCalibration = 0.25;
    const uint32_t m_uIterations                = 100;
    const double m_dLevenbergDamping            = 5.0;
    //const double m_dFactorDamping               = 1.05;
    const double m_dConvergenceDelta            = 1e-5;
    const double m_dMaximumError                = 25.0;

    //ds debug logging
    std::FILE* m_pFilePosition;

public:

    //ds add detection
    void addPosition( const uint64_t& p_uFrame,
                      const cv::Point2d& p_ptUVLEFT,
                      const cv::Point2d& p_ptUVRIGHT,
                      const CPoint3DInCameraFrame& p_vecPointXYZ,
                      const CPoint3DInWorldFrame& p_vecPointXYZWORLD,
                      const Eigen::Vector3d& p_vecCameraPosition,
                      //const Eigen::Matrix3d& p_matKRotation,
                      //const Eigen::Vector3d& p_vecKTranslation,
                      const MatrixProjection& p_matProjectionWORLDtoLEFT );

    const cv::Point2d getLastDetectionLEFT( ) const;
    const cv::Point2d getLastDetectionRIGHT( ) const;

    const CMeasurementLandmark* getLastMeasurement( ) const;
    const std::vector< CMeasurementLandmark* >::size_type getNumberOfMeasurements( ) const { return m_vecMeasurements.size( ); }

private:

    //ds calibrate 3d point
    const CPoint3DInWorldFrame _getOptimizedLandmarkKLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess );
    const CPoint3DInWorldFrame _getOptimizedLandmarkIDLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess );
    const CPoint3DInWorldFrame _getOptimizedLandmarkIDWA( );

};

#endif //CLANDMARK_H
