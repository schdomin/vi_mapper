#ifndef CLANDMARK_H
#define CLANDMARK_H

#include "types/Typedefs.h"

class CLandmark
{

//ds eigen memory alignment
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:

    CLandmark( const UIDLandmark& p_uID,
               const CDescriptor& p_matDescriptorLEFT,
               const CDescriptor& p_matDescriptorRIGHT,
               const double& p_dKeyPointSize,
               const CPoint3DInWorldFrame& p_vecPointXYZ,
               const CPoint2DInCameraFrameHomogenized& p_cPositionUVLEFTReference,
               const cv::Point2d& p_ptUVLEFT,
               const cv::Point2d& p_ptUVRIGHT,
               const CPoint3DInCameraFrame& p_vecPointXYZCamera,
               const CPoint3DInWorldFrame& p_vecCameraPosition,
               const Eigen::Vector3d& p_vecCameraOrientation,
               //const Eigen::Matrix3d& p_matKRotation,
               //const Eigen::Vector3d& p_vecKTranslation,
               const MatrixProjection& p_matProjectionWORLDtoLEFT,
               const uint64_t& p_uFrame );

    ~CLandmark( );

public:

    const UIDLandmark uID;

    const CDescriptor matDescriptorReferenceLEFT;
    CDescriptor matDescriptorLASTLEFT;
    CDescriptor matDescriptorLASTRIGHT;
    const double dKeyPointSize;

    const CPoint3DInWorldFrame vecPointXYZInitial;
    CPoint3DInWorldFrame vecPointXYZOptimized;
    const CPoint2DInCameraFrameHomogenized vecUVLEFTReference;
    const CPoint2DHomogenized vecUVReferenceLEFT;

    uint8_t uFailedSubsequentTrackings;
    uint32_t uOptimizationsSuccessful;
    uint32_t uOptimizationsFailed;
    double dCurrentAverageSquaredError;
    CPoint3DInWorldFrame vecPointXYZMean;

    bool bIsCurrentlyVisible;

private:

    //ds all measurements of this landmark
    std::vector< CMeasurementLandmark* > m_vecMeasurements;

    double m_dDepthLastOptimizationMeters;

    //ds last camera position for calibration
    CPoint3DInWorldFrame m_vecCameraPositionLAST;
    Eigen::Vector3d m_vecCameraOrientationAccumulated;

//ds public for logging purposes
public:

    //ds optimization configuration (shared among all landmarks)
    static constexpr double m_dDistanceDeltaForOptimizationMeters = 0.25; //ds squared measurement
    static constexpr double m_dAngleDeltaForOptimizationRadians   = 0.5; //ds squared measurement
    static constexpr uint32_t m_uCapIterations                    = 500;
    static constexpr double m_dLevenbergDamping                   = 5.0;
    //const double m_dFactorDynamicDamping        = 1.05;
    static constexpr double m_dConvergenceDelta                   = 1e-5;
    static constexpr double m_dKernelMaximumError                 = 25.0;

private:

    //ds debug logging
    std::FILE* m_pFilePositionOptimization;

public:

    //ds add detection
    void addMeasurement( const uint64_t& p_uFrame,
                      const cv::Point2d& p_ptUVLEFT,
                      const cv::Point2d& p_ptUVRIGHT,
                      const CPoint3DInCameraFrame& p_vecPointXYZ,
                      const CPoint3DInWorldFrame& p_vecPointXYZWORLD,
                      const CPoint3DInWorldFrame& p_vecCameraPosition,
                      const Eigen::Vector3d& p_vecCameraOrientation,
                      //const Eigen::Matrix3d& p_matKRotation,
                      //const Eigen::Vector3d& p_vecKTranslation,
                      const MatrixProjection& p_matProjectionWORLDtoLEFT );

    const cv::Point2d getLastDetectionLEFT( ) const { return m_vecMeasurements.back( )->ptUVLEFT; }
    const cv::Point2d getLastDetectionRIGHT( ) const { return m_vecMeasurements.back( )->ptUVRIGHT; }
    const CMeasurementLandmark* getLastMeasurement( ) const { return m_vecMeasurements.back( ); }
    const std::vector< CMeasurementLandmark* >::size_type getNumberOfMeasurements( ) const { return m_vecMeasurements.size( ); }

private:

    //ds calibrate 3d point
    const CPoint3DInWorldFrame _getOptimizedLandmarkKLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess );
    const CPoint3DInWorldFrame _getOptimizedLandmarkIDLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess );
    const CPoint3DInWorldFrame _getOptimizedLandmarkKRDLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess );
    const CPoint3DInWorldFrame _getOptimizedLandmarkIDWA( );

};

#endif //CLANDMARK_H
