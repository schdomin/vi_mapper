#ifndef CLANDMARK_H
#define CLANDMARK_H

#include "types/Types.h"

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
               const CPoint3DWORLD& p_vecPointXYZ,
               const CPoint2DInCameraFrameHomogenized& p_cPositionUVLEFTReference,
               const cv::Point2d& p_ptUVLEFT,
               const cv::Point2d& p_ptUVRIGHT,
               const CPoint3DCAMERA& p_vecPointXYZCamera,
               const CPoint3DWORLD& p_vecCameraPosition,
               const Eigen::Vector3d& p_vecCameraOrientation,
               //const Eigen::Matrix3d& p_matKRotation,
               //const Eigen::Vector3d& p_vecKTranslation,
               const MatrixProjection& p_matProjectionWORLDtoLEFT,
               const UIDFrame& p_uFrame );

    ~CLandmark( );

public:

    const UIDLandmark uID;

    const CDescriptor matDescriptorReferenceLEFT;
    CDescriptor matDescriptorLASTLEFT;
    CDescriptor matDescriptorLASTRIGHT;
    const double dKeyPointSize;

    const CPoint3DWORLD vecPointXYZInitial;
    CPoint3DWORLD vecPointXYZOptimized;
    const CPoint2DInCameraFrameHomogenized vecUVLEFTReference;
    const CPoint2DHomogenized vecUVReferenceLEFT;

    uint8_t uFailedSubsequentTrackings = 0;
    uint32_t uOptimizationsSuccessful  = 0;
    uint32_t uOptimizationsFailed      = 0;
    double dCurrentAverageSquaredError = 0.0;
    CPoint3DWORLD vecPointXYZMean;

    bool bIsCurrentlyVisible;
    uint32_t uNumberOfKeyFramePresences = 0;

    std::vector< CDescriptor > vecDescriptorsLEFT;

private:

    //ds all measurements of this landmark
    std::vector< CMeasurementLandmark* > m_vecMeasurements;

    double m_dDepthLastOptimizationMeters;

    //ds last camera position for calibration
    CPoint3DWORLD m_vecCameraPositionLAST;
    Eigen::Vector3d m_vecCameraOrientationAccumulated;

//ds public for logging purposes
public:

    //ds optimization configuration (shared among all landmarks)
    static constexpr double m_dDistanceDeltaForOptimizationMeters = 0.1; //ds squared measurement
    static constexpr double m_dAngleDeltaForOptimizationRadians   = 0.5; //ds squared measurement
    static constexpr uint32_t m_uCapIterations                    = 100;
    static constexpr double m_dLevenbergDamping                   = 5.0;
    //const double m_dFactorDynamicDamping        = 1.05;
    static constexpr double m_dConvergenceDelta                   = 1e-5;
    static constexpr double m_dKernelMaximumError                 = 25.0;
    static constexpr double m_dMaximumErrorSquaredAveragePixels   = 15.0;
    static constexpr uint8_t m_uMinimumInliers                    = 10;

private:

    //ds debug logging
    //std::FILE* m_pFilePositionOptimization;

public:

    //ds add detection
    void addMeasurement( const UIDFrame& p_uFrame,
                      const cv::Point2d& p_ptUVLEFT,
                      const cv::Point2d& p_ptUVRIGHT,
                      const CPoint3DCAMERA& p_vecPointXYZ,
                      const CPoint3DWORLD& p_vecPointXYZWORLD,
                      const CPoint3DWORLD& p_vecCameraPosition,
                      const Eigen::Vector3d& p_vecCameraOrientation,
                      //const Eigen::Matrix3d& p_matKRotation,
                      //const Eigen::Vector3d& p_vecKTranslation,
                      const MatrixProjection& p_matProjectionWORLDtoLEFT,
                      const CDescriptor& p_matDescriptorLEFT );

    const cv::Point2d getLastDetectionLEFT( ) const { return m_vecMeasurements.back( )->ptUVLEFT; }
    const cv::Point2d getLastDetectionRIGHT( ) const { return m_vecMeasurements.back( )->ptUVRIGHT; }
    const CMeasurementLandmark* getLastMeasurement( ) const { return m_vecMeasurements.back( ); }
    const double getLastDepth( ) const { return m_vecMeasurements.back( )->vecPointXYZLEFT.z( ); }
    const CPoint3DCAMERA getLastPointXYZLEFT( ) const { return m_vecMeasurements.back( )->vecPointXYZLEFT; }
    const std::vector< CMeasurementLandmark* >::size_type getNumberOfMeasurements( ) const { return m_vecMeasurements.size( ); }
    void optimize( const UIDFrame& p_uFrame );

private:

    //ds calibrate 3d point
    const CPoint3DWORLD _getOptimizedLandmarkKLMA( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess );
    const CPoint3DWORLD _getOptimizedLandmarkIDLMA( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess );
    const CPoint3DWORLD _getOptimizedLandmarkKRDLMA( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess );
    const CPoint3DWORLD _getOptimizedLandmarkIDWA( );

};

#endif //CLANDMARK_H
