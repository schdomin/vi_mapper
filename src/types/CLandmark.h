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
               const cv::Point2d& p_ptUVLEFT,
               const cv::Point2d& p_ptUVRIGHT,
               const CPoint3DCAMERA& p_vecPointXYZCamera,
               const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
               const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
               const MatrixProjection& p_matProjectionLEFT,
               const MatrixProjection& p_matProjectionRIGHT,
               const MatrixProjection& p_matProjectionWORLDtoLEFT,
               const MatrixProjection& p_matProjectionWORLDtoRIGHT,
               const UIDFrame& p_uFrame );

    ~CLandmark( );

public:

    const UIDLandmark uID;
    const CDescriptor matDescriptorReferenceLEFT;
    const CDescriptor matDescriptorReferenceRIGHT;
    const double dKeyPointSize;

    const CPoint3DWORLD vecPointXYZInitial;
    CPoint3DWORLD vecPointXYZOptimized;
    const CPoint2DHomogenized vecUVReferenceLEFT;

    //ds optimization
    uint8_t uFailedSubsequentTrackings = 0;
    uint32_t uOptimizationsSuccessful  = 0;
    uint32_t uOptimizationsFailed      = 0;
    double dCurrentAverageSquaredError = 0.0;
    bool bIsOptimal                    = false;
    CPoint3DWORLD vecPointXYZMean;

    //ds mapping
    bool bIsCurrentlyVisible            = false;
    uint32_t uNumberOfKeyFramePresences = 0;

    //ds needed for cloud matching
    std::vector< CDescriptor > vecDescriptorsLEFT;
    std::vector< CDescriptor > vecDescriptorsRIGHT;

private:

    //ds all measurements of this landmark
    std::vector< CMeasurementLandmark* > m_vecMeasurements;

    //ds last camera position for calibration
    //CPoint3DWORLD m_vecCameraPositionLAST;
    //Eigen::Vector3d m_vecCameraOrientationAccumulated;

    //ds projection matrices (used for optimization)
    const MatrixProjection m_matProjectionLEFT;
    const MatrixProjection m_matProjectionRIGHT;

    //ds debug logging
    //std::FILE* m_pFilePositionOptimization;

//ds public for logging purposes
public:

    //ds optimization configuration (shared among all landmarks)
    //static constexpr double m_dDistanceDeltaForOptimizationMeters = 0.1; //ds squared measurement
    //static constexpr double m_dAngleDeltaForOptimizationRadians   = 0.5; //ds squared measurement
    static constexpr uint32_t uCapIterations                  = 100;
    //static constexpr double m_dLevenbergDamping                   = 5.0;
    //const double m_dFactorDynamicDamping        = 1.05;
    static constexpr double dConvergenceDelta                 = 1e-5;
    static constexpr double dKernelMaximumErrorSquaredPixels  = 10.0;
    static constexpr double dMaximumErrorSquaredAveragePixels = 9.0;
    //static constexpr uint8_t m_uMinimumInliers                    = 10;

public:

    //ds add detection
    void addMeasurement( const UIDFrame& p_uFrame,
                         const cv::Point2d& p_ptUVLEFT,
                         const cv::Point2d& p_ptUVRIGHT,
                         const CDescriptor& p_matDescriptorLEFT,
                         const CDescriptor& p_matDescriptorRIGHT,
                         const CPoint3DCAMERA& p_vecPointXYZ,
                         const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                         const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                         const MatrixProjection& p_matProjectionWORLDtoLEFT,
                         const MatrixProjection& p_matProjectionWORLDtoRIGHT );

    //ds getters ONLY used for wrapping
    const cv::Point2d getLastDetectionLEFT( ) const { return m_vecMeasurements.back( )->ptUVLEFT; }
    const cv::Point2d getLastDetectionRIGHT( ) const { return m_vecMeasurements.back( )->ptUVRIGHT; }
    const CDescriptor getLastDescriptorLEFT( ) const { return vecDescriptorsLEFT.back( ); }
    const CDescriptor getLastDescriptorRIGHT( ) const { return vecDescriptorsRIGHT.back( ); }
    const CMeasurementLandmark* getLastMeasurement( ) const { return m_vecMeasurements.back( ); }
    const double getLastDepth( ) const { return m_vecMeasurements.back( )->vecPointXYZLEFT.z( ); }
    const CPoint3DCAMERA getLastPointXYZLEFT( ) const { return m_vecMeasurements.back( )->vecPointXYZLEFT; }
    const std::vector< CMeasurementLandmark* >::size_type getNumberOfMeasurements( ) const { return m_vecMeasurements.size( ); }
    void optimize( const UIDFrame& p_uFrame );

private:

    //ds calibrate 3d point
    const CPoint3DWORLD _getOptimizedLandmarkLEFT3D( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess );
    const CPoint3DWORLD _getOptimizedLandmarkSTEREOUV( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess );
    const CPoint3DWORLD _getOptimizedLandmarkIDWA( );

};

#endif //CLANDMARK_H
