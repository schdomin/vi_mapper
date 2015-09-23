#ifndef CG2OOPTIMIZER_H
#define CG2OOPTIMIZER_H

#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/core/robust_kernel_impl.h"

#include "vision/CStereoCamera.h"
#include "types/CLandmark.h"
#include "types/CKeyFrame.h"

//#include "closure_buffer.h"
//#include "closure_checker.h"

class Cg2oOptimizer
{

    enum EG2OParameterID
    {
        eWORLD        = 0,
        eCAMERA_LEFT  = 1,
        eCAMERA_RIGHT = 2,
        eOFFSET_IMUtoLEFT = 3
    };

public:

    Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                   const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                   const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames,
                   const Eigen::Isometry3d& p_matTransformationLEFTtoWORLDInitial );
    Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                   const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                   const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames );
    ~Cg2oOptimizer( );

private:

    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;
    const std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;
    const std::shared_ptr< std::vector< CKeyFrame* > > m_vecKeyFrames;

    g2o::SparseOptimizer m_cOptimizerSparse;
    const UIDKeyFrame m_uIDShift      = 1000000; //ds required to navigate between landmarks and poses
    g2o::VertexSE3 *m_pVertexPoseLAST = 0;
    const uint32_t m_uIterations      = 100;
    uint32_t m_uOptimizations         = 0;

    const double m_dMaximumReliableDepthForPointXYZ  = 2.5;
    const double m_dMaximumReliableDepthForUVDepth   = 7.5;

    //ds optimized structures
    std::vector< CLandmark* > m_vecLandmarksInGraph;
    std::vector< CKeyFrame* > m_vecKeyFramesInGraph;

    //ds currently movable poses
    std::vector< g2o::VertexSE3* > m_vecActivePosesInGraph;

//ds information matrices ground structures
private:

    const Eigen::Matrix< double, 6, 6 > m_matInformationPose;
    const Eigen::Matrix< double, 6, 6 > m_matInformationLoopClosure;
    const Eigen::Matrix< double, 3, 3 > m_matInformationLandmarkClosure;

public:

    void optimizeTailLoopClosuresOnly( const UIDKeyFrame& p_uIDBeginKeyFrame );
    void optimizeTail( const UIDKeyFrame& p_uIDBeginKeyFrame );
    void optimizeContinuous( const UIDKeyFrame& p_uIDBeginKeyFrame );

    const uint32_t getNumberOfSegmentOptimizations( ) const { return m_uOptimizations; }

    //ds clears g2o files in logging directory
    void clearFiles( ) const;

private:

    g2o::EdgeSE3LinearAcceleration* _getEdgeLinearAcceleration( g2o::VertexSE3* p_pVertexPose,
                                                                const CLinearAccelerationIMU& p_vecLinearAccelerationNormalized ) const;
    g2o::EdgeSE3PointXYZ* _getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                            g2o::VertexPointXYZ* p_pVertexLandmark,
                                            const CPoint3DWORLD& p_vecPointXYZ,
                                            const double& p_dInformationFactor ) const;
    g2o::EdgeSE3PointXYZDepth* _getEdgeUVDepthLEFT( g2o::VertexSE3* p_pVertexPose,
                                                    g2o::VertexPointXYZ* p_pVertexLandmark,
                                                    const CMeasurementLandmark* p_pMeasurement,
                                                    const double& p_dInformationFactor ) const;
    g2o::EdgeSE3PointXYZDisparity* _getEdgeUVDisparityLEFT( g2o::VertexSE3* p_pVertexPose,
                                                            g2o::VertexPointXYZ* p_pVertexLandmark,
                                                            const double& p_dDisparityPixels,
                                                            const CMeasurementLandmark* p_pMeasurement,
                                                            const double& p_dFxPixels,
                                                            const double& p_dBaselineMeters,
                                                            const double& p_dInformationFactor ) const;

    void _loadLandmarksToGraph( const std::vector< CLandmark* >& p_vecChunkLandmarks );
    g2o::VertexSE3* _setAndgetPose( g2o::VertexSE3* p_pVertexPoseFrom, const CKeyFrame* pKeyFrameCurrent );
    void _setLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent, const CKeyFrame::CMatchICP* p_pClosure );
    void _setLandmarkMeasurementsWORLD( g2o::VertexSE3* p_pVertexPoseCurrent,
                           const CKeyFrame* pKeyFrameCurrent,
                           UIDLandmark& p_uMeasurementsStoredPointXYZ,
                           UIDLandmark& p_uMeasurementsStoredUVDepth,
                           UIDLandmark& p_uMeasurementsStoredUVDisparity );

    void _applyOptimization( const std::vector< CLandmark* >& p_vecChunkLandmarks );
    void _applyOptimization( const std::vector< CKeyFrame* >& p_vecChunkKeyFrames );
    void _fixateTrajectory( );


};

#endif //CG2OOPTIMIZER_H
