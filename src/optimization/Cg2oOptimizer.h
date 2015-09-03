#ifndef CG2OOPTIMIZER_H
#define CG2OOPTIMIZER_H

#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "vision/CStereoCamera.h"
#include "types/CLandmark.h"
#include "types/CKeyFrame.h"

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
    const uint32_t m_uIterations      = 500;
    uint32_t m_uOptimizations         = 0;

    const double m_dMaximumReliableDepthForPointXYZ = 2.5;
    const double m_dMaximumReliableDepthForUVDepth  = 7.5;
    static constexpr uint8_t m_uMinimumOptimizations           = 0;
    static constexpr double m_dMaximumErrorPerOptimization     = 10.0; //ds e.g after 3 optimizations an error of 30.0 is allowed
    static constexpr uint8_t m_uMinimumKeyFramePresences       = 1;

    //ds optimized structures
    std::vector< CLandmark* > m_vecLandmarksInGraph;
    std::vector< CKeyFrame* > m_vecKeyFramesInGraph;

public:

    void optimizeTailLoopClosuresOnly( const UIDKeyFrame& p_uIDBeginKeyFrame );
    void optimizeTail( const UIDKeyFrame& p_uIDBeginKeyFrame );
    void optimizeContinuous( const UIDKeyFrame& p_uIDBeginKeyFrame );

    const uint32_t getNumberOfSegmentOptimizations( ) const { return m_uOptimizations; }
    static bool isOptimized( const CLandmark* p_pLandmark );
    static bool isKeyFramed( const CLandmark* p_pLandmark );

    //ds clears g2o files in logging directory
    void clearFiles( ) const;

private:

    g2o::EdgeSE3LinearAcceleration* _getEdgeLinearAcceleration( g2o::VertexSE3* p_pVertexPose,
                                                                const CLinearAccelerationIMU& p_vecLinearAccelerationNormalized ) const;
    g2o::EdgeSE3PointXYZ* _getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                                  g2o::VertexPointXYZ* p_pVertexLandmark,
                                                  const CPoint3DWORLD& p_vecPointXYZ ) const;
    g2o::EdgeSE3PointXYZDepth* _getEdgeUVDepthLEFT( g2o::VertexSE3* p_pVertexPose,
                                                      g2o::VertexPointXYZ* p_pVertexLandmark,
                                                      const CMeasurementLandmark* p_pMeasurement ) const;
    g2o::EdgeSE3PointXYZDisparity* _getEdgeUVDisparityLEFT( g2o::VertexSE3* p_pVertexPose,
                                                              g2o::VertexPointXYZ* p_pVertexLandmark,
                                                              const double& p_dDisparityPixels,
                                                              const CMeasurementLandmark* p_pMeasurement,
                                                              const double& p_dFxPixels,
                                                              const double& p_dBaselineMeters ) const;

    void _loadLandmarksToGraph( const std::vector< CLandmark* >& p_vecChunkLandmarks );
    g2o::VertexSE3* _setAndgetPose( g2o::VertexSE3* p_pVertexPoseFrom, const CKeyFrame* pKeyFrameCurrent );
    void _setLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent );
    void _setLandmarkMeasurements( g2o::VertexSE3* p_pVertexPoseCurrent,
                           const CKeyFrame* pKeyFrameCurrent,
                           UIDLandmark& p_uMeasurementsStoredPointXYZ,
                           UIDLandmark& p_uMeasurementsStoredUVDepth,
                           UIDLandmark& p_uMeasurementsStoredUVDisparity );

    void _applyOptimization( const std::vector< CLandmark* >& p_vecChunkLandmarks );
    void _applyOptimization( const std::vector< CKeyFrame* >& p_vecChunkKeyFrames );


};

#endif //CG2OOPTIMIZER_H