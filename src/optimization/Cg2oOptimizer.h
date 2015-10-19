#ifndef CG2OOPTIMIZER_H
#define CG2OOPTIMIZER_H

#include "g2o/core/sparse_optimizer.h"

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
    const UIDKeyFrame m_uIDShift       = 1000000; //ds required to navigate between landmarks and poses
    g2o::VertexSE3* m_pVertexPoseLAST  = 0;
    const uint32_t m_uIterations       = 1000;
    uint32_t m_uOptimizations          = 0;
    double m_dCurrentTrajectoryWeight  = 1.0;

    const double m_dMaximumReliableDepthForPointXYZ    = 2.5;
    const double m_dMaximumReliableDepthForUVDepth     = 7.5;
    const double m_dMaximumReliableDepthForUVDisparity = 100.0;
    const double m_dMaximumReliableDepthForPointXYZL2    = 10.0;
    const double m_dMaximumReliableDepthForUVDepthL2     = 50.0;
    const double m_dMaximumReliableDepthForUVDisparityL2 = 10000.0;

    //ds optimized structures
    std::vector< CLandmark* > m_vecLandmarksInGraph;
    std::vector< CKeyFrame* > m_vecKeyFramesInGraph;

    //ds first pose vertex (for full graph drawing)
    g2o::VertexSE3* m_pVertexPoseFIRSTNOTINGRAPH = 0;

    //ds stiffness for loop closing
    std::vector< g2o::EdgeSE3* > m_vecPoseEdges;

//ds information matrices ground structures
private:

    const Eigen::Matrix< double, 6, 6 > m_matInformationPose;
    const Eigen::Matrix< double, 6, 6 > m_matInformationLoopClosure;
    const Eigen::Matrix< double, 3, 3 > m_matInformationLandmarkClosure;

public:

    void optimizeTailLoopClosuresOnly( const UIDKeyFrame& p_uIDBeginKeyFrame, const Eigen::Vector3d& p_vecTranslationToG2o );
    void optimizeTail( const UIDKeyFrame& p_uIDBeginKeyFrame );
    void optimizeContinuous( const UIDFrame& p_uFrame, const UIDKeyFrame& p_uIDBeginKeyFrame, const std::vector< CLandmark* >::size_type p_uIDBeginLandmark, const Eigen::Vector3d& p_vecTranslationToG2o );

    const uint32_t getNumberOfOptimizations( ) const { return m_uOptimizations; }

    //ds clears g2o files in logging directory
    void clearFiles( ) const;

    //ds saves final graph
    void saveFinalGraph( const UIDFrame& p_uFrame, const Eigen::Vector3d& p_vecTranslationToG2o );

    //ds manual loop closing
    void updateLoopClosuresFrom( const std::vector< CKeyFrame* >::size_type& p_uIDBeginKeyFrame, const Eigen::Vector3d& p_vecTranslationToG2o );

    /*ds first pose
    void updateSTART( const Eigen::Vector3d& p_vecTranslationWORLD )
    {
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itPoseSTART( m_cOptimizerSparse.vertices( ).find( m_uIDShift-1 ) );

        //ds has to be found
        assert( itPoseSTART != m_cOptimizerSparse.vertices( ).end( ) );

        //ds extract and update the estimate
        g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPoseSTART->second );
        Eigen::Isometry3d matPoseSTARTShifted( pVertex->estimate( ) );
        matPoseSTARTShifted.translation( ) -= p_vecTranslationWORLD;
        pVertex->setEstimate( matPoseSTARTShifted );
    }

    //ds change vertex
    void updateEstimate( const CKeyFrame* p_pKeyFrame )
    {
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itKeyFrame( m_cOptimizerSparse.vertices( ).find( p_pKeyFrame->uID+m_uIDShift ) );

        //ds if found
        if( itKeyFrame != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds extract and update the estimate
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itKeyFrame->second );
            pVertex->setEstimate( p_pKeyFrame->matTransformationLEFTtoWORLD );
        }
    }
    void updateEstimate( const CLandmark* p_pLandmark )
    {
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( p_pLandmark->uID ) );

        //ds if found
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds extract and update the estimate
            g2o::VertexPointXYZ* pVertex = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );
            pVertex->setEstimate( p_pLandmark->vecPointXYZOptimized );
        }
    }*/

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

    void _loadLandmarksToGraph( const std::vector< CLandmark* >::size_type& p_uIDLandmark, const Eigen::Vector3d& p_vecTranslationToG2o );
    g2o::VertexSE3* _setAndgetPose( g2o::VertexSE3* p_pVertexPoseFrom, CKeyFrame* pKeyFrameCurrent, const Eigen::Vector3d& p_vecTranslationToG2o );
    void _setLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent, const CKeyFrame::CMatchICP* p_pClosure, const Eigen::Vector3d& p_vecTranslationToG2o );
    void _setLandmarkMeasurementsWORLD( g2o::VertexSE3* p_pVertexPoseCurrent,
                           const CKeyFrame* pKeyFrameCurrent,
                           UIDLandmark& p_uMeasurementsStoredPointXYZ,
                           UIDLandmark& p_uMeasurementsStoredUVDepth,
                           UIDLandmark& p_uMeasurementsStoredUVDisparity );

    void _applyOptimization( const UIDFrame& p_uFrame, const std::vector< CLandmark* >::size_type& p_uIDLandmark, const Eigen::Vector3d& p_vecTranslationToG2o );
    void _applyOptimization( const Eigen::Vector3d& p_vecTranslationToG2o );


};

#endif //CG2OOPTIMIZER_H
