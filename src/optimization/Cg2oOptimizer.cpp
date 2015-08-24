#include "Cg2oOptimizer.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "configuration/CConfigurationCamera.h"

Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames,
                              const Eigen::Isometry3d& p_matTransformationLEFTtoWORLDInitial ): m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                                       m_vecLandmarks( p_vecLandmarks ),
                                                                                       m_vecKeyFrames( p_vecKeyFrames )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );

    //ds configure the solver (var_cholmod)
    m_cOptimizerSparse.setVerbose( true );
    g2o::BlockSolverX::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >( ) );
    g2o::BlockSolverX* pSolver( new g2o::BlockSolverX( pLinearSolver ) );
    //g2o::OptimizationAlgorithmGaussNewton* pAlgorithm( new g2o::OptimizationAlgorithmGaussNewton( pSolver ) );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm( new g2o::OptimizationAlgorithmLevenberg( pSolver ) );
    m_cOptimizerSparse.setAlgorithm( pAlgorithm );

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    m_cOptimizerSparse.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_pCameraLEFT->m_dFyP, m_pCameraSTEREO->m_pCameraLEFT->m_dCxP, m_pCameraSTEREO->m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    m_cOptimizerSparse.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( m_pCameraSTEREO->m_pCameraRIGHT->m_dFxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dFyP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCyP );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    m_cOptimizerSparse.addParameter( pCameraParametersRIGHT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( m_pCameraSTEREO->m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    m_cOptimizerSparse.addParameter( pOffsetIMUtoLEFT );

    //ds add the first pose separately
    g2o::VertexSE3* m_pVertexPoseLAST = new g2o::VertexSE3( );
    m_pVertexPoseLAST->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseLAST->setId( m_uIDShift-1 );
    m_pVertexPoseLAST->setFixed( true );
    m_cOptimizerSparse.addVertex( m_pVertexPoseLAST );

    CLogger::openBox( );
    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) iterations: %u\n", m_uIterations );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement PointXYZ: %f\n", m_dMaximumReliableDepthForPointXYZ );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement UVDepth: %f\n", m_dMaximumReliableDepthForUVDepth );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum optimization error (landmark): %f\n", m_dMaximumErrorPerOptimization );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames ): m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                                       m_vecLandmarks( p_vecLandmarks ),
                                                                                       m_vecKeyFrames( p_vecKeyFrames )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );

    //ds configure the solver (var_cholmod)
    m_cOptimizerSparse.setVerbose( true );
    g2o::BlockSolverX::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >( ) );
    g2o::BlockSolverX* pSolver( new g2o::BlockSolverX( pLinearSolver ) );
    //g2o::OptimizationAlgorithmGaussNewton* pAlgorithm( new g2o::OptimizationAlgorithmGaussNewton( pSolver ) );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm( new g2o::OptimizationAlgorithmLevenberg( pSolver ) );
    m_cOptimizerSparse.setAlgorithm( pAlgorithm );

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    m_cOptimizerSparse.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_pCameraLEFT->m_dFyP, m_pCameraSTEREO->m_pCameraLEFT->m_dCxP, m_pCameraSTEREO->m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    m_cOptimizerSparse.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( m_pCameraSTEREO->m_pCameraRIGHT->m_dFxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dFyP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCyP );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    m_cOptimizerSparse.addParameter( pCameraParametersRIGHT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( m_pCameraSTEREO->m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    m_cOptimizerSparse.addParameter( pOffsetIMUtoLEFT );

    CLogger::openBox( );
    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) iterations: %u\n", m_uIterations );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement PointXYZ: %f\n", m_dMaximumReliableDepthForPointXYZ );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement UVDepth: %f\n", m_dMaximumReliableDepthForUVDepth );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum optimization error (landmark): %f\n", m_dMaximumErrorPerOptimization );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

Cg2oOptimizer::~Cg2oOptimizer( )
{
    //ds nothing to do
}

void Cg2oOptimizer::optimizeTail( const UIDKeyFrame& p_uIDBeginKeyFrame )
{
    //ds clear nodes and edges
    m_cOptimizerSparse.clear( );

    //ds local optimization chunks: +1 at the end because std::vector[begin,end)
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );
    std::vector< CLandmark* > vecChunkLandmarks( m_vecLandmarks->begin( )+vecChunkKeyFrames.front( )->vecMeasurements.front( )->uID, m_vecLandmarks->end( ) );

    //ds info
    UIDLandmark uDroppedLandmarksInvalid    = 0;
    UIDLandmark uDroppedLandmarksKeyFraming = 0;

    //ds add landmarks
    for( CLandmark* pLandmark: vecChunkLandmarks )
    {
        //ds check if optimiziation criterias are met
        if( isOptimized( pLandmark ) )
        {
            if( isKeyFramed( pLandmark ) )
            {
                //ds set landmark vertex
                g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
                pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
                pVertexLandmark->setId( pLandmark->uID );

                //ds add vertex to optimizer
                m_cOptimizerSparse.addVertex( pVertexLandmark );
            }
            else
            {
                ++uDroppedLandmarksKeyFraming;
            }
        }
        else
        {
            ++uDroppedLandmarksInvalid;
        }
    }

    std::printf( "<Cg2oOptimizer>(optimizeTail) dropped landmarks due to invalid optimization: %lu (%4.2f)\n", uDroppedLandmarksInvalid, static_cast< double >( uDroppedLandmarksInvalid )/vecChunkLandmarks.size( ) );
    std::printf( "<Cg2oOptimizer>(optimizeTail) dropped landmarks due to missing key frame presence: %lu (%4.2f)\n", uDroppedLandmarksKeyFraming, static_cast< double >( uDroppedLandmarksKeyFraming )/vecChunkLandmarks.size( ) );

    //ds add the first pose separately
    g2o::VertexSE3* pVertexPoseFrom = new g2o::VertexSE3( );
    pVertexPoseFrom->setEstimate( vecChunkKeyFrames.front( )->matTransformationLEFTtoWORLD );
    pVertexPoseFrom->setId( vecChunkKeyFrames.front( )->uID+m_uIDShift );
    pVertexPoseFrom->setFixed( true );
    m_cOptimizerSparse.addVertex( pVertexPoseFrom );

    //ds always save acceleration data
    //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseInitial, vecChunkKeyFrames.front( )->vecLinearAccelerationNormalized ) );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;
    uint32_t    uLoopClosures                  = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame* >::size_type u = 1; u < vecChunkKeyFrames.size( ); ++u )
    {
        //ds buffer current keyframe
        const CKeyFrame* pKeyFrame( vecChunkKeyFrames[u] );

        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent( new g2o::VertexSE3( ) );
        pVertexPoseCurrent->setEstimate( pKeyFrame->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( pKeyFrame->uID+m_uIDShift );
        m_cOptimizerSparse.addVertex( pVertexPoseCurrent );

        //ds set up the edge to connect the poses
        g2o::EdgeSE3* pEdgePoseFromTo( new g2o::EdgeSE3( ) );

        //ds set viewpoints and measurement
        pEdgePoseFromTo->setVertex( 0, pVertexPoseFrom );
        pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
        pEdgePoseFromTo->setMeasurement( pVertexPoseFrom->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

        //ds information quality
        const double arrInformationMatrixPose[36] = { 100,0,0,0,0,0,
                                                      0,100,0,0,0,0,
                                                      0,0,100,0,0,0,
                                                      0,0,0,10000,0,0,
                                                      0,0,0,0,10000,0,
                                                      0,0,0,0,0,10000 };
        pEdgePoseFromTo->setInformation( Eigen::Matrix< double, 6, 6 >( arrInformationMatrixPose ) );

        //ds add to optimizer
        m_cOptimizerSparse.addEdge( pEdgePoseFromTo );

        //ds check if we got a loop closure for this frame
        if( 0 != pKeyFrame->pLoopClosure )
        {
            //ds find the corresponding pose
            const g2o::HyperGraph::VertexIDMap::iterator itLoopClosure( m_cOptimizerSparse.vertices( ).find( pKeyFrame->pLoopClosure->uID+m_uIDShift ) );

            //ds if found
            if( itLoopClosure != m_cOptimizerSparse.vertices( ).end( ) )
            {
                //ds extract the pose
                g2o::VertexSE3* pVertexLoopClosure( dynamic_cast< g2o::VertexSE3* >( itLoopClosure->second ) );

                //ds set up the edge
                g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

                //ds set viewpoints and measurement
                pEdgeLoopClosure->setVertex( 0, pVertexPoseCurrent );
                pEdgeLoopClosure->setVertex( 1, pVertexLoopClosure );
                pEdgeLoopClosure->setMeasurement( pVertexPoseCurrent->estimate( ).inverse( )*pVertexLoopClosure->estimate( ) );

                //ds information quality
                const double arrInformationMatrixLoopClosure[36] = { 100,0,0,0,0,0,
                                                                     0,100,0,0,0,0,
                                                                     0,0,100,0,0,0,
                                                                     0,0,0,10000,0,0,
                                                                     0,0,0,0,10000,0,
                                                                     0,0,0,0,0,10000 };
                pEdgeLoopClosure->setInformation( Eigen::Matrix< double, 6, 6 >( arrInformationMatrixLoopClosure ) );

                //ds add to optimizer
                m_cOptimizerSparse.addEdge( pEdgeLoopClosure );

                ++uLoopClosures;
            }
        }

        //ds always save acceleration data
        //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: pKeyFrame->vecMeasurements )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

                //ds get optimized landmark into current pose
                //const CPoint3DCAMERA vecPointOptimized( pVertexPoseCurrent->estimate( ).inverse( )*pVertexLandmark->estimate( ) );

                //ds current depth
                const double dDepthMeters = pMeasurementLandmark->vecPointXYZLEFT.z( );

                /*ds check depth change
                const double dDepthChange = std::fabs( vecPointOptimized.z( )-dDepthMeters );
                if( CBridgeG2O::m_dMaximumReliableDepthForUVDepth > dDepthMeters && 1.0 < dDepthChange )
                {
                    std::printf( "<CBridgeG2O>(saveCOMBO) depth change: %f\n", dDepthChange );
                }*/

                //ds maximum depth to produce a reliable XYZ estimate
                if( m_dMaximumReliableDepthForPointXYZ > dDepthMeters )
                {
                    m_cOptimizerSparse.addEdge( _getEdgePointXYZ( pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark->vecPointXYZLEFT ) );
                    ++uMeasurementsStoredPointXYZ;
                }

                /*ds still good enough for uvdepth
                else if( m_dMaximumReliableDepthForUVDepth > dDepthMeters )
                {
                    //ds get the edge
                    g2o::EdgeSE3PointXYZDepth* pEdgeUVDepth( _getEdgeUVDepthLEFT( pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark ) );

                    //ds set robust kernel
                    //pEdgeUVDepth->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                    //ds projected depth (LEFT camera)
                    m_cOptimizerSparse.addEdge( pEdgeUVDepth );
                    ++uMeasurementsStoredUVDepth;
                }

                //ds go with disparity
                else
                {
                    //ds current disparity
                    const double dDisparity = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;

                    //ds get a disparity edge
                    g2o::EdgeSE3PointXYZDisparity* pEdgeUVDisparity( _getEdgeUVDisparityLEFT( pVertexPoseCurrent, pVertexLandmark, dDisparity, pMeasurementLandmark, m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_dBaselineMeters ) );

                    //ds set robust kernel
                    //pEdgeUVDisparity->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                    //ds disparity (LEFT camera)
                    m_cOptimizerSparse.addEdge( pEdgeUVDisparity );
                    ++uMeasurementsStoredUVDisparity;
                }*/
            }
        }

        //ds update from
        pVertexPoseFrom = pVertexPoseCurrent;
    }

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "<Cg2oOptimizer>(optimizeTail) optimizing [keyframes: %06lu-%06lu landmarks: %06lu-%06lu] (measurements xyz: %lu depth: %lu disparity: %lu, loop closures: %u)\n",
                 vecChunkKeyFrames.front( )->uID,
                 vecChunkKeyFrames.back( )->uID,
                 vecChunkLandmarks.front( )->uID,
                 vecChunkLandmarks.back( )->uID,
                 uMeasurementsStoredPointXYZ,
                 uMeasurementsStoredUVDepth,
                 uMeasurementsStoredUVDisparity,
                 uLoopClosures );

    //ds initialize optimization
    m_cOptimizerSparse.initializeOptimization( );

    //ds do ten iterations
    m_cOptimizerSparse.optimize( m_uIterations );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );

    //ds update landmarks
    for( CLandmark* pLandmark: vecChunkLandmarks )
    {
        //ds find the corresponding landmark
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

        //ds if present
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            const g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

            //ds update position
            pLandmark->vecPointXYZOptimized = pVertexLandmark->estimate( );
        }
    }

    //ds update trajectory
    for( CKeyFrame* pKeyFrame: vecChunkKeyFrames )
    {
        //ds find the corresponding pose
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

        //ds must be present
        assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

        const g2o::VertexSE3* pVertexPose( dynamic_cast< g2o::VertexSE3* >( itPose->second ) );

        //ds update position
        pKeyFrame->matTransformationLEFTtoWORLD = pVertexPose->estimate( );
        pKeyFrame->bIsOptimized                 = true;
    }

    std::printf( "<Cg2oOptimizer>(optimizeTail) optimized poses: %lu and landmarks: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), vecChunkLandmarks.size( ), m_cOptimizerSparse.activeChi2( ) );

    ++m_uOptimizations;
}

void Cg2oOptimizer::optimizeContinuous( const UIDKeyFrame& p_uIDBeginKeyFrame, const UIDKeyFrame& p_uIDEndKeyFrame )
{
    //ds local optimization chunks: +1 at the end because std::vector[begin,end)
    std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->begin( )+p_uIDEndKeyFrame+1 );
    const UIDLandmark uIDBeginLandmark = std::max( vecChunkKeyFrames.front( )->vecMeasurements.front( )->uID, m_uIDLandmarkOptimizationLAST );
    const UIDLandmark uIDEndLandmark   = vecChunkKeyFrames.back( )->vecMeasurements.back( )->uID;
    std::vector< CLandmark* > vecChunkLandmarks( m_vecLandmarks->begin( )+uIDBeginLandmark, m_vecLandmarks->begin( )+uIDEndLandmark+1 );

    //ds info
    UIDLandmark uDroppedLandmarksInvalid    = 0;
    UIDLandmark uDroppedLandmarksKeyFraming = 0;

    //ds add landmarks
    for( CLandmark* pLandmark: vecChunkLandmarks )
    {
        //ds check if optimiziation criterias are met
        if( isOptimized( pLandmark ) )
        {
            if( isKeyFramed( pLandmark ) )
            {
                //ds set landmark vertex
                g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
                pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
                pVertexLandmark->setId( pLandmark->uID );

                //ds add vertex to optimizer
                m_cOptimizerSparse.addVertex( pVertexLandmark );
                m_vecLandmarksInGraph.push_back( pLandmark );
            }
            else
            {
                ++uDroppedLandmarksKeyFraming;
            }
        }
        else
        {
            ++uDroppedLandmarksInvalid;
        }
    }

    std::printf( "<Cg2oOptimizer>(optimizeSegment) dropped landmarks due to invalid optimization: %lu (%4.2f)\n", uDroppedLandmarksInvalid, static_cast< double >( uDroppedLandmarksInvalid )/vecChunkLandmarks.size( ) );
    std::printf( "<Cg2oOptimizer>(optimizeSegment) dropped landmarks due to missing key frame presence: %lu (%4.2f)\n", uDroppedLandmarksKeyFraming, static_cast< double >( uDroppedLandmarksKeyFraming )/vecChunkLandmarks.size( ) );

    //ds add the first pose separately
    g2o::VertexSE3* pVertexPoseInitial = new g2o::VertexSE3( );
    pVertexPoseInitial->setEstimate( vecChunkKeyFrames.front( )->matTransformationLEFTtoWORLD );
    pVertexPoseInitial->setId( vecChunkKeyFrames.front( )->uID+m_uIDShift );
    m_cOptimizerSparse.addVertex( pVertexPoseInitial );

    assert( 0 != m_pVertexPoseLAST );

    //ds set up the edge
    g2o::EdgeSE3* pEdgePoseFromToLAST( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgePoseFromToLAST->setVertex( 0, m_pVertexPoseLAST );
    pEdgePoseFromToLAST->setVertex( 1, pVertexPoseInitial );
    pEdgePoseFromToLAST->setMeasurement( m_pVertexPoseLAST->estimate( ).inverse( )*pVertexPoseInitial->estimate( ) );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgePoseFromToLAST );

    //ds always save acceleration data
    //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseInitial, vecChunkKeyFrames.front( )->vecLinearAccelerationNormalized ) );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;
    uint32_t    uLoopClosures                  = 0;

    m_pVertexPoseLAST = pVertexPoseInitial;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame* >::size_type u = 1; u < vecChunkKeyFrames.size( ); ++u )
    {
        //ds buffer current keyframe
        const CKeyFrame* pKeyFrame( vecChunkKeyFrames[u] );

        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent( new g2o::VertexSE3( ) );
        pVertexPoseCurrent->setEstimate( pKeyFrame->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( pKeyFrame->uID+m_uIDShift );
        m_cOptimizerSparse.addVertex( pVertexPoseCurrent );

        //ds set up the edge
        g2o::EdgeSE3* pEdgePoseFromTo( new g2o::EdgeSE3( ) );

        //ds set viewpoints and measurement
        pEdgePoseFromTo->setVertex( 0, m_pVertexPoseLAST );
        pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
        pEdgePoseFromTo->setMeasurement( m_pVertexPoseLAST->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

        //ds information quality
        const double arrInformationMatrixPose[36] = { 100,0,0,0,0,0,
                                                      0,100,0,0,0,0,
                                                      0,0,100,0,0,0,
                                                      0,0,0,10000,0,0,
                                                      0,0,0,0,10000,0,
                                                      0,0,0,0,0,10000 };
        pEdgePoseFromTo->setInformation( Eigen::Matrix< double, 6, 6 >( arrInformationMatrixPose ) );

        //ds add to optimizer
        m_cOptimizerSparse.addEdge( pEdgePoseFromTo );

        //ds check if we got a loop closure for this frame
        if( 0 != pKeyFrame->pLoopClosure )
        {
            //ds find the corresponding pose
            const g2o::HyperGraph::VertexIDMap::iterator itLoopClosure( m_cOptimizerSparse.vertices( ).find( pKeyFrame->pLoopClosure->uID+m_uIDShift ) );

            //ds if found
            if( itLoopClosure != m_cOptimizerSparse.vertices( ).end( ) )
            {
                //ds extract the pose
                g2o::VertexSE3* pVertexLoopClosure( dynamic_cast< g2o::VertexSE3* >( itLoopClosure->second ) );

                //ds set up the edge
                g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

                //ds set viewpoints and measurement
                pEdgeLoopClosure->setVertex( 0, pVertexPoseCurrent );
                pEdgeLoopClosure->setVertex( 1, pVertexLoopClosure );
                pEdgeLoopClosure->setMeasurement( pVertexPoseCurrent->estimate( ).inverse( )*pVertexLoopClosure->estimate( ) );

                //ds information quality
                const double arrInformationMatrixLoopClosure[36] = { 100,0,0,0,0,0,
                                                                     0,100,0,0,0,0,
                                                                     0,0,100,0,0,0,
                                                                     0,0,0,10000,0,0,
                                                                     0,0,0,0,10000,0,
                                                                     0,0,0,0,0,10000 };
                pEdgeLoopClosure->setInformation( Eigen::Matrix< double, 6, 6 >( arrInformationMatrixLoopClosure ) );

                //ds add to optimizer
                m_cOptimizerSparse.addEdge( pEdgeLoopClosure );

                ++uLoopClosures;
            }
        }

        //ds update last
        m_pVertexPoseLAST = pVertexPoseCurrent;

        //ds always save acceleration data
        //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: pKeyFrame->vecMeasurements )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

                //ds get optimized landmark into current pose
                //const CPoint3DCAMERA vecPointOptimized( pVertexPoseCurrent->estimate( ).inverse( )*pVertexLandmark->estimate( ) );

                //ds current depth
                const double dDepthMeters = pMeasurementLandmark->vecPointXYZLEFT.z( );

                /*ds check depth change
                const double dDepthChange = std::fabs( vecPointOptimized.z( )-dDepthMeters );
                if( CBridgeG2O::m_dMaximumReliableDepthForUVDepth > dDepthMeters && 1.0 < dDepthChange )
                {
                    std::printf( "<CBridgeG2O>(saveCOMBO) depth change: %f\n", dDepthChange );
                }*/

                //ds maximum depth to produce a reliable XYZ estimate
                if( m_dMaximumReliableDepthForPointXYZ > dDepthMeters )
                {
                    m_cOptimizerSparse.addEdge( _getEdgePointXYZ( pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark->vecPointXYZLEFT ) );
                    ++uMeasurementsStoredPointXYZ;
                }

                /*ds still good enough for uvdepth
                else if( m_dMaximumReliableDepthForUVDepth > dDepthMeters )
                {
                    //ds get the edge
                    g2o::EdgeSE3PointXYZDepth* pEdgeUVDepth( _getEdgeUVDepthLEFT( pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark ) );

                    //ds set robust kernel
                    //pEdgeUVDepth->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                    //ds projected depth (LEFT camera)
                    m_cOptimizerSparse.addEdge( pEdgeUVDepth );
                    ++uMeasurementsStoredUVDepth;
                }

                //ds go with disparity
                else
                {
                    //ds current disparity
                    const double dDisparity = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;

                    //ds get a disparity edge
                    g2o::EdgeSE3PointXYZDisparity* pEdgeUVDisparity( _getEdgeUVDisparityLEFT( pVertexPoseCurrent, pVertexLandmark, dDisparity, pMeasurementLandmark, m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_dBaselineMeters ) );

                    //ds set robust kernel
                    //pEdgeUVDisparity->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                    //ds disparity (LEFT camera)
                    m_cOptimizerSparse.addEdge( pEdgeUVDisparity );
                    ++uMeasurementsStoredUVDisparity;
                }*/
            }
        }
    }

    //ds we always process all keyframes
    m_vecKeyFramesInGraph.insert( m_vecKeyFramesInGraph.begin( ), vecChunkKeyFrames.begin( ), vecChunkKeyFrames.end( ) );

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", p_uIDBeginKeyFrame, p_uIDEndKeyFrame );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "<Cg2oOptimizer>(optimizeSegment) optimizing [keyframes: %06lu-%06lu landmarks: %06lu-%06lu] (measurements xyz: %lu depth: %lu disparity: %lu, loop closures: %u)\n", p_uIDBeginKeyFrame, p_uIDEndKeyFrame, uIDBeginLandmark, uIDEndLandmark, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity, uLoopClosures );

    //ds initialize optimization
    m_cOptimizerSparse.initializeOptimization( );

    //ds do ten iterations
    m_cOptimizerSparse.optimize( m_uIterations );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );

    //ds update landmarks
    for( CLandmark* pLandmark: m_vecLandmarksInGraph )
    {
        //ds find the corresponding landmark
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

        //ds must be present
        assert( itLandmark != m_cOptimizerSparse.vertices( ).end( ) );

        const g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

        //ds update position
        pLandmark->vecPointXYZOptimized = pVertexLandmark->estimate( );
    }

    //ds update trajectory
    for( CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds find the corresponding pose
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

        //ds must be present
        assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

        const g2o::VertexSE3* pVertexPose( dynamic_cast< g2o::VertexSE3* >( itPose->second ) );

        //ds update position
        pKeyFrame->matTransformationLEFTtoWORLD = pVertexPose->estimate( );
        pKeyFrame->bIsOptimized                 = true;
    }

    std::printf( "<Cg2oOptimizer>(optimizeSegment) optimized poses: %lu and landmarks: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), m_vecLandmarksInGraph.size( ), m_cOptimizerSparse.activeChi2( ) );

    m_uIDLandmarkOptimizationLAST = uIDEndLandmark;

    ++m_uOptimizations;
}

const bool Cg2oOptimizer::isOptimized( const CLandmark* p_pLandmark ) const
{
    //ds criteria
    return ( m_uMinimumOptimizations < p_pLandmark->uOptimizationsSuccessful ) && ( p_pLandmark->uOptimizationsSuccessful*m_dMaximumErrorPerOptimization > p_pLandmark->dCurrentAverageSquaredError );
}

const bool Cg2oOptimizer::isKeyFramed( const CLandmark* p_pLandmark ) const
{
    //ds criteria
    return ( m_uMinimumKeyFramePresences < p_pLandmark->uNumberOfKeyFramePresences );
}

g2o::EdgeSE3LinearAcceleration* Cg2oOptimizer::_getEdgeLinearAcceleration( g2o::VertexSE3* p_pVertexPose,
                                                                           const CLinearAccelerationIMU& p_vecLinearAccelerationNormalized ) const
{
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAcceleration( new g2o::EdgeSE3LinearAcceleration( ) );

    pEdgeLinearAcceleration->setVertex( 0, p_pVertexPose );
    pEdgeLinearAcceleration->setMeasurement( p_vecLinearAccelerationNormalized );
    pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 100, 0, 0, 0, 100, 0, 0, 0, 100 };
    pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );

    return pEdgeLinearAcceleration;
}

g2o::EdgeSE3PointXYZ* Cg2oOptimizer::_getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                               const CPoint3DWORLD& p_vecPointXYZ ) const
{
    g2o::EdgeSE3PointXYZ* pEdgePointXYZ( new g2o::EdgeSE3PointXYZ( ) );

    //ds triangulated 3d point (uncalibrated)
    pEdgePointXYZ->setVertex( 0, p_pVertexPose );
    pEdgePointXYZ->setVertex( 1, p_pVertexLandmark );
    pEdgePointXYZ->setMeasurement( p_vecPointXYZ );
    pEdgePointXYZ->setParameterId( 0, EG2OParameterID::eWORLD );

    //ds the closer to the camera the point is the more meaningful is the measurement
    //const double dInformationStrengthXYZ( m_dMaximumReliableDepth/( 1+pMeasurementLandmark->vecPointXYZ.z( ) ) );
    const double arrInformationMatrixXYZ[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgePointXYZ->setInformation( g2o::Matrix3D( arrInformationMatrixXYZ ) );

    return pEdgePointXYZ;
}

g2o::EdgeSE3PointXYZDepth* Cg2oOptimizer::_getEdgeUVDepthLEFT( g2o::VertexSE3* p_pVertexPose,
                                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                                               const CMeasurementLandmark* p_pMeasurement ) const
{
    //ds projected depth
    g2o::EdgeSE3PointXYZDepth* pEdgeProjectedDepth( new g2o::EdgeSE3PointXYZDepth( ) );

    pEdgeProjectedDepth->setVertex( 0, p_pVertexPose );
    pEdgeProjectedDepth->setVertex( 1, p_pVertexLandmark );
    pEdgeProjectedDepth->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, p_pMeasurement->vecPointXYZLEFT.z( ) ) );
    pEdgeProjectedDepth->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

    //ds information matrix
    //const double dInformationQualityDepth( m_dMaximumReliableDepth/dDepthMeters );
    const double arrInformationMatrixDepth[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1000 };
    pEdgeProjectedDepth->setInformation( g2o::Matrix3D( arrInformationMatrixDepth ) );

    return pEdgeProjectedDepth;
}

g2o::EdgeSE3PointXYZDisparity* Cg2oOptimizer::_getEdgeUVDisparityLEFT( g2o::VertexSE3* p_pVertexPose,
                                                           g2o::VertexPointXYZ* p_pVertexLandmark,
                                                           const double& p_dDisparityPixels,
                                                           const CMeasurementLandmark* p_pMeasurement,
                                                           const double& p_dFxPixels,
                                                           const double& p_dBaselineMeters ) const
{
    //ds disparity
    g2o::EdgeSE3PointXYZDisparity* pEdgeDisparity( new g2o::EdgeSE3PointXYZDisparity( ) );

    const double dDisparityNormalized( p_dDisparityPixels/( p_dFxPixels*p_dBaselineMeters ) );
    pEdgeDisparity->setVertex( 0, p_pVertexPose );
    pEdgeDisparity->setVertex( 1, p_pVertexLandmark );
    pEdgeDisparity->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, dDisparityNormalized ) );
    pEdgeDisparity->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

    //ds information matrix
    //const double dInformationQualityDisparity( std::abs( dDisparity )+1.0 );
    const double arrInformationMatrixDisparity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1000 };
    pEdgeDisparity->setInformation( g2o::Matrix3D( arrInformationMatrixDisparity ) );

    return pEdgeDisparity;
}
