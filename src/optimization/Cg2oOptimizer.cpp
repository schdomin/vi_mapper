#include "Cg2oOptimizer.h"

#include <dirent.h>
#include <stdio.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "configuration/CConfigurationCamera.h"
#include "exceptions/CExceptionLogfileTree.h"

//ds CONTINOUS CONSTRUCTOR
Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames,
                              const Eigen::Isometry3d& p_matTransformationLEFTtoWORLDInitial ): Cg2oOptimizer( p_pCameraSTEREO, p_vecLandmarks, p_vecKeyFrames )
{
    CLogger::openBox( );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod COMPLETE\n" );

    //ds add the first pose separately
    m_pVertexPoseLAST = new g2o::VertexSE3( );
    m_pVertexPoseLAST->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseLAST->setId( m_uIDShift-1 );
    m_pVertexPoseLAST->setFixed( true );

    //ds set to graph
    m_cOptimizerSparse.addVertex( m_pVertexPoseLAST );
    //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( m_pVertexPoseLAST, Eigen::Vector3d( 0.0, -1.0, 0.0 ) ) );

    //ds get a copy
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDShift-1 );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );

    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

//ds TAILWISE CONSTRUCTOR
Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames ): m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                                       m_vecLandmarks( p_vecLandmarks ),
                                                                                       m_vecKeyFrames( p_vecKeyFrames ),
                                                                                       m_matInformationPose( 100000*Eigen::Matrix< double, 6, 6 >::Identity( ) ),
                                                                                       m_matInformationLoopClosure( 10*m_matInformationPose ),
                                                                                       m_matInformationLandmarkClosure( 1000*Eigen::Matrix< double, 3, 3 >::Identity( ) )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );
    m_vecPoseEdges.clear( );

    //ds configure the solver (var_cholmod)
    //m_cOptimizerSparse.setVerbose( true );
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
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod TAILWISE\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) iterations: %u\n", m_uIterations );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement PointXYZ: %f\n", m_dMaximumReliableDepthForPointXYZ );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement UVDepth: %f\n", m_dMaximumReliableDepthForUVDepth );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

Cg2oOptimizer::~Cg2oOptimizer( )
{
    //ds nothing to do
}

void Cg2oOptimizer::optimizeTailLoopClosuresOnly( const UIDKeyFrame& p_uIDBeginKeyFrame, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds clear nodes and edges
    m_cOptimizerSparse.clear( );

    //ds local optimization chunks: +1 at the end because std::vector[begin,end)
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );

    //ds add the first pose separately
    g2o::VertexSE3* pVertexPoseFrom = new g2o::VertexSE3( );
    pVertexPoseFrom->setEstimate( vecChunkKeyFrames.front( )->matTransformationLEFTtoWORLD );
    pVertexPoseFrom->setId( vecChunkKeyFrames.front( )->uID+m_uIDShift );
    pVertexPoseFrom->setFixed( true );
    m_cOptimizerSparse.addVertex( pVertexPoseFrom );

    //ds info
    uint32_t uLoopClosures = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame* >::size_type u = 1; u < vecChunkKeyFrames.size( ); ++u )
    {
        //ds buffer current keyframe
        CKeyFrame* pKeyFrame( vecChunkKeyFrames[u] );

        //ds set the vertex to the graph
        g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( pVertexPoseFrom, pKeyFrame, p_vecTranslationToG2o );

        //ds check if we got loop closures for this frame
        for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
        {
            //ds set it to the graph
            _setLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure, p_vecTranslationToG2o );
            ++uLoopClosures;
        }

        //ds update from
        pVertexPoseFrom = pVertexPoseCurrent;
    }

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "<Cg2oOptimizer>(optimizeTailLoopClosuresOnly) optimizing [keyframes: %06lu-%06lu] (loop closures: %u)\n",
                 vecChunkKeyFrames.front( )->uID,
                 vecChunkKeyFrames.back( )->uID,
                 uLoopClosures );

    //ds initialize optimization
    m_cOptimizerSparse.initializeOptimization( );

    //ds do ten iterations
    m_cOptimizerSparse.optimize( m_uIterations );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );

    //ds update trajectory
    _applyOptimization( p_vecTranslationToG2o );

    //ds done
    std::printf( "<Cg2oOptimizer>(optimizeTailLoopClosuresOnly) optimized poses: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), m_cOptimizerSparse.activeChi2( ) );
    ++m_uOptimizations;
}

void Cg2oOptimizer::optimizeTail( const UIDKeyFrame& p_uIDBeginKeyFrame )
{
    //ds disabled for now
    assert( false );

    /*ds clear nodes and edges
    m_cOptimizerSparse.clear( );

    //ds local optimization chunks: +1 at the end because std::vector[begin,end)
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );
    const std::vector< CLandmark* >::size_type uIDLandmarkFIRST = vecChunkKeyFrames.front( )->vecMeasurements.front( )->uID;

    //ds set landmarks
    _loadLandmarksToGraph( uIDLandmarkFIRST );

    //ds add the first pose separately and fix it
    const CKeyFrame* pKeyFrameInitial = vecChunkKeyFrames.front( );
    g2o::VertexSE3* pVertexPoseFrom = new g2o::VertexSE3( );
    pVertexPoseFrom->setEstimate( pKeyFrameInitial->matTransformationLEFTtoWORLD );
    pVertexPoseFrom->setId( pKeyFrameInitial->uID+m_uIDShift );
    pVertexPoseFrom->setFixed( true );
    m_cOptimizerSparse.addVertex( pVertexPoseFrom );

    //ds always save acceleration data
    m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseFrom, vecChunkKeyFrames.front( )->vecLinearAccelerationNormalized ) );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;
    uint32_t    uLoopClosures                  = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame* >::size_type u = 1; u < vecChunkKeyFrames.size( ); ++u )
    {
        //ds buffer current keyframe
        CKeyFrame* pKeyFrame( vecChunkKeyFrames[u] );

        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( pVertexPoseFrom, pKeyFrame );

        //ds check if we got loop closures for this frame
        for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
        {
            //ds set it to the graph
            _setLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure );
            ++uLoopClosures;
        }

        //ds always save acceleration data
        m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

        //ds set landmark measurements
        _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

        //ds update from
        pVertexPoseFrom = pVertexPoseCurrent;
    }

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "<Cg2oOptimizer>(optimizeTail) optimizing [keyframes: %06lu-%06lu landmarks: %06lu-%06lu][measurements xyz: %lu depth: %lu disparity: %lu, loop closures: %u]\n",
                 vecChunkKeyFrames.front( )->uID,
                 vecChunkKeyFrames.back( )->uID,
                 uIDLandmarkFIRST,
                 m_vecLandmarksInGraph.back( )->uID,
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

    //ds update landmarks and trajectory
    _applyOptimization( vecChunkLandmarks );
    _applyOptimization( vecChunkKeyFrames );

    //ds done
    //std::printf( "<Cg2oOptimizer>(optimizeTail) optimized poses: %lu and landmarks: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), vecChunkLandmarks.size( ), m_cOptimizerSparse.activeChi2( ) );
    ++m_uOptimizations;*/
}

void Cg2oOptimizer::optimizeContinuous( const UIDFrame& p_uFrame,
                                        const UIDKeyFrame& p_uIDBeginKeyFrame,
                                        const std::vector< CLandmark* >::size_type p_uIDBeginLandmark,
                                        const Eigen::Vector3d& p_vecTranslationToG2o,
                                        const bool& p_bLoopClosed )
{
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );

    //ds local optimization chunks
    assert( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame < m_vecKeyFrames->end( ) );
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );
    assert( !vecChunkKeyFrames.empty( ) );
    assert( !vecChunkKeyFrames.front( )->vecMeasurements.empty( ) );
    //std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) loading landmarks (total duration: %.2fs)\n", p_uFrame, CLogger::getTimeSeconds( )-dTimeStartSeconds );

    //ds sane input
    ++m_uOptimizations;

    //ds check if we have a loop closure (landmark free optimization)
    if( p_bLoopClosed )
    {
        //ds closure count
        uint32_t uLoopClosures = 0;

        //ds loop over the camera vertices vector
        for( CKeyFrame* pKeyFrame: vecChunkKeyFrames )
        {
            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_pVertexPoseLAST, pKeyFrame, p_vecTranslationToG2o );

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                //ds set it to the graph
                _setLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure, p_vecTranslationToG2o );
                ++uLoopClosures;
            }

            //ds always save acceleration data
            //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds update last
            m_pVertexPoseLAST = pVertexPoseCurrent;
        }

        std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) optimizing LOOP CLOSURES: %u\n", p_uFrame, uLoopClosures );

        //ds initialize optimization
        m_cOptimizerSparse.initializeOptimization( );

        //ds do iterations
        m_cOptimizerSparse.optimize( m_uIterations );

        std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) optimization complete (total duration: %.2fs)\n", p_uFrame, CLogger::getTimeSeconds( )-dTimeStartSeconds );
    }

    //ds set landmarks
    _loadLandmarksToGraph( p_uIDBeginLandmark, p_vecTranslationToG2o );
    //std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) loading keyframes (total duration: %.2fs)\n", p_uFrame, CLogger::getTimeSeconds( )-dTimeStartSeconds );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;

    //ds if loop closed - we dont have to add the frames again
    if( p_bLoopClosed )
    {
        //ds loop over the camera vertices vector
        for( CKeyFrame* pKeyFrame: vecChunkKeyFrames )
        {
            //ds find the corresponding pose in the current graph
            const g2o::HyperGraph::VertexIDMap::iterator itPoseCurrent( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );
            assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseCurrent );

            //ds get current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = dynamic_cast< g2o::VertexSE3* >( itPoseCurrent->second );
            assert( 0 != pVertexPoseCurrent );

            //ds always save acceleration data
            //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );
        }
    }
    else
    {
        //ds loop over the camera vertices vector
        for( CKeyFrame* pKeyFrame: vecChunkKeyFrames )
        {
            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_pVertexPoseLAST, pKeyFrame, p_vecTranslationToG2o );

            //ds always save acceleration data
            //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            //ds update last
            m_pVertexPoseLAST = pVertexPoseCurrent;
        }
    }

    //ds we always process all keyframes
    m_vecKeyFramesInGraph.insert( m_vecKeyFramesInGraph.end( ), vecChunkKeyFrames.begin( ), vecChunkKeyFrames.end( ) );
    //std::printf( "- complete!\n" );

    /*ds if there were loop closures and we are working in large scale
    if( 0 < uLoopClosures && 200 < m_vecPoseEdges.size( ) )
    {
        //ds first index
        const std::vector< g2o::EdgeSE3* >::size_type uStart = m_vecPoseEdges.size( )-200;

        //ds enhance stiffness of recent trajectory
        for( std::vector< g2o::EdgeSE3* >::size_type u = uStart; u < m_vecPoseEdges.size( ); ++u )
        {
            m_vecPoseEdges[u]->setInformation( static_cast< Eigen::Matrix< double, 6, 6 > >( 100*m_vecPoseEdges[u]->information( ) ) );
        }
    }*/

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", m_vecKeyFramesInGraph.front( )->uID, m_vecKeyFramesInGraph.back( )->uID );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) optimizing [keyframes: %06lu-%06lu (%3lu) landmarks: %06lu-%06lu][measurements xyz: %3lu depth: %3lu disparity: %3lu] \n",
                 p_uFrame, vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID, m_vecKeyFramesInGraph.size( ), p_uIDBeginLandmark, m_vecLandmarksInGraph.back( )->uID, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

    //ds initialize optimization
    m_cOptimizerSparse.initializeOptimization( );

    //ds do iterations
    m_cOptimizerSparse.optimize( m_uIterations );

    //std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) applying optimization (total duration: %.2fs)\n", p_uFrame, CLogger::getTimeSeconds( )-dTimeStartSeconds );

    /*ds if there were loop closures and we are working in large scale
    if( 0 < uLoopClosures && 200 < m_vecPoseEdges.size( ) )
    {
        //ds first index
        const std::vector< g2o::EdgeSE3* >::size_type uStart = m_vecPoseEdges.size( )-200;

        //ds renormalize stiffness of recent trajectory
        for( std::vector< g2o::EdgeSE3* >::size_type u = uStart; u < m_vecPoseEdges.size( ); ++u )
        {
            m_vecPoseEdges[u]->setInformation( static_cast< Eigen::Matrix< double, 6, 6 > >( m_vecPoseEdges[u]->information( )/100 ) );
        }
    }*/

    //ds update points
    _applyOptimization( p_uFrame, p_uIDBeginLandmark, p_vecTranslationToG2o );
    _applyOptimization( p_vecTranslationToG2o );

    /*ds if there were loop closures
    if( 0 < uLoopClosures )
    {
        //ds save optimized situation
        m_cOptimizerSparse.save( ( strPrefix + "_closed.g2o" ).c_str( ) );

        //ds clear the graph as everything will be fixed for now
        m_cOptimizerSparse.clear( );
        m_vecLandmarksInGraph.clear( );
        m_vecKeyFramesInGraph.clear( );

        //ds buffer last keyframe
        const CKeyFrame* pKeyFrameLAST = vecChunkKeyFrames.back( );

        //ds add the copy of the last vertex again and lock it
        m_pVertexPoseLAST = new g2o::VertexSE3( );
        m_pVertexPoseLAST->setEstimate( pKeyFrameLAST->matTransformationLEFTtoWORLD );
        m_pVertexPoseLAST->setId( pKeyFrameLAST->uID+m_uIDShift );
        m_pVertexPoseLAST->setFixed( true );
        m_cOptimizerSparse.addVertex( m_pVertexPoseLAST );

        //ds include acceleration data
        m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( m_pVertexPoseLAST, pKeyFrameLAST->vecLinearAccelerationNormalized ) );
        std::printf( "- cleared graph\n" );
    }
    else
    {*/
        //ds save optimized situation
        m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );
        std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) optimization complete (total duration: %.2fs)\n", p_uFrame, CLogger::getTimeSeconds( )-dTimeStartSeconds );
    //}

    //ds done
    //std::printf( "<Cg2oOptimizer>(optimizeContinuous) optimized poses: %lu and landmarks: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), vecChunkLandmarks.size( ), m_cOptimizerSparse.activeChi2( ) );
}

void Cg2oOptimizer::clearFiles( ) const
{
    //ds directory handle
    DIR *pDirectory = opendir ( "g2o/local" );

    //ds try to open the directory
    if( 0 != pDirectory )
    {
        //ds file handle
        struct dirent *pFile;

        //ds delete all files
        while( ( pFile = readdir( pDirectory ) ) != NULL )
        {
            //ds validate filename
            if( '.' != pFile->d_name[0] )
            {
                //ds construct full filename and delete
                std::string strFile( "g2o/local/" );
                strFile += pFile->d_name;
                //std::printf( "<Cg2oOptimizer>(clearFiles) removing file: %s\n", strFile.c_str( ) );
                std::remove( strFile.c_str( ) );
            }
        }

        //ds close directory
        closedir( pDirectory );
    }
    else
    {
        //ds FATAL
        throw CExceptionLogfileTree( "g2o folder: g2o/local not set" );
    }
}

void Cg2oOptimizer::saveFinalGraph( const UIDFrame& p_uFrame, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds clear all structures
    m_cOptimizerSparse.clear( );

    //ds add all the landmarks again
    m_vecLandmarksInGraph.clear( );
    _loadLandmarksToGraph( 0, p_vecTranslationToG2o );

    //ds add first pose
    m_vecKeyFramesInGraph.clear( );
    m_pVertexPoseLAST = m_pVertexPoseFIRSTNOTINGRAPH;
    m_cOptimizerSparse.addVertex( m_pVertexPoseLAST );
    //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( m_pVertexPoseLAST, Eigen::Vector3d( 0.0, -1.0, 0.0 ) ) );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;
    uint32_t    uLoopClosures                  = 0;

    //ds loop over the camera vertices vector
    for( CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_pVertexPoseLAST, pKeyFrame, p_vecTranslationToG2o );

        //ds check if we got loop closures for this frame
        for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
        {
            //ds set it to the graph
            _setLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure, p_vecTranslationToG2o );
            ++uLoopClosures;
        }

        //ds always save acceleration data
        //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

        //ds set landmark measurements
        _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

        //ds update last
        m_pVertexPoseLAST = pVertexPoseCurrent;
    }

    //ds save to a file
    m_cOptimizerSparse.save( "g2o/local/FINAL.g2o" );

    std::printf( "[%06lu]<Cg2oOptimizer>(saveFinalGraph) saved [keyframes: %lu landmarks %lu][measurements xyz: %3lu depth: %3lu disparity: %3lu, loop closures: %3u]\n",
                 p_uFrame, m_vecKeyFrames->size( ), m_vecLandmarksInGraph.size( ), uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity, uLoopClosures );

}

//ds manual loop closing
void Cg2oOptimizer::updateLoopClosuresFrom( const std::vector< CKeyFrame* >::size_type& p_uIDBeginKeyFrame, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    assert( p_uIDBeginKeyFrame < m_vecKeyFrames->size( ) );
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );

    //ds loop over the camera vertices vector
    for( CKeyFrame* pKeyFrame: vecChunkKeyFrames )
    {
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

        //ds if we can update the vertex
        if( itPose != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds extract the pose
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPose->second );

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                //ds set the closure
                _setLoopClosure( pVertex, pKeyFrame, pClosure, p_vecTranslationToG2o );
            }
        }
    }
}

g2o::EdgeSE3LinearAcceleration* Cg2oOptimizer::_getEdgeLinearAcceleration( g2o::VertexSE3* p_pVertexPose,
                                                                           const CLinearAccelerationIMU& p_vecLinearAccelerationNormalized ) const
{
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAcceleration = new g2o::EdgeSE3LinearAcceleration( );

    pEdgeLinearAcceleration->setVertex( 0, p_pVertexPose );
    pEdgeLinearAcceleration->setMeasurement( p_vecLinearAccelerationNormalized );
    pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 5, 0, 0, 0, 5, 0, 0, 0, 5 };
    pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );

    //ds set robust kernel
    //pEdgeLinearAcceleration->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgeLinearAcceleration;
}

g2o::EdgeSE3PointXYZ* Cg2oOptimizer::_getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                               const CPoint3DWORLD& p_vecPointXYZ,
                                               const double& p_dInformationFactor ) const
{
    g2o::EdgeSE3PointXYZ* pEdgePointXYZ( new g2o::EdgeSE3PointXYZ( ) );

    //ds triangulated 3d point (uncalibrated)
    pEdgePointXYZ->setVertex( 0, p_pVertexPose );
    pEdgePointXYZ->setVertex( 1, p_pVertexLandmark );
    pEdgePointXYZ->setMeasurement( p_vecPointXYZ );
    pEdgePointXYZ->setParameterId( 0, EG2OParameterID::eWORLD );

    //ds the closer to the camera the point is the more meaningful is the measurement
    //const double dInformationStrengthXYZ( m_dMaximumReliableDepth/( 1+pMeasurementLandmark->vecPointXYZ.z( ) ) );
    const double arrInformationMatrixXYZ[9] = { p_dInformationFactor*1000, 0, 0, 0, p_dInformationFactor*1000, 0, 0, 0, p_dInformationFactor*1000 };
    pEdgePointXYZ->setInformation( g2o::Matrix3D( arrInformationMatrixXYZ ) );

    //ds set robust kernel
    pEdgePointXYZ->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgePointXYZ;
}

g2o::EdgeSE3PointXYZDepth* Cg2oOptimizer::_getEdgeUVDepthLEFT( g2o::VertexSE3* p_pVertexPose,
                                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                                               const CMeasurementLandmark* p_pMeasurement,
                                                               const double& p_dInformationFactor ) const
{
    //ds projected depth
    g2o::EdgeSE3PointXYZDepth* pEdgeProjectedDepth( new g2o::EdgeSE3PointXYZDepth( ) );

    pEdgeProjectedDepth->setVertex( 0, p_pVertexPose );
    pEdgeProjectedDepth->setVertex( 1, p_pVertexLandmark );
    pEdgeProjectedDepth->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, p_pMeasurement->vecPointXYZLEFT.z( ) ) );
    pEdgeProjectedDepth->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

    //ds information matrix
    //const double dInformationQualityDepth( m_dMaximumReliableDepth/dDepthMeters );
    const double arrInformationMatrixDepth[9] = { p_dInformationFactor, 0, 0, 0, p_dInformationFactor, 0, 0, 0, p_dInformationFactor*100 };
    pEdgeProjectedDepth->setInformation( g2o::Matrix3D( arrInformationMatrixDepth ) );

    //ds set robust kernel
    pEdgeProjectedDepth->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgeProjectedDepth;
}

g2o::EdgeSE3PointXYZDisparity* Cg2oOptimizer::_getEdgeUVDisparityLEFT( g2o::VertexSE3* p_pVertexPose,
                                                           g2o::VertexPointXYZ* p_pVertexLandmark,
                                                           const double& p_dDisparityPixels,
                                                           const CMeasurementLandmark* p_pMeasurement,
                                                           const double& p_dFxPixels,
                                                           const double& p_dBaselineMeters,
                                                           const double& p_dInformationFactor ) const
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
    const double arrInformationMatrixDisparity[9] = { p_dInformationFactor, 0, 0, 0, p_dInformationFactor, 0, 0, 0, p_dInformationFactor*100 };
    pEdgeDisparity->setInformation( g2o::Matrix3D( arrInformationMatrixDisparity ) );

    //ds set robust kernel
    pEdgeDisparity->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgeDisparity;
}

void Cg2oOptimizer::_loadLandmarksToGraph( const std::vector< CLandmark* >::size_type& p_uIDLandmark, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds info
    UIDLandmark uDroppedLandmarksInvalid    = 0;
    UIDLandmark uDroppedLandmarksKeyFraming = 0;
    UIDLandmark uLandmarksAlreadyInGraph    = 0;

    assert( p_uIDLandmark < m_vecLandmarks->size( ) );

    //ds add landmarks
    for( std::vector< CLandmark* >::size_type uID = p_uIDLandmark; uID < m_vecLandmarks->size( ); ++uID )
    {
        //ds buffer landmark
        CLandmark* pLandmark = m_vecLandmarks->at( uID );

        assert( 0 != pLandmark );

        //ds check if optimiziation criterias are met
        if( pLandmark->bIsOptimal )
        {
            //ds landmark has to be present in 2 keyframes
            if( 1 < pLandmark->uNumberOfKeyFramePresences )
            {
                //ds try to retrieve vertex
                const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

                //ds if found
                if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
                {
                    //ds extract and update the estimate
                    g2o::VertexPointXYZ* pVertex = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );
                    pVertex->setEstimate( pLandmark->vecPointXYZOptimized+p_vecTranslationToG2o );

                    ++uLandmarksAlreadyInGraph;
                }
                else if( 1e9 > ( pLandmark->vecPointXYZOptimized+p_vecTranslationToG2o ).squaredNorm( ) )
                {
                    //ds set landmark vertex
                    g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
                    pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized+p_vecTranslationToG2o );
                    pVertexLandmark->setId( pLandmark->uID );

                    //ds add vertex to optimizer
                    m_cOptimizerSparse.addVertex( pVertexLandmark );
                    m_vecLandmarksInGraph.push_back( pLandmark );
                }
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

    //std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) dropped landmarks due to invalid optimization: %lu (%4.2f)\n", uDroppedLandmarksInvalid, static_cast< double >( uDroppedLandmarksInvalid )/p_vecChunkLandmarks.size( ) );
    //std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) dropped landmarks due to missing key frame presence: %lu (%4.2f)\n", uDroppedLandmarksKeyFraming, static_cast< double >( uDroppedLandmarksKeyFraming )/p_vecChunkLandmarks.size( ) );
    //std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) landmarks already present in graph: %lu\n", uLandmarksAlreadyInGraph );
}

g2o::VertexSE3* Cg2oOptimizer::_setAndgetPose( g2o::VertexSE3* p_pVertexPoseFrom, CKeyFrame* pKeyFrameCurrent, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds compose transformation matrix
    Eigen::Isometry3d matTransformationLEFTtoWORLD = pKeyFrameCurrent->matTransformationLEFTtoWORLD;
    matTransformationLEFTtoWORLD.translation( ) += p_vecTranslationToG2o;

    //ds add current camera pose
    g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
    pVertexPoseCurrent->setEstimate( matTransformationLEFTtoWORLD );
    pVertexPoseCurrent->setId( pKeyFrameCurrent->uID+m_uIDShift );
    m_cOptimizerSparse.addVertex( pVertexPoseCurrent );

    //ds set up the edge to connect the poses
    g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

    //ds set viewpoints and measurement
    pEdgePoseFromTo->setVertex( 0, p_pVertexPoseFrom );
    pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
    pEdgePoseFromTo->setMeasurement( p_pVertexPoseFrom->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );
    //pEdgePoseFromTo->setInformation( static_cast< Eigen::Matrix< double, 6, 6 > >( pKeyFrameCurrent->dInformationFactor*m_matInformationPose ) );
    pEdgePoseFromTo->setInformation( m_matInformationPose );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgePoseFromTo );

    //ds add to control structure
    m_vecPoseEdges.push_back( pEdgePoseFromTo );

    return pVertexPoseCurrent;
}

void Cg2oOptimizer::_setLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent, const CKeyFrame::CMatchICP* p_pClosure, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds find the corresponding pose in the current graph
    const g2o::HyperGraph::VertexIDMap::iterator itClosure( m_cOptimizerSparse.vertices( ).find( p_pClosure->pKeyFrameReference->uID+m_uIDShift ) );

    //ds has to be found
    assert( itClosure != m_cOptimizerSparse.vertices( ).end( ) );

    //ds pose to close
    g2o::VertexSE3* pVertexClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );

    /*ds if found
    if( itClosure != m_cOptimizerSparse.vertices( ).end( ) )
    {
        //ds extract the pose
        pVertexClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );
    }
    else
    {
        //ds compose transformation matrix
        Eigen::Isometry3d matTransformationLEFTtoWORLD = p_pClosure->pKeyFrameReference->matTransformationLEFTtoWORLD;
        matTransformationLEFTtoWORLD.translation( ) += p_vecTranslationToG2o;

        //ds we have to add the pose temporarily and fix it
        pVertexClosure = new g2o::VertexSE3( );
        pVertexClosure->setEstimate( matTransformationLEFTtoWORLD );
        pVertexClosure->setId( p_pClosure->pKeyFrameReference->uID+m_uIDShift );
        m_cOptimizerSparse.addVertex( pVertexClosure );
    }*/

    //ds fix reference
    pVertexClosure->setFixed( true );

    //ds set up the edge
    g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgeLoopClosure->setVertex( 0, pVertexClosure );
    pEdgeLoopClosure->setVertex( 1, p_pVertexPoseCurrent );
    pEdgeLoopClosure->setMeasurement( p_pClosure->matTransformationToClosure );
    pEdgeLoopClosure->setInformation( m_matInformationLoopClosure );

    //ds kernelize loop closing
    //pEdgeLoopClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgeLoopClosure );

    /*ds connect all closed landmarks
    for( const CMatchCloud& cMatch: *p_pClosure->vecMatches )
    {
        //ds find the corresponding landmarks
        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkQuery( m_cOptimizerSparse.vertices( ).find( cMatch.cPointQuery.uID ) );
        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkReference( m_cOptimizerSparse.vertices( ).find( cMatch.cPointReference.uID ) );

        //ds if query is in the graph
        if( itLandmarkQuery != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds consistency
            assert( cMatch.cPointReference.uID == m_vecLandmarks->at( cMatch.cPointReference.uID )->uID );

            //ds reference landmark
            g2o::VertexPointXYZ* pVertexLandmarkReference = 0;

            //ds check if the reference landmark is present in the graph
            if( itLandmarkReference != m_cOptimizerSparse.vertices( ).end( ) )
            {
                //ds get it from the graph
                pVertexLandmarkReference = dynamic_cast< g2o::VertexPointXYZ* >( itLandmarkReference->second );

                //ds fix reference landmark
                pVertexLandmarkReference->setFixed( true );

                //ds set up the edge
                g2o::EdgePointXYZ* pEdgeLandmarkClosure( new g2o::EdgePointXYZ( ) );

                //ds set viewpoints and measurement
                pEdgeLandmarkClosure->setVertex( 0, itLandmarkQuery->second );
                pEdgeLandmarkClosure->setVertex( 1, pVertexLandmarkReference );
                pEdgeLandmarkClosure->setMeasurement( Eigen::Vector3d::Zero( ) );
                pEdgeLandmarkClosure->setInformation( m_matInformationLandmarkClosure );
                pEdgeLandmarkClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                //ds add to optimizer
                m_cOptimizerSparse.addEdge( pEdgeLandmarkClosure );
            }
        }
    }*/
}

void Cg2oOptimizer::_setLandmarkMeasurementsWORLD( g2o::VertexSE3* p_pVertexPoseCurrent,
                                      const CKeyFrame* pKeyFrameCurrent,
                                      UIDLandmark& p_uMeasurementsStoredPointXYZ,
                                      UIDLandmark& p_uMeasurementsStoredUVDepth,
                                      UIDLandmark& p_uMeasurementsStoredUVDisparity )
{
    //ds check visible landmarks and add the edges for the current pose
    for( const CMeasurementLandmark* pMeasurementLandmark: pKeyFrameCurrent->vecMeasurements )
    {
        //ds find the corresponding landmark
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pMeasurementLandmark->uID ) );

        //ds if found
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
            g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

            //ds get optimized landmark into current pose
            const CPoint3DCAMERA vecPointLEFTEstimate( p_pVertexPoseCurrent->estimate( ).inverse( )*pVertexLandmark->estimate( ) );

            //ds current distance situation
            const double dDistanceL2Absolute = pMeasurementLandmark->vecPointXYZLEFT.squaredNorm( );
            const double dDistanceL2Relative = vecPointLEFTEstimate.squaredNorm( )/dDistanceL2Absolute;

            //ds check if the measurement is sufficiently consistent
            if( 0.75 < dDistanceL2Relative && 1.25 > dDistanceL2Relative )
            {
                //ds the closer the more reliable (a depth less than 1 meter actually increases the informational value)
                const double dInformationFactor  = 1.0/pMeasurementLandmark->vecPointXYZLEFT.z( );

                //ds maximum depth to produce a reliable XYZ estimate
                if( m_dMaximumReliableDepthForPointXYZL2 > dDistanceL2Absolute )
                {
                    //ds retrieve the edge
                    g2o::EdgeSE3PointXYZ* pEdgePointXYZ = _getEdgePointXYZ( p_pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark->vecPointXYZLEFT, dInformationFactor );

                    //ds add the edge to the graph
                    m_cOptimizerSparse.addEdge( pEdgePointXYZ );
                    ++p_uMeasurementsStoredPointXYZ;
                }

                //ds still good enough for uvdepth
                else if( m_dMaximumReliableDepthForUVDepthL2 > dDistanceL2Absolute )
                {
                    //ds get the edge
                    g2o::EdgeSE3PointXYZDepth* pEdgeUVDepth = _getEdgeUVDepthLEFT( p_pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark, dInformationFactor );

                    //ds projected depth (LEFT camera)
                    m_cOptimizerSparse.addEdge( pEdgeUVDepth );
                    ++p_uMeasurementsStoredUVDepth;
                }

                //ds go with disparity
                else if( m_dMaximumReliableDepthForUVDisparityL2 > dDistanceL2Absolute )
                {
                    //ds current disparity
                    const double dDisparity = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;

                    //ds at least 2 pixels stereo distance
                    if( 1.0 < dDisparity )
                    {
                        //ds get a disparity edge
                        g2o::EdgeSE3PointXYZDisparity* pEdgeUVDisparity( _getEdgeUVDisparityLEFT( p_pVertexPoseCurrent,
                                                                                                  pVertexLandmark,
                                                                                                  dDisparity,
                                                                                                  pMeasurementLandmark,
                                                                                                  m_pCameraSTEREO->m_pCameraLEFT->m_dFxP,
                                                                                                  m_pCameraSTEREO->m_dBaselineMeters,
                                                                                                  dInformationFactor ) );

                        //ds disparity (LEFT camera)
                        m_cOptimizerSparse.addEdge( pEdgeUVDisparity );
                        ++p_uMeasurementsStoredUVDisparity;
                    }
                }
            }
            else
            {
                //std::printf( "<Cg2oOptimizer>(_setLandmarkMeasurementsWORLD) ignoring malformed landmark [%06lu] depth difference: %f\n", pMeasurementLandmark->uID, dDifferenceDepth );
            }
        }
    }
}

void Cg2oOptimizer::_applyOptimization( const UIDFrame& p_uFrame, const std::vector< CLandmark* >::size_type& p_uIDLandmark, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    std::vector< CLandmark* >::size_type uNumberOfErasedLandmarks = 0;

    //ds update landmarks
    for( CLandmark* pLandmark: m_vecLandmarksInGraph )
    {
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

        //ds must be present
        //assert( itLandmark != m_cOptimizerSparse.vertices( ).end( ) );
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds get the vertex
            g2o::VertexPointXYZ* pVertex = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

            //ds if added in a previous optimization
            if( p_uIDLandmark > pLandmark->uID )
            {
                //ds update position
                pLandmark->vecPointXYZOptimized = pVertex->estimate( )-p_vecTranslationToG2o;
            }
            else
            {
                //ds check if not madly optimized (might happen with disparity egdes)
                if( 1e9 > pVertex->estimate( ).squaredNorm( )                                                           &&
                    1e6 > ( pLandmark->vecPointXYZOptimized+p_vecTranslationToG2o-pVertex->estimate( ) ).squaredNorm( ) )
                {
                    //ds update position
                    pLandmark->vecPointXYZOptimized = pVertex->estimate( )-p_vecTranslationToG2o;
                }
                else
                {
                    //ds reset g2o instance
                    ++uNumberOfErasedLandmarks;

                    //ds detach the vertex and its edges - edges first
                    for( g2o::HyperGraph::EdgeSet::const_iterator itEdge = pVertex->edges( ).begin( ); itEdge != pVertex->edges( ).end( ); ++itEdge )
                    {
                        m_cOptimizerSparse.removeEdge( *itEdge );
                    }

                    //ds remove the vertex
                    m_cOptimizerSparse.removeVertex( pVertex );
                }
            }
        }
    }

    if( 0 < uNumberOfErasedLandmarks )
    {
        std::printf( "[%06lu]<Cg2oOptimizer>(_applyOptimization) erased badly optimized landmarks: %lu\n", p_uFrame, uNumberOfErasedLandmarks );
    }
}

void Cg2oOptimizer::_applyOptimization( const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds update poses
    for( CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds find vertex in graph
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itKeyFrame( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

        //ds has to be found
        assert( itKeyFrame != m_cOptimizerSparse.vertices( ).end( ) );

        //ds extract and update the keyframe
        g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itKeyFrame->second );
        pKeyFrame->matTransformationLEFTtoWORLD                = pVertex->estimate( );
        pKeyFrame->matTransformationLEFTtoWORLD.translation( ) -= p_vecTranslationToG2o;
        pKeyFrame->bIsOptimized                                = true;

        //ds relax the vertex again for next loop-closing optimization
        pVertex->setFixed( false );
    }
}
