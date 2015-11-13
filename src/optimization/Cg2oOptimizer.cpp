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

    m_uIDKeyFrameFrom = -1;

    //ds trajectory only
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+1 );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );
    m_cOptimizerSparseTrajectoryOnly.addVertex( m_pVertexPoseFIRSTNOTINGRAPH );

    //ds add the first pose separately
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+m_uIDShift );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );

    //ds set to graph
    m_cOptimizerSparse.addVertex( m_pVertexPoseFIRSTNOTINGRAPH );
    m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( m_pVertexPoseFIRSTNOTINGRAPH, Eigen::Vector3d( 0.0, -1.0, 0.0 ) ) );

    //ds get a copy
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+m_uIDShift );
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
                                                                                       m_matInformationLoopClosure( _getInformationNoZ( static_cast< Eigen::Matrix< double, 6, 6 > >( 10*m_matInformationPose ) ) ),
                                                                                       m_matInformationLandmarkClosure( 1000*Eigen::Matrix< double, 3, 3 >::Identity( ) )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );
    m_vecPoseEdges.clear( );
    m_mapEdgeIDs.clear( );

    //ds configure the solver (var_cholmod)
    //m_cOptimizerSparse.setVerbose( true );
    g2o::BlockSolverX::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >( ) );
    g2o::BlockSolverX* pSolver( new g2o::BlockSolverX( pLinearSolver ) );
    //g2o::OptimizationAlgorithmGaussNewton* pAlgorithm( new g2o::OptimizationAlgorithmGaussNewton( pSolver ) );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm( new g2o::OptimizationAlgorithmLevenberg( pSolver ) );
    m_cOptimizerSparse.setAlgorithm( pAlgorithm );

    //ds trajectory only
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolverTrajectoryOnly( new g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType >( ) );
    g2o::BlockSolver_6_3* pSolverTrajectoryOnly( new g2o::BlockSolver_6_3( pLinearSolverTrajectoryOnly ) );
    g2o::OptimizationAlgorithmGaussNewton* pAlgorithmTrajectoryOnly( new g2o::OptimizationAlgorithmGaussNewton( pSolverTrajectoryOnly ) );
    //g2o::OptimizationAlgorithmLevenberg* pAlgorithmTrajectoryOnly( new g2o::OptimizationAlgorithmLevenberg( pSolverTrajectoryOnly ) );
    m_cOptimizerSparseTrajectoryOnly.setAlgorithm( pAlgorithmTrajectoryOnly );

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
    //ds disabled for now
    assert( false );

    /*ds clear nodes and edges
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
    _applyOptimization( 0, p_vecTranslationToG2o );

    //ds done
    std::printf( "<Cg2oOptimizer>(optimizeTailLoopClosuresOnly) optimized poses: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), m_cOptimizerSparse.activeChi2( ) );
    ++m_uOptimizations;*/
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
                                        const std::vector< CKeyFrame* >::size_type& p_uLoopClosureKeyFrames )
{
    //ds local optimization chunks
    assert( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame <= m_vecKeyFrames->end( ) );
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );
    if( vecChunkKeyFrames.empty( ) ){ return; }
    assert( !vecChunkKeyFrames.front( )->vecMeasurements.empty( ) );

    //ds sane input
    ++m_uOptimizations;

    //ds check if we have a loop closure (landmark free optimization)
    if( 0 < p_uLoopClosureKeyFrames )
    {
        //ds closure count
        uint32_t uLoopClosuresTotal = 0;

        //ds loop closure holder
        std::vector< CLoopClosureRaw > vecLoopClosures;

        //ds loop over the camera vertices vector
        for( CKeyFrame* pKeyFrame: vecChunkKeyFrames )
        {
            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_uIDKeyFrameFrom, pKeyFrame, p_vecTranslationToG2o );

            //ds closed edges
            OptimizableGraph::EdgeSet vecLoopClosureEdges;

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                //ds accumulate the elements
                vecLoopClosureEdges.insert( _getEdgeLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure ) );
                ++uLoopClosuresTotal;

                //ds get loop closure
                //g2o::EdgeSE3* pEdgeLoopClosure = _getEdgeLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure );

                //ds add to graph
                //m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
                //vecLoopClosures.push_back( CLoopClosureRaw( pClosure->pKeyFrameReference->uID, pKeyFrame->uID, pEdgeLoopClosure ) );
            }

            //ds if there were any loop closures for this keyframe
            if( !vecLoopClosureEdges.empty( ) )
            {
                const g2o::HyperGraph::VertexIDMap::iterator itPoseCurrent( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrame->uID+1 ) );
                assert( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) != itPoseCurrent );

                m_cBufferClosures.addEdgeSet( vecLoopClosureEdges );
                m_cBufferClosures.addVertex( dynamic_cast< g2o::VertexSE3* >( itPoseCurrent->second ) );
            }

            //ds evaluate loop closure window
            if( m_cBufferClosures.checkList( p_uLoopClosureKeyFrames ) )
            {
                //ds determine valid closures
                m_cClosureChecker.init( m_cBufferClosures.vertices( ), m_cBufferClosures.edgeSet( ), m_dMaximumThresholdLoopClosing );
                m_cClosureChecker.check( );

                std::printf( "WINDOW SIZE: %lu\n", p_uLoopClosureKeyFrames );
                std::printf( "INLIERS: %i\n", m_cClosureChecker.inliers( ) );

                //ds if at least one inlier
                if( 0 < m_cClosureChecker.inliers( ) )
                {
                    for( LoopClosureChecker::EdgeDoubleMap::iterator itEdgeLoopClosure = m_cClosureChecker.closures( ).begin(); itEdgeLoopClosure != m_cClosureChecker.closures( ).end( ); itEdgeLoopClosure++ )
                    {
                        //ds get edge form
                        g2o::EdgeSE3* pEdgeLoopClosure = dynamic_cast< g2o::EdgeSE3* >( itEdgeLoopClosure->first );

                        if( m_dMaximumThresholdLoopClosing > itEdgeLoopClosure->second )
                        {
                            std::printf( "ADDING CLOSURE %06i to %06i WITH CHI2: %f\n", pEdgeLoopClosure->vertices( )[0]->id( )-1, pEdgeLoopClosure->vertices( )[1]->id( )-1, itEdgeLoopClosure->second );

                            //ds add to graph
                            m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
                            vecLoopClosures.push_back( CLoopClosureRaw( pEdgeLoopClosure->vertices( )[0]->id( )-1, pEdgeLoopClosure->vertices( )[1]->id( )-1, pEdgeLoopClosure ) );
                        }
                    }

                    //ds remove all candidate edges added so far
                    m_cBufferClosures.removeEdgeSet( m_cBufferClosures.edgeSet( ) );
                }
            }

            //ds update buffer
            m_cBufferClosures.updateList( p_uLoopClosureKeyFrames );

            //ds update last
            m_uIDKeyFrameFrom = pKeyFrame->uID;
        }

        //ds we always process all keyframes
        m_vecKeyFramesInGraph.insert( m_vecKeyFramesInGraph.end( ), vecChunkKeyFrames.begin( ), vecChunkKeyFrames.end( ) );

        //ds if there were any closures accepted
        if( !vecLoopClosures.empty( ) )
        {
            std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) optimizing loop closures: %lu/%u\n", p_uFrame, vecLoopClosures.size( ), uLoopClosuresTotal );

            //ds lock selected closures
            for( CLoopClosureRaw cEdgeLoopClosureRaw: vecLoopClosures )
            {
                const g2o::HyperGraph::VertexIDMap::iterator itReference( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( cEdgeLoopClosureRaw.uIDReference+1 ) );
                assert( itReference != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
                g2o::VertexSE3* pVertexReference = dynamic_cast< g2o::VertexSE3* >( itReference->second );
                pVertexReference->setFixed( true );
            }

            //ds save closure situation
            char chBufferLC[256];
            std::snprintf( chBufferLC, 256, "g2o/local/keyframes_%06lu-%06lu", vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID );
            std::string strPrefixLC( chBufferLC );
            m_cOptimizerSparseTrajectoryOnly.save( ( strPrefixLC + "_closed.g2o" ).c_str( ) );

            //ds timing
            const double dTimeStartSecondsLC = CLogger::getTimeSeconds( );
            const uint64_t uIterationsLC     = 1000;

            //ds always 1000 steps for loop closing!
            m_cOptimizerSparseTrajectoryOnly.initializeOptimization( );
            m_cOptimizerSparseTrajectoryOnly.optimize( uIterationsLC );

            //ds unlock added closures
            for( CLoopClosureRaw cEdgeLoopClosureRaw: vecLoopClosures )
            {
                //ds relax locked reference vertices again
                const g2o::HyperGraph::VertexIDMap::iterator itReference( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( cEdgeLoopClosureRaw.uIDReference+1 ) );
                assert( itReference != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
                g2o::VertexSE3* pVertexReference = dynamic_cast< g2o::VertexSE3* >( itReference->second );
                pVertexReference->setFixed( false );
            }

            //ds back propagate the trajectory only graph into the full one
            _backPropagateTrajectoryToFull( vecLoopClosures );

            m_cOptimizerSparseTrajectoryOnly.save( ( strPrefixLC + "_closed_optimized.g2o" ).c_str( ) );
            std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) optimization complete (total duration: %.2fs | iterations: %lu)\n",
                         p_uFrame, ( CLogger::getTimeSeconds( )-dTimeStartSecondsLC ), uIterationsLC );
        }
        else
        {
            std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) DROPPED loop closures: %u\n", p_uFrame, uLoopClosuresTotal );
        }
    }

    //ds set landmarks
    _loadLandmarksToGraph( p_uIDBeginLandmark, p_vecTranslationToG2o );
    //std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) loading keyframes (total duration: %.2fs)\n", p_uFrame, CLogger::getTimeSeconds( )-dTimeStartSeconds );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;

    //ds if loop closed - we don't have to add the frames again, just the landmarks
    if( 0 < p_uLoopClosureKeyFrames )
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
            m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            /*ds if the key frame was closed
            if( !pKeyFrame->vecLoopClosures.empty( ) )
            {
                //ds for each closure
                for( const CKeyFrame::CMatchICP* pMatch: pKeyFrame->vecLoopClosures )
                {
                    //ds connect all closed landmarks
                    for( const CMatchCloud& cMatch: *pMatch->vecMatches )
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
                                pEdgeLandmarkClosure->setVertex( 0, pVertexLandmarkReference );
                                pEdgeLandmarkClosure->setVertex( 1, itLandmarkQuery->second );
                                pEdgeLandmarkClosure->setMeasurement( Eigen::Vector3d::Zero( ) );
                                pEdgeLandmarkClosure->setInformation( m_matInformationLandmarkClosure );
                                pEdgeLandmarkClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                                //ds add to optimizer
                                m_cOptimizerSparse.addEdge( pEdgeLandmarkClosure );
                            }
                        }
                    }
                }
            }*/
        }
    }
    else
    {
        //ds loop over the camera vertices vector
        for( CKeyFrame* pKeyFrame: vecChunkKeyFrames )
        {
            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_uIDKeyFrameFrom, pKeyFrame, p_vecTranslationToG2o );

            //ds always save acceleration data
            m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            //ds update last
            m_uIDKeyFrameFrom = pKeyFrame->uID;
        }

        //ds we always process all keyframes
        m_vecKeyFramesInGraph.insert( m_vecKeyFramesInGraph.end( ), vecChunkKeyFrames.begin( ), vecChunkKeyFrames.end( ) );
    }

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
                 p_uFrame, vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID, ( vecChunkKeyFrames.back( )->uID-vecChunkKeyFrames.front( )->uID ), p_uIDBeginLandmark, m_vecLandmarksInGraph.back( )->uID, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

    //ds timing
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );
    const uint64_t uIterations     = _optimizeLimited( m_cOptimizerSparse );

    /*ds start cascading iteration steps
    m_cOptimizerSparse.optimize( 1 );
    uint32_t uIterationsDone = 1;
    if( 10.0 > ( CLogger::getTimeSeconds( )-dTimeStartSeconds ) )
    {
        m_cOptimizerSparse.optimize( 10 );
        uIterationsDone += 10;
        if( 20.0 > ( CLogger::getTimeSeconds( )-dTimeStartSeconds ) )
        {
            m_cOptimizerSparse.optimize( 100 );
            uIterationsDone += 100;
            if( 30.0 > ( CLogger::getTimeSeconds( )-dTimeStartSeconds ) )
            {
                m_cOptimizerSparse.optimize( 1000 );
                uIterationsDone += 1000;
            }
        }
    }*/

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
    _applyOptimization( p_uFrame, p_vecTranslationToG2o );

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

        //ds back propagate the full graph into trajectory only
        _backPropagateTrajectoryToPure( );

        //ds save optimized situation
        m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );
        std::printf( "[%06lu]<Cg2oOptimizer>(optimizeContinuous) optimization complete (total duration: %.2fs | iterations: %lu)\n", p_uFrame, ( CLogger::getTimeSeconds( )-dTimeStartSeconds ), uIterations );
    //}

    //ds done
    //std::printf( "<Cg2oOptimizer>(optimizeContinuous) optimized poses: %lu and landmarks: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), vecChunkLandmarks.size( ), m_cOptimizerSparse.activeChi2( ) );

    m_dTotalOptimizationDurationSeconds += ( CLogger::getTimeSeconds( )-dTimeStartSeconds );
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
    assert( !m_vecKeyFramesInGraph.empty( ) );

    //ds clear all structures
    m_uIDKeyFrameFrom = -1;
    m_cOptimizerSparse.clear( );

    //ds add all the landmarks again
    m_vecLandmarksInGraph.clear( );
    _loadLandmarksToGraph( 0, p_vecTranslationToG2o );

    //ds add first pose
    m_cOptimizerSparse.addVertex( m_pVertexPoseFIRSTNOTINGRAPH );
    m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( m_pVertexPoseFIRSTNOTINGRAPH, Eigen::Vector3d( 0.0, -1.0, 0.0 ) ) );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;
    uint32_t    uLoopClosures                  = 0;

    //ds loop over the camera vertices vector
    for( CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        assert( 0 != pKeyFrame );

        //ds for sane id
        if( 0 <= pKeyFrame->uID && 2*m_uIDShift > pKeyFrame->uID )
        {
            //ds compose transformation matrix
            Eigen::Isometry3d matTransformationLEFTtoWORLD = pKeyFrame->matTransformationLEFTtoWORLD;
            matTransformationLEFTtoWORLD.translation( ) += p_vecTranslationToG2o;

            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
            pVertexPoseCurrent->setEstimate( matTransformationLEFTtoWORLD );
            pVertexPoseCurrent->setId( pKeyFrame->uID+m_uIDShift );
            m_cOptimizerSparse.addVertex( pVertexPoseCurrent );

            //ds set up the edge to connect the poses
            g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

            //ds set viewpoints and measurement
            const g2o::HyperGraph::VertexIDMap::iterator itPoseFrom = m_cOptimizerSparse.vertices( ).find( m_uIDKeyFrameFrom+m_uIDShift );
            assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseFrom );
            g2o::VertexSE3* pVertexPoseFrom = dynamic_cast< g2o::VertexSE3* >( itPoseFrom->second );
            pEdgePoseFromTo->setVertex( 0, pVertexPoseFrom );
            pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
            pEdgePoseFromTo->setMeasurement( pVertexPoseFrom->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

            //ds add to optimizer
            m_cOptimizerSparse.addEdge( pEdgePoseFromTo );

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                //ds accumulate the elements
                _setLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure, p_vecTranslationToG2o );
                ++uLoopClosures;
            }

            //ds always save acceleration data
            m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            //ds update last
            m_uIDKeyFrameFrom = pKeyFrame->uID;
        }
        else
        {
            std::printf( "[%06lu]<Cg2oOptimizer>(saveFinalGraph) caught unsafe key frame id: %06lu\n", p_uFrame, pKeyFrame->uID );
        }
    }

    //ds save to a file
    m_cOptimizerSparse.save( "g2o/local/FINAL.g2o" );

    std::printf( "[%06lu]<Cg2oOptimizer>(saveFinalGraph) saved [keyframes: %lu landmarks %lu][measurements xyz: %3lu depth: %3lu disparity: %3lu, loop closures: %3u]\n",
                 p_uFrame, m_vecKeyFrames->size( ), m_vecLandmarksInGraph.size( ), uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity, uLoopClosures );

}

//ds manual loop closing
void Cg2oOptimizer::updateLoopClosuresFromKeyFrame( const std::vector< CKeyFrame* >::size_type& p_uIDBeginKeyFrame,
                                                    const std::vector< CKeyFrame* >::size_type& p_uIDEndKeyFrame,
                                                    const Eigen::Vector3d& p_vecTranslationToG2o )
{
    assert( p_uIDBeginKeyFrame < p_uIDEndKeyFrame );
    assert( p_uIDBeginKeyFrame < m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFramesInGraph.size( ) );

    //ds update key frames in graph (null now) and g2o
    for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < p_uIDEndKeyFrame; ++uID )
    {
        assert( 0 != m_vecKeyFrames->at( uID ) );

        //ds only if updated
        if( 0 == m_vecKeyFramesInGraph[uID] )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds update internals
            m_vecKeyFramesInGraph[uID] = pKeyFrame;

            //ds update g2o: try to retrieve vertex
            const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

            //ds if we can update the vertex
            assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

            //ds extract the pose
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPose->second );

            assert( 0 != pVertex );

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                assert( 0 != pClosure );

                //ds set the closure
                //_setLoopClosure( pVertex, pKeyFrame, pClosure, p_vecTranslationToG2o );

                //ds get loop closure
                g2o::EdgeSE3* pEdgeLoopClosure = _getEdgeLoopClosure( pVertex, pKeyFrame, pClosure );

                //ds add to graph
                m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
            }
        }
    }
}

/*void Cg2oOptimizer::lockTrajectory( const std::vector< CKeyFrame* >::size_type& p_uIDBeginKeyFrame,
                                    const std::vector< CKeyFrame* >::size_type& p_uIDEndKeyFrame )
{
    assert( p_uIDBeginKeyFrame < p_uIDEndKeyFrame );
    assert( p_uIDBeginKeyFrame < m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFramesInGraph.size( ) );

    //ds lock all poses
    for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < p_uIDEndKeyFrame; ++uID )
    {
        //ds update g2o: try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( uID+m_uIDShift ) );

        //ds if we can update the vertex
        assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

        //ds extract the pose and lock it
        g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPose->second );
        pVertex->setFixed( true );

        std::cerr << "fixed keyframe: " << uID << std::endl;
    }

    //ds if feasiable
    if( 50 < m_vecKeyFramesInGraph.back( )->uID )
    {
        //ds lower limit
        const std::vector< CKeyFrame* >::size_type uIDStart = m_vecKeyFramesInGraph.back( )->uID-50;

        //ds increase stiffness of recent trajectory
        for( std::vector< CKeyFrame* >::size_type uID = m_vecKeyFramesInGraph.back( )->uID; uID > uIDStart; --uID )
        {
            //ds update g2o: try to retrieve vertex
            const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( uID+m_uIDShift ) );

            //ds if we can update the vertex
            assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

            //ds extract the pose and lock it
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPose->second );

            assert( 0 != pVertex );

            //ds buffer connecting edge
            g2o::EdgeSE3* pEdgePoseFromTo = dynamic_cast< g2o::EdgeSE3* >( *( pVertex->edges( ).begin( ) ) );

            if( 0 != pEdgePoseFromTo )
            {
                //ds increase information matrix
                pEdgePoseFromTo->setInformation( static_cast< Eigen::Matrix< double, 6,6 > >( 100*pEdgePoseFromTo->information( ) ) );

                std::cerr << "strengthened keyframe: " << uID << std::endl;
            }
        }
    }
}*/

uint64_t Cg2oOptimizer::_optimizeLimited( g2o::SparseOptimizer& p_cOptimizer )
{
    //ds initialize optimization
    p_cOptimizer.initializeOptimization( );

    //ds initial optimization
    p_cOptimizer.optimize( 1 );

    //ds iterations count
    uint64_t uIterations = 1;

    //ds initial timestamp
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );

    //ds start optimization
    double dPreviousChi2 = p_cOptimizer.chi2( )+2.0;

    //ds start cycle (maximum 100s)
    while( ( 0.1 < dPreviousChi2-p_cOptimizer.chi2( ) ) && ( 100.0 > CLogger::getTimeSeconds( )-dTimeStartSeconds ) )
    {
        //ds update chi2
        dPreviousChi2 = p_cOptimizer.chi2( );

        //ds do ten iterations
        p_cOptimizer.optimize( 10 );
        uIterations += 10;
    }

    return uIterations;
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
    //pEdgePointXYZ->setRobustKernel( new g2o::RobustKernelCauchy( ) );

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
    //pEdgeProjectedDepth->setRobustKernel( new g2o::RobustKernelCauchy( ) );

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
    const double arrInformationMatrixDisparity[9] = { p_dInformationFactor, 0, 0, 0, p_dInformationFactor, 0, 0, 0, p_dInformationFactor*1000 };
    pEdgeDisparity->setInformation( g2o::Matrix3D( arrInformationMatrixDisparity ) );

    //ds set robust kernel
    pEdgeDisparity->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgeDisparity;
}

g2o::EdgeSE3* Cg2oOptimizer::_getEdgeLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent, const CKeyFrame::CMatchICP* p_pClosure )
{
    /*ds find the corresponding pose in the current graph
    const g2o::HyperGraph::VertexIDMap::iterator itClosure( m_cOptimizerSparse.vertices( ).find( p_pClosure->pKeyFrameReference->uID+m_uIDShift ) );

    //ds has to be found
    assert( itClosure != m_cOptimizerSparse.vertices( ).end( ) );

    //ds pose to close
    g2o::VertexSE3* pVertexClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );

    //ds set up the edge
    g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgeLoopClosure->setVertex( 0, pVertexClosure );
    pEdgeLoopClosure->setVertex( 1, p_pVertexPoseCurrent );
    pEdgeLoopClosure->setMeasurement( p_pClosure->matTransformationToClosure );
    pEdgeLoopClosure->setInformation( m_matInformationLoopClosure );*/

    //ds find the corresponding poses in the current graph
    const g2o::HyperGraph::VertexIDMap::iterator itCurrent( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrameCurrent->uID+1 ) );
    const g2o::HyperGraph::VertexIDMap::iterator itClosure( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( p_pClosure->pKeyFrameReference->uID+1 ) );

    //ds have to be found
    assert( itCurrent != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
    assert( itClosure != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );

    //ds buffer poses
    g2o::VertexSE3* pVertexPoseClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );
    g2o::VertexSE3* pVertexPoseCurrent = dynamic_cast< g2o::VertexSE3* >( itCurrent->second );

    //ds set up the edge
    g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgeLoopClosure->setVertex( 0, pVertexPoseClosure );
    pEdgeLoopClosure->setVertex( 1, pVertexPoseCurrent );
    pEdgeLoopClosure->setMeasurement( p_pClosure->matTransformationToClosure );
    pEdgeLoopClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    //ds compute error
    //const Eigen::Vector3d verTranslationDifference( ( pVertexPoseClosure->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) ).translation( )-p_pClosure->matTransformationToClosure.translation( ) );

    /*ds check information matrix - check if loop closure is effective or not
    if( 5.0 < std::fabs( verTranslationDifference.z( ) ) )
    {
        //ds full information
        pEdgeLoopClosure->setInformation( static_cast< Eigen::Matrix< double, 6, 6 > >( 10*pEdgeLoopClosure->information( ) ) );
    }
    else
    {*/
        //ds damp z information
        pEdgeLoopClosure->setInformation( _getInformationNoZ( static_cast< Eigen::Matrix< double, 6, 6 > >( 10*pEdgeLoopClosure->information( ) ) ) );
        //pEdgeLoopClosure->setInformation( Eigen::Matrix< double, 6, 6 >::Zero( ) );
    //}

    return pEdgeLoopClosure;
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
                //ds updated estimate
                const CPoint3DWORLD vecPosition( pLandmark->vecPointXYZOptimized+p_vecTranslationToG2o );

                //ds if sane TODO make redundant
                if( m_dSanePositionThresholdL2 > vecPosition.squaredNorm( ) )
                {
                    //ds try to retrieve vertex
                    const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

                    //ds if found
                    if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
                    {
                        //ds extract and update the estimate
                        g2o::VertexPointXYZ* pVertex = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );
                        pVertex->setEstimate( vecPosition );

                        ++uLandmarksAlreadyInGraph;
                    }
                    else
                    {
                        //ds set landmark vertex
                        g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
                        pVertexLandmark->setEstimate( vecPosition );
                        pVertexLandmark->setId( pLandmark->uID );

                        //ds add vertex to optimizer
                        m_cOptimizerSparse.addVertex( pVertexLandmark );
                        m_vecLandmarksInGraph.push_back( pLandmark );
                    }
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

g2o::VertexSE3* Cg2oOptimizer::_setAndgetPose( std::vector< CKeyFrame* >::size_type& p_uIDKeyFrameFrom, CKeyFrame* pKeyFrameCurrent, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds compose transformation matrix
    Eigen::Isometry3d matTransformationLEFTtoWORLD = pKeyFrameCurrent->matTransformationLEFTtoWORLD;
    matTransformationLEFTtoWORLD.translation( ) += p_vecTranslationToG2o;

    //ds add current camera pose
    g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
    pVertexPoseCurrent->setEstimate( matTransformationLEFTtoWORLD );
    pVertexPoseCurrent->setId( pKeyFrameCurrent->uID+m_uIDShift );
    m_cOptimizerSparse.addVertex( pVertexPoseCurrent );

    //ds for trajectory only as well
    g2o::VertexSE3* pVertexPoseCurrentTrajectoryOnly = new g2o::VertexSE3( );
    pVertexPoseCurrentTrajectoryOnly->setEstimate( matTransformationLEFTtoWORLD );
    pVertexPoseCurrentTrajectoryOnly->setId( pKeyFrameCurrent->uID+1 );
    m_cOptimizerSparseTrajectoryOnly.addVertex( pVertexPoseCurrentTrajectoryOnly );

    //ds set up the edge to connect the poses
    g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

    //ds set viewpoints and measurement
    const g2o::HyperGraph::VertexIDMap::iterator itPoseFrom = m_cOptimizerSparse.vertices( ).find( p_uIDKeyFrameFrom+m_uIDShift );
    assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseFrom );
    g2o::VertexSE3* pVertexPoseFrom = dynamic_cast< g2o::VertexSE3* >( itPoseFrom->second );
    pEdgePoseFromTo->setVertex( 0, pVertexPoseFrom );
    pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
    pEdgePoseFromTo->setMeasurement( pVertexPoseFrom->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

    //ds compute information value based on absolute distance between keyframes
    const double dInformationFactor = 1.0/(1.0+pEdgePoseFromTo->measurement( ).translation( ).squaredNorm( ) );

    //ds new information matrix
    Eigen::Matrix< double, 6, 6 > matInformation( m_matInformationPose );
    matInformation.block< 3,3 >(0,0) *= dInformationFactor;

    //ds set specific information matrix (far connections between keyframes are less rigid)
    pEdgePoseFromTo->setInformation( matInformation );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgePoseFromTo );

    //ds find the corresponding pose in the trajectory only graph
    const g2o::HyperGraph::VertexIDMap::iterator itPoseFromTrajectoryOnly( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( p_uIDKeyFrameFrom+1 ) );
    assert( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) != itPoseFromTrajectoryOnly );

    //ds trajectory only
    g2o::EdgeSE3* pEdgePoseFromToTrajectoryOnly = new g2o::EdgeSE3( );
    pEdgePoseFromToTrajectoryOnly->setVertex( 0, dynamic_cast< g2o::VertexSE3* >( itPoseFromTrajectoryOnly->second ) );
    pEdgePoseFromToTrajectoryOnly->setVertex( 1, pVertexPoseCurrentTrajectoryOnly );
    pEdgePoseFromToTrajectoryOnly->setMeasurement( pEdgePoseFromTo->measurement( ) );
    Eigen::Matrix< double, 6, 6 > matInformationTrajectoryOnly( pEdgePoseFromToTrajectoryOnly->information( ) );
    matInformationTrajectoryOnly.block< 3,3 >(0,0) *= dInformationFactor;
    pEdgePoseFromToTrajectoryOnly->setInformation( matInformationTrajectoryOnly );
    m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgePoseFromToTrajectoryOnly );

    //ds add to control structure
    m_vecPoseEdges.push_back( pEdgePoseFromTo );
    m_mapEdgeIDs.insert( std::make_pair( pKeyFrameCurrent->uID, std::make_pair( pEdgePoseFromTo, pEdgePoseFromToTrajectoryOnly ) ) );

    return pVertexPoseCurrent;
}

void Cg2oOptimizer::_setLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent, const CKeyFrame::CMatchICP* p_pClosure, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds find the corresponding pose in the current graph
    assert( 0 != p_pClosure->pKeyFrameReference );
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

            //ds check if sane
            if( m_dSanePositionThresholdL2 > pVertex->estimate( ).squaredNorm( ) )
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

    if( 0 < uNumberOfErasedLandmarks )
    {
        std::printf( "[%06lu]<Cg2oOptimizer>(_applyOptimization) erased badly optimized landmarks: %lu\n", p_uFrame, uNumberOfErasedLandmarks );
    }
}

void Cg2oOptimizer::_applyOptimization( const UIDFrame& p_uFrame, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds update poses
    for( CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds find vertex in graph
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itKeyFrame( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

        //ds has to be found
        if( itKeyFrame != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds extract and update the keyframe
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itKeyFrame->second );
            pKeyFrame->matTransformationLEFTtoWORLD                = pVertex->estimate( );
            pKeyFrame->matTransformationLEFTtoWORLD.translation( ) -= p_vecTranslationToG2o;
            pKeyFrame->bIsOptimized                                = true;

            //ds relax the vertex again for next loop-closing optimization
            //pVertex->setFixed( false );
        }
        else
        {
            std::printf( "[%06lu]<Cg2oOptimizer>(_applyOptimization) caught invalid key frame ID: %lu\n", p_uFrame, pKeyFrame->uID );
        }
    }
}

const Eigen::Matrix< double, 6, 6 > Cg2oOptimizer::_getInformationNoZ( const Eigen::Matrix< double, 6, 6 >& p_matInformationIN ) const
{
    Eigen::Matrix< double, 6, 6 > matInformationOUT( p_matInformationIN );

    //ds lower z by a factor
    matInformationOUT(2,2) /= 100.0;

    return matInformationOUT;
}

void Cg2oOptimizer::_backPropagateTrajectoryToFull( const std::vector< CLoopClosureRaw >& p_vecClosures )
{
    //ds back propagate the trajectory only graph into the full one
    for( const CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds search vertices
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );
        const g2o::HyperGraph::VertexIDMap::iterator itPoseTrajectoryOnly( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrame->uID+1 ) );

        //ds must exist
        assert( m_cOptimizerSparse.vertices( ).end( )               != itPose );
        assert( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) != itPoseTrajectoryOnly );

        //ds //ds update estimate
        g2o::VertexSE3* pVertexPose               = dynamic_cast< g2o::VertexSE3* >( itPose->second );
        g2o::VertexSE3* pVertexPoseTrajectoryOnly = dynamic_cast< g2o::VertexSE3* >( itPoseTrajectoryOnly->second );
        pVertexPose->setEstimate( pVertexPoseTrajectoryOnly->estimate( ) );

        //ds fetch pose egdes
        const std::pair< g2o::EdgeSE3*, g2o::EdgeSE3* > prEdges( m_mapEdgeIDs[pKeyFrame->uID] );

        //ds update connecting edge measurement
        g2o::EdgeSE3* pEdge               = dynamic_cast< g2o::EdgeSE3* >( *pVertexPose->edges( ).find( prEdges.first ) );
        g2o::EdgeSE3* pEdgeTrajectoryOnly = dynamic_cast< g2o::EdgeSE3* >( *pVertexPoseTrajectoryOnly->edges( ).find( prEdges.second ) );
        pEdge->setMeasurement( pEdgeTrajectoryOnly->measurement( ) );
    }

    //ds back propagate loop closures to full graph
    for( CLoopClosureRaw cLoopClosure: p_vecClosures )
    {
        //ds search vertices in full graph
        const g2o::HyperGraph::VertexIDMap::iterator itPoseReference( m_cOptimizerSparse.vertices( ).find( cLoopClosure.uIDReference+m_uIDShift ) );
        const g2o::HyperGraph::VertexIDMap::iterator itPoseQuery( m_cOptimizerSparse.vertices( ).find( cLoopClosure.uIDQuery+m_uIDShift ) );

        //ds must exist
        assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseReference );
        assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseQuery );

        //ds find edge in optimized graph
        const g2o::HyperGraph::EdgeSet::iterator itEdgeLoopClosure( m_cOptimizerSparseTrajectoryOnly.edges( ).find( cLoopClosure.pEdgeLoopClosureReferenceToQuery ) );
        assert( m_cOptimizerSparseTrajectoryOnly.edges( ).end( ) != itEdgeLoopClosure );
        g2o::EdgeSE3* pEdgeLoopClosureInGraph = dynamic_cast< g2o::EdgeSE3* >( *itEdgeLoopClosure );

        //ds set loop closure edge for full graph
        g2o::EdgeSE3* pEdgeLoopClosure = new g2o::EdgeSE3( );
        pEdgeLoopClosure->setVertex( 0, dynamic_cast< g2o::VertexSE3* >( itPoseReference->second ) );
        pEdgeLoopClosure->setVertex( 1, dynamic_cast< g2o::VertexSE3* >( itPoseQuery->second ) );
        pEdgeLoopClosure->setMeasurement( pEdgeLoopClosureInGraph->measurement( ) );
        pEdgeLoopClosure->setInformation( m_matInformationLoopClosure );
        m_cOptimizerSparse.addEdge( pEdgeLoopClosure );
    }
}

void Cg2oOptimizer::_backPropagateTrajectoryToPure( )
{
    //ds back propagate the trajectory only graph into the full one
    for( const CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds search vertices
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );
        const g2o::HyperGraph::VertexIDMap::iterator itPoseTrajectoryOnly( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrame->uID+1 ) );

        //ds must exist (except for final optimization) - check if not
        if( ( m_cOptimizerSparse.vertices( ).end( )               == itPose               ) ||
            ( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) == itPoseTrajectoryOnly ) )
        {
            //ds back propagation failed
            std::printf( "<Cg2oOptimizer>(_backPropagateTrajectoryToPure) back propagation to trajectory only failed for frame ID: %lu\n", pKeyFrame->uID );
            return;
        }

        //ds //ds update estimate
        g2o::VertexSE3* pVertexPose               = dynamic_cast< g2o::VertexSE3* >( itPose->second );
        g2o::VertexSE3* pVertexPoseTrajectoryOnly = dynamic_cast< g2o::VertexSE3* >( itPoseTrajectoryOnly->second );
        pVertexPoseTrajectoryOnly->setEstimate( pVertexPose->estimate( ) );

        //ds fetch pose egdes
        const std::pair< g2o::EdgeSE3*, g2o::EdgeSE3* > prEdges( m_mapEdgeIDs[pKeyFrame->uID] );

        //ds update connecting edge measurement
        g2o::EdgeSE3* pEdge               = dynamic_cast< g2o::EdgeSE3* >( *pVertexPose->edges( ).find( prEdges.first ) );
        g2o::EdgeSE3* pEdgeTrajectoryOnly = dynamic_cast< g2o::EdgeSE3* >( *pVertexPoseTrajectoryOnly->edges( ).find( prEdges.second ) );
        pEdgeTrajectoryOnly->setMeasurement( pEdge->measurement( ) );
    }
}
