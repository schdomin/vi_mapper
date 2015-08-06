#include "CEpipolarDetectorBRIEF.h"

#include <opencv/highgui.h>
#include <Eigen/Core>
#include <fstream>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/edge_se3_pointxyz_uv.h"

#include "configuration/CConfigurationOpenCV.h"
#include "utility/CWrapperOpenCV.h"
#include "utility/CMiniVisionToolbox.h"
#include "exceptions/CExceptionNoMatchFound.h"
#include "configuration/CConfigurationCamera.h"

CEpipolarDetectorBRIEF::CEpipolarDetectorBRIEF( const uint32_t& p_uImageRows,
                                  const uint32_t& p_uImageCols,
                                  const bool& p_bDisplayImages,
                                  const uint32_t p_uFrequencyPlaybackHz ): m_uImageRows( p_uImageRows ),
                                                                           m_uImageCols( p_uImageCols ),
                                                                           m_uFrameCount( 0 ),
                                                                           m_vecTranslationLast( 1.0, 1.0, 1.0 ),
                                                                           m_dTranslationDeltaForMAPMeters( 0.5 ),
                                                                           m_cExtractorBRIEF( 64 ),
                                                                           m_cMatcherBRIEF( new cv::flann::LshIndexParams( 20, 10, 2 ) ),
                                                                           m_uKeyPointSizeLimit( 10 ),
                                                                           m_iSearchUMin( m_uKeyPointSizeLimit/2 ),
                                                                           m_iSearchUMax( m_uImageCols-m_uKeyPointSizeLimit/2 ),
                                                                           m_iSearchVMin( m_uKeyPointSizeLimit/2 ),
                                                                           m_iSearchVMax( m_uImageRows-m_uKeyPointSizeLimit/2 ),
                                                                           m_cSearchROI( cv::Rect( cv::Point2i( m_iSearchUMin, m_iSearchVMin ), cv::Point2i( m_iSearchUMax, m_iSearchVMax ) ) ),
                                                                           m_uNumberOfTilesBase( 3 ),
                                                                           m_fMatchingDistanceCutoffTracking( 50.0 ),
                                                                           m_fMatchingDistanceCutoffTriangulation( 75.0 ),
                                                                           m_uLimitLandmarksPerScan( 500 ),
                                                                           m_uMaximumNonMatches( 20 ),
                                                                           m_uVisibleLandmarksMinimum( 20 ),
                                                                           m_dMaximumDepthMeters( 1000.0 ),
                                                                           m_uAvailableLandmarkID( 0 ),
                                                                           m_bIsShutdownRequested( false ),
                                                                           m_dFrequencyPlaybackHz( p_uFrequencyPlaybackHz ),
                                                                           m_uFrequencyPlaybackDeltaHz( 50 ),
                                                                           m_iPlaybackSpeedupCounter( 0 ),
                                                                           m_cRandomGenerator( 1337 ),
                                                                           m_uFramesCurrentCycle( 0 ),
                                                                           m_dPreviousFrameRate( 0.0 ),
                                                                           m_cCameraLEFT( CConfigurationCamera::LEFT::cPinholeCamera ),
                                                                           m_cCameraRIGHT( CConfigurationCamera::RIGHT::cPinholeCamera ),
                                                                           m_cStereoCamera( m_cCameraLEFT, m_cCameraRIGHT )
{
    //ds initialize reference frames with black images
    m_matDisplayLowerReference = cv::Mat::zeros( m_uImageRows, 2*m_uImageCols, CV_8UC3 );

    //ds trajectory maps
    m_matTrajectoryXY = cv::Mat( 350, 350, CV_8UC3, CColorCodeBGR( 255, 255, 255 ) );
    m_matTrajectoryZ = cv::Mat( 350, 1500, CV_8UC3, CColorCodeBGR( 255, 255, 255 ) );

    //ds draw meters grid
    for( uint32_t x = 0; x < 350; x += 10 )
    {
        cv::line( m_matTrajectoryXY, cv::Point( x, 0 ),cv::Point( x, 350 ), CColorCodeBGR( 175, 175, 175 ) );
    }
    for( uint32_t x = 0; x < 1500; x += 10 )
    {
        cv::line( m_matTrajectoryZ, cv::Point( x, 0 ),cv::Point( x, 350 ), CColorCodeBGR( 175, 175, 175 ) );
    }
    for( uint32_t y = 0; y < 350; y += 10 )
    {
        cv::line( m_matTrajectoryXY, cv::Point( 0, y ),cv::Point( 350, y ), CColorCodeBGR( 175, 175, 175 ) );
        cv::line( m_matTrajectoryZ, cv::Point( 0, y ),cv::Point( 1500, y ), CColorCodeBGR( 175, 175, 175 ) );
    }

    //ds initialize the window
    cv::namedWindow( "stereo matching", cv::WINDOW_AUTOSIZE );

    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) instance allocated\n" );
}

CEpipolarDetectorBRIEF::~CEpipolarDetectorBRIEF( )
{

}

void CEpipolarDetectorBRIEF::receivevDataVIWithPose( std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageLeft, std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageRight, const txt_io::CIMUMessage& p_cIMU, const std::shared_ptr< txt_io::CPoseMessage >& p_cPose )
{
    //ds get images into opencv format
    const cv::Mat matImageLeft( p_cImageLeft->image( ) );
    const cv::Mat matImageRight( p_cImageRight->image( ) );

    //ds preprocessed images
    cv::Mat matPreprocessedLEFT;
    cv::Mat matPreprocessedRIGHT;

    //ds preprocess images
    cv::equalizeHist( matImageLeft, matPreprocessedLEFT );
    cv::equalizeHist( matImageRight, matPreprocessedRIGHT );
    m_cStereoCamera.undistortAndrectify( matPreprocessedLEFT, matPreprocessedRIGHT );

    //ds pose information
    Eigen::Isometry3d m_matTransformation;
    m_matTransformation.translation( ) = p_cPose->getPosition( );
    m_matTransformation.linear( )      = p_cPose->getOrientationMatrix( );

    //ds process images
    _trackLandmarksAuto( matPreprocessedLEFT, matPreprocessedRIGHT, m_matTransformation );
    //_trackLandmarksManual( matPreprocessedLEFT, matPreprocessedRIGHT, m_matTransformation );
}

//ds postprocessing
void CEpipolarDetectorBRIEF::solveAndOptimizeG2O( const std::string& p_strOutfile ) const
{
    //ds allocate an optimizer
    g2o::SparseOptimizer cOptimizer;
    cOptimizer.setVerbose( true );

    //ds set the solver
    g2o::BlockSolverX::LinearSolverType* pLinearSolver = new g2o::LinearSolverDense< g2o::BlockSolverX::PoseMatrixType> ( );
    g2o::BlockSolverX* pSolver                         = new g2o::BlockSolverX( pLinearSolver );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm    = new g2o::OptimizationAlgorithmLevenberg( pSolver );
    cOptimizer.setAlgorithm( pAlgorithm );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( m_cCameraLEFT.m_dFx/m_cCameraLEFT.m_uWidthPixel, m_cCameraLEFT.m_dFy/m_cCameraLEFT.m_uHeightPixel, m_cCameraLEFT.m_dCx/m_cCameraLEFT.m_uWidthPixel, m_cCameraLEFT.m_dCy/m_cCameraLEFT.m_uHeightPixel );
    pCameraParametersLEFT->setId( 0 );
    cOptimizer.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( m_cCameraRIGHT.m_dFx/m_cCameraRIGHT.m_uWidthPixel, m_cCameraRIGHT.m_dFy/m_cCameraRIGHT.m_uHeightPixel, m_cCameraRIGHT.m_dCx/m_cCameraRIGHT.m_uWidthPixel, m_cCameraRIGHT.m_dCy/m_cCameraRIGHT.m_uHeightPixel );
    pCameraParametersRIGHT->setId( 1 );
    cOptimizer.addParameter( pCameraParametersRIGHT );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( 0 );

    //ds add landmarks
    for( const CLandmarkInWorldFrame& cLandmark: m_vecLandmarks )
    {
        //ds set landmark vertex
        g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
        pVertexLandmark->setEstimate( cLandmark.vecPositionXYZ );
        pVertexLandmark->setId( uNextAvailableUID );

        assert( cLandmark.uID == uNextAvailableUID );

        //ds add vertex to optimizer
        cOptimizer.addVertex( pVertexLandmark );
        ++uNextAvailableUID;
    }

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPose = new g2o::VertexSE3( );
    pVertexPose->setEstimate( m_vecLogMeasurementPoints.front( ).first );
    pVertexPose->setId( uNextAvailableUID );
    pVertexPose->setFixed( true );
    cOptimizer.addVertex( pVertexPose );
    ++uNextAvailableUID;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< std::pair< Eigen::Isometry3d, std::vector< CMeasurementLandmark > > >::const_iterator pPose = m_vecLogMeasurementPoints.begin( )+1; pPose != m_vecLogMeasurementPoints.end( ); ++pPose )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pPose->first );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cOptimizer.addVertex( pVertexPoseCurrent );
        ++uNextAvailableUID;

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cOptimizer.vertices( ).find( uNextAvailableUID-2 )->second );

        //ds set up the edge
        g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

        //ds set viewpoints and measurement
        pEdgePoseFromTo->setVertex( 0, pVertexPosePrevious );
        pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
        pEdgePoseFromTo->setMeasurement( pVertexPosePrevious->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

        //ds add to optimizer
        cOptimizer.addEdge( pEdgePoseFromTo );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark& pLandmark: pPose->second )
        {
            //ds set up the edges
            g2o::EdgeSE3PointXYZUV* pEdgeLandmarkCoordinates      = new g2o::EdgeSE3PointXYZUV( );
            g2o::EdgeSE3PointXYZDisparity* pEdgeLandmarkDisparity = new g2o::EdgeSE3PointXYZDisparity( );

            //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
            g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( cOptimizer.vertices( ).find( pLandmark.uID )->second );

            //ds set viewpoints and measurements
            pEdgeLandmarkCoordinates->setVertex( 0, pVertexPoseCurrent );
            pEdgeLandmarkCoordinates->setVertex( 1, pVertexLandmark );
            pEdgeLandmarkCoordinates->setMeasurement( static_cast< g2o::Vector2D >( pLandmark.vecPositionUV ) );
            pEdgeLandmarkCoordinates->setParameterId( 0, 0 );

            //lo dasdasd
            pEdgeLandmarkDisparity->setVertex( 0, pVertexPoseCurrent );
            pEdgeLandmarkDisparity->setVertex( 1, pVertexLandmark );
            pEdgeLandmarkDisparity->setParameterId( 0, 0 );

            //ds add to optimizer
            cOptimizer.addEdge( pEdgeLandmarkCoordinates );
            cOptimizer.addEdge( pEdgeLandmarkDisparity );
        }
    }

    cOptimizer.save( p_strOutfile.c_str( ) );

    //ds optimize!
    //cOptimizer.initializeOptimization( );
    //cOptimizer.computeActiveErrors( );
    //cOptimizer.optimize( 1 );
}

void CEpipolarDetectorBRIEF::_trackLandmarksAuto( cv::Mat& p_matImageLEFT, const cv::Mat& p_matImageRIGHT, const Eigen::Isometry3d& p_matTransformation )
{
    //ds increment count
    ++m_uFrameCount;

    //ds current translation
    const CPoint3DWORLD vecTranslationCurrent( p_matTransformation.translation( ) );

    //ds get delta to evaluate precision
    const double dTransformationDelta( CMiniVisionToolbox::getTransformationDelta( m_matPreviousTransformationLeft, p_matTransformation ) );
    m_matPreviousTransformationLeft = p_matTransformation;

    //ds draw position on trajectory mat
    cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matTransformation.translation( )( 0 )*10, 175-p_matTransformation.translation( )( 1 )*10 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-dTransformationDelta*1000 ), 1, cv::Scalar( 0, 0, 255 ), -1 );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLeft;
    cv::Mat matDisplayRight;

    //ds get images to triple channel for colored display
    cv::cvtColor( p_matImageLEFT, matDisplayLeft, cv::COLOR_GRAY2BGR );
    cv::cvtColor( p_matImageRIGHT, matDisplayRight, cv::COLOR_GRAY2BGR );

    //ds get clean copies
    const cv::Mat matDisplayLeftClean( matDisplayLeft.clone( ) );
    const cv::Mat matDisplayRightClean( matDisplayRight.clone( ) );

    //ds draw the search space
    cv::rectangle( matDisplayLeft, m_cSearchROI, CColorCodeBGR( 255, 255, 255 ) );



    //ds draw current lines and retrieve active landmarks
    const std::vector< CMeasurementLandmark > vecVisibleLandmarks( _getVisibleLandmarksOnEpipolarLineEssential( p_matTransformation, matDisplayLeft, p_matImageLEFT, 10+dTransformationDelta*500 ) );

    //ds add to data structure if delta is sufficiently high
    if( m_dTranslationDeltaForMAPMeters < ( vecTranslationCurrent-m_vecTranslationLast ).squaredNorm( ) )
    {
        m_vecLogMeasurementPoints.push_back( std::pair< Eigen::Isometry3d, std::vector< CMeasurementLandmark > >( p_matTransformation, vecVisibleLandmarks ) );
        m_vecTranslationLast = vecTranslationCurrent;
        std::printf( "<>(_trackLandmarksAuto) stashed pose ( number: %lu ) with landmarks ( total: %lu )\n", m_vecLogMeasurementPoints.size( ), vecVisibleLandmarks.size( ) );
    }



    //ds check if we have to detect new landmarks
    if( m_uVisibleLandmarksMinimum > vecVisibleLandmarks.size( ) )
    {
        //ds clean the lower display
        cv::hconcat( matDisplayLeftClean, matDisplayRightClean, m_matDisplayLowerReference );

        //ds detect landmarks
        m_vecActiveMeasurementPoints.push_back( std::pair< Eigen::Isometry3d, std::vector< CLandmark > >( p_matTransformation, _getLandmarksGFTT( m_matDisplayLowerReference, p_matImageLEFT, p_matImageRIGHT, m_uNumberOfTilesBase, p_matTransformation ) ) );
    }

    //ds info
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-vecVisibleLandmarks.size( )*2 ), 1, cv::Scalar( 255, 0, 0 ), -1 );

    //ds build display mat
    cv::Mat matDisplayUpper = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpper );
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );

    //ds display
    cv::putText( matDisplayComplete, "FPS " + std::to_string( m_dPreviousFrameRate ), cv::Point2i( m_iSearchUMin+2, m_iSearchVMin+12 ), cv::FONT_HERSHEY_PLAIN, 0.9, CColorCodeBGR( 0, 0, 255 ) );
    cv::imshow( "stereo matching", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", m_matTrajectoryXY );
    cv::imshow( "some stuff", m_matTrajectoryZ );

    //ds if there was a keystroke
    int iLastKeyStroke( cv::waitKey( 1 ) );
    if( -1 != iLastKeyStroke )
    {
        //ds user input - reset frame rate counting
        m_uFramesCurrentCycle = 0;

        //ds evaluate keystroke
        switch( iLastKeyStroke )
        {
            case CConfigurationOpenCV::KeyStroke::iEscape:
            {
                _shutDown( );
                return;
            }
            case CConfigurationOpenCV::KeyStroke::iNumpadMinus:
            {
                _slowDown( );
                return;
            }
            case CConfigurationOpenCV::KeyStroke::iNumpadPlus:
            {
                _speedUp( );
                return;
            }
            default:
            {
                std::printf( "<>(_trackLandmarksAuto) unknown keystroke: %i\n", iLastKeyStroke );
                return;
            }
        }
    }
    else
    {
        _updateFrameRateDisplay( 10 );
    }
}

void CEpipolarDetectorBRIEF::_trackLandmarksManual( cv::Mat& p_matImageLEFT, const cv::Mat& p_matImageRIGHT, const Eigen::Isometry3d& p_matTransformation )
{
    //ds increment count
    ++m_uFrameCount;

    //ds current translation
    const CPoint3DWORLD vecTranslationCurrent( p_matTransformation.translation( ) );

    //ds get delta to evaluate precision
    const double dTransformationDelta( CMiniVisionToolbox::getTransformationDelta( m_matPreviousTransformationLeft, p_matTransformation ) );
    m_matPreviousTransformationLeft = p_matTransformation;

    //ds log info
    //std::printf( "[%lu] transformation delta: %f\n", m_uFrameCount, dTransformationDelta );

    //ds draw position on trajectory mat
    cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matTransformation.translation( )( 0 )*10, 175-p_matTransformation.translation( )( 1 )*10 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-dTransformationDelta*1000 ), 1, cv::Scalar( 0, 0, 255 ), -1 );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLeft;
    cv::Mat matDisplayRight;

    //ds get images to triple channel for colored display
    cv::cvtColor( p_matImageLEFT, matDisplayLeft, cv::COLOR_GRAY2BGR );
    cv::cvtColor( p_matImageRIGHT, matDisplayRight, cv::COLOR_GRAY2BGR );

    //ds get clean copies
    const cv::Mat matDisplayLeftClean( matDisplayLeft.clone( ) );
    const cv::Mat matDisplayRightClean( matDisplayRight.clone( ) );

    //ds draw the search space
    cv::rectangle( matDisplayLeft, m_cSearchROI, CColorCodeBGR( 255, 255, 255 ) );



    //ds draw current lines and retrieve active landmarks
    const std::vector< CMeasurementLandmark > vecVisibleLandmarks( _getVisibleLandmarksOnEpipolarLineEssential( p_matTransformation, matDisplayLeft, p_matImageLEFT, 10+dTransformationDelta*500 ) );

    //ds add to data structure if delta is sufficiently high
    if( m_dTranslationDeltaForMAPMeters < ( vecTranslationCurrent-m_vecTranslationLast ).squaredNorm( ) )
    {
        m_vecLogMeasurementPoints.push_back( std::pair< Eigen::Isometry3d, std::vector< CMeasurementLandmark > >( p_matTransformation, vecVisibleLandmarks ) );
        m_vecTranslationLast = vecTranslationCurrent;
        std::printf( "stashed pose with landmarks (current poses: %lu)\n", m_vecLogMeasurementPoints.size( ) );
    }

    //ds info
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-vecVisibleLandmarks.size( )*2 ), 1, cv::Scalar( 255, 0, 0 ), -1 );

    //ds build display mat
    cv::Mat matDisplayUpper          = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::Mat matDisplayUpperTemporary = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpper );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpperTemporary );

    //ds show the image
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
    cv::putText( matDisplayComplete, "FPS " + std::to_string( m_dPreviousFrameRate ), cv::Point2i( m_iSearchUMin+2, m_iSearchVMin+12 ), cv::FONT_HERSHEY_PLAIN, 0.9, CColorCodeBGR( 0, 0, 255 ) );
    cv::imshow( "stereo matching", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", m_matTrajectoryXY );
    cv::imshow( "some stuff", m_matTrajectoryZ );

    //ds if there was a keystroke
    int iLastKeyStroke( cv::waitKey( 1 ) );
    if( -1 != iLastKeyStroke )
    {
        //ds user input - reset frame rate counting
        m_uFramesCurrentCycle = 0;

        //ds evaluate keystroke
        switch( iLastKeyStroke )
        {
            case CConfigurationOpenCV::KeyStroke::iEscape:
            {
                _shutDown( );
                return;
            }
            case CConfigurationOpenCV::KeyStroke::iNumpadMinus:
            {
                _slowDown( );
                return;
            }
            case CConfigurationOpenCV::KeyStroke::iNumpadPlus:
            {
                _speedUp( );
                return;
            }
            case CConfigurationOpenCV::KeyStroke::iSpace:
            {
                break;
            }
            default:
            {
                std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) unknown keystroke: %i\n", iLastKeyStroke );
                return;
            }
        }

        //ds reset keystroke
        iLastKeyStroke = -1;

        //ds update image
        cv::hconcat( matDisplayLeftClean, matDisplayRightClean, matDisplayUpper );

        //ds redraw
        cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) moved distance: %f requesting feature detection at frame: %lu\n", vecTranslationCurrent.squaredNorm( ), m_uFrameCount );

        //ds add a new scan vector
        m_vecActiveMeasurementPoints.push_back( std::pair< Eigen::Isometry3d, std::vector< CLandmark > >( p_matTransformation, _getLandmarksGFTT( matDisplayUpper, p_matImageLEFT, p_matImageRIGHT, m_uNumberOfTilesBase, p_matTransformation ) ) );

        cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );
        cv::imshow( "trajectory (x,y)", m_matTrajectoryXY );

        //ds escape with space
        while( CConfigurationOpenCV::KeyStroke::iSpace != iLastKeyStroke )
        {
            //ds check key
            iLastKeyStroke = cv::waitKey( 1 );

            //ds evaluate keystroke
            switch( iLastKeyStroke )
            {
                case CConfigurationOpenCV::KeyStroke::iEscape:
                {
                    _shutDown( );
                    return;
                }
                case CConfigurationOpenCV::KeyStroke::iNumpadMinus:
                {
                    _slowDown( );
                    break;
                }
                case CConfigurationOpenCV::KeyStroke::iNumpadPlus:
                {
                    _speedUp( );
                    break;
                }
                case CConfigurationOpenCV::KeyStroke::iBackspace:
                {
                    //ds clean all references
                    m_vecActiveMeasurementPoints.clear( );
                    std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) cleared all scan points\n" );
                    return;
                }
                default:
                {
                    //ds nothing to do
                    break;
                }
            }
        }

        //ds reset keystroke
        iLastKeyStroke = -1;

        //ds get a copy of the upper to the lower display
        m_matDisplayLowerReference = matDisplayUpper.clone( );
        cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );

        //ds mark position of user input (persistently)
        cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matTransformation.translation( )( 0 )*10, 175-p_matTransformation.translation( )( 1 )*10 ), 5, CColorCodeBGR( 0, 255, 0 ), 1 );
        cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matTransformation.translation( )( 2 )*100 ), 5, CColorCodeBGR( 0, 255, 0 ), 1 );
    }
    else
    {
        _updateFrameRateDisplay( 10 );
    }
}

const std::vector< CMeasurementLandmark > CEpipolarDetectorBRIEF::_getVisibleLandmarksOnEpipolarLineEssential( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_iLineLength )
{
    //ds detected landmarks at this position - VISIBLE != ACTIVE
    std::vector< CMeasurementLandmark > vecVisibleLandmarksTotal;

    //ds active scan points after detection
    std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmark > > > vecActiveMeasurementPoints;

    //ds check all scan points we have
    for( std::pair< Eigen::Isometry3d, std::vector< CLandmark > >& cScanPoint: m_vecActiveMeasurementPoints )
    {
        //ds currently visible landmarks for this scan
        std::vector< CLandmark > vecActiveLandmarksPerScan;

        //ds match count
        uint64_t uCountMatches( 0 );
        uint64_t uTotalLandmarks( cScanPoint.second.size( ) );

        //ds compute essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( cScanPoint.first, p_matCurrentTransformation ) );

        //ds loop over the points for the current scan
        for( CLandmark& cLandmarkReference: cScanPoint.second )
        {
            //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
            const Eigen::Vector3d vecCoefficients( matEssential*cLandmarkReference.vecPositionUVLEFTReference );

            //ds compute maximum and minimum points (from top to bottom line)
            const CPoint2DInCameraFrameHomogenized& vecLastDetection( cLandmarkReference.vecPositionUVLast );
            const int32_t iULastDetection( m_cCameraLEFT.getU( vecLastDetection(0) ) );

            //ds get back to pixel coordinates
            int32_t iUMinimum( std::max( iULastDetection-p_iLineLength, m_iSearchUMin ) );
            int32_t iUMaximum( std::min( iULastDetection+p_iLineLength, m_iSearchUMax ) );
            int32_t iVForUMinimum( m_cCameraLEFT.getDenormalizedY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( iUMinimum ) )/vecCoefficients(1) ) );
            int32_t iVForUMaximum( m_cCameraLEFT.getDenormalizedY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( iUMaximum ) )/vecCoefficients(1) ) );

            //ds compute V limits on the fly
            const int32_t iVCenter( ( iVForUMinimum+iVForUMaximum )/2 );
            const int32_t iVLimitMaximum( std::min( iVCenter+p_iLineLength, m_iSearchVMax ) );
            const int32_t iVLimitMinimum( std::max( iVCenter-p_iLineLength, m_iSearchVMin ) );

            //ds negative slope (max v is also at max u)
            if( iVForUMaximum > iVForUMinimum )
            {
                //ds check if the line is out of sight
                if( m_iSearchVMin > iVForUMaximum || m_iSearchVMax < iVForUMinimum )
                {
                    std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark out of sight\n" );
                    continue;
                }
                else
                {
                    //ds adjust ROI (recompute U)
                    if( iVLimitMinimum > iVForUMinimum )
                    {
                        iVForUMinimum = iVLimitMinimum;
                        iUMinimum = m_cCameraLEFT.getU( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMinimum ) )/vecCoefficients(0) );
                    }
                    if( iVLimitMaximum < iVForUMaximum )
                    {
                        iVForUMaximum = iVLimitMaximum;
                        iUMaximum = m_cCameraLEFT.getU( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMaximum ) )/vecCoefficients(0) );
                    }

                    //ds check if computed line is invalid (happens due to border shifting)
                    if( iVForUMinimum > iVForUMaximum )
                    {
                        std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark out of sight\n" );
                        continue;
                    }
                }
            }

            //ds positive slope (max v is at min u)
            else
            {
                //ds check if the line is out of sight
                if( m_iSearchVMin > iVForUMinimum || m_iSearchVMax < iVForUMaximum )
                {
                    std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark out of sight\n" );
                    continue;
                }
                else
                {
                    //ds adjust ROI (recompute U)
                    if( iVLimitMaximum < iVForUMinimum )
                    {
                        iVForUMinimum = iVLimitMaximum;
                        iUMinimum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMaximum ) )/vecCoefficients(0) );
                    }
                    if( iVLimitMinimum > iVForUMaximum )
                    {
                        iVForUMaximum = iVLimitMinimum;
                        iUMaximum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMinimum ) )/vecCoefficients(0) );
                    }

                    //ds swap required for ROI generation
                    std::swap( iVForUMinimum, iVForUMaximum );

                    //ds check if computed line is invalid (happens due to border shifting)
                    if( iVForUMinimum > iVForUMaximum )
                    {
                        std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark out of sight\n" );
                        continue;
                    }
                }
            }

            //ds compute pixel ranges to sample
            const int32_t iDeltaU( iUMaximum-iUMinimum );
            const int32_t iDeltaV( iVForUMaximum-iVForUMinimum );

            //ds escape for single points
            if( 0 == iDeltaU && 0 == iDeltaV )
            {
                std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark out of sight\n" );
                continue;
            }

            //ds keypoint vectors and descriptors to match
            std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
            vecReferenceKeyPoints[0] = cLandmarkReference.cKeyPoint;
            const CDescriptor& matReferenceDescriptor( cLandmarkReference.matDescriptor );
            const float& fKeyPointSize( vecReferenceKeyPoints[0].size );

            //ds sample the larger range
            if( iDeltaU > iDeltaV )
            {
                try
                {
                    //ds get the match over U
                    cLandmarkReference.vecPositionUVLast = _getMatchSampleUBRIEF( p_matDisplay, p_matImage, iUMinimum, iDeltaU, vecCoefficients, matReferenceDescriptor, m_cExtractorBRIEF, m_cMatcherBRIEF, fKeyPointSize );
                    vecVisibleLandmarksTotal.push_back( CMeasurementLandmark( cLandmarkReference.uID, cLandmarkReference.vecPositionUVLast ) );
                    vecActiveLandmarksPerScan.push_back( cLandmarkReference );
                    ++uCountMatches;
                }
                catch( const CExceptionNoMatchFound& p_eException )
                {
                    //ds check if we dont have to drop the landmark yet
                    if( cLandmarkReference.uFailedSubsequentTrackings < m_uMaximumNonMatches )
                    {
                        //ds reset reference point and mark non-match
                        cLandmarkReference.vecPositionUVLast = vecLastDetection;
                        ++cLandmarkReference.uFailedSubsequentTrackings;
                        vecActiveLandmarksPerScan.push_back( cLandmarkReference );
                    }
                    else
                    {
                        std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark dropped\n" );
                    }
                }
            }
            else
            {
                try
                {
                    //ds get the match over V
                    cLandmarkReference.vecPositionUVLast = _getMatchSampleVBRIEF( p_matDisplay, p_matImage, iVForUMinimum, iDeltaV, vecCoefficients, matReferenceDescriptor, m_cExtractorBRIEF, m_cMatcherBRIEF, fKeyPointSize );
                    vecVisibleLandmarksTotal.push_back( CMeasurementLandmark( cLandmarkReference.uID, cLandmarkReference.vecPositionUVLast ) );
                    vecActiveLandmarksPerScan.push_back( cLandmarkReference );
                    ++uCountMatches;
                }
                catch( const CExceptionNoMatchFound& p_eException )
                {
                    //ds check if we dont have to drop the landmark yet
                    if( cLandmarkReference.uFailedSubsequentTrackings < m_uMaximumNonMatches )
                    {
                        //ds reset reference point and mark non-match
                        cLandmarkReference.vecPositionUVLast = vecLastDetection;
                        ++cLandmarkReference.uFailedSubsequentTrackings;
                        vecActiveLandmarksPerScan.push_back( cLandmarkReference );
                    }
                    else
                    {
                        std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark dropped\n" );
                    }
                }
            }
        }

        //ds compute match ratio
        std::printf( "<>(_drawProjectedEpipolarLineEssential) matching ratio: %f (%lu/%lu)\n", static_cast< double >( uCountMatches )/uTotalLandmarks, uCountMatches, uTotalLandmarks );

        //ds check if we still have active landmarks
        if( !vecActiveLandmarksPerScan.empty( ) )
        {
            //ds update scan point vector
            vecActiveMeasurementPoints.push_back( std::pair< Eigen::Isometry3d, std::vector< CLandmark > >( cScanPoint.first, vecActiveLandmarksPerScan ) );
        }
        else
        {
            //ds erase the scan point
            std::printf( "<>(_drawProjectedEpipolarLineEssential) erased scan point\n" );
        }
    }

    //ds update scan points handle
    m_vecActiveMeasurementPoints.swap( vecActiveMeasurementPoints );

    //std::printf( "<>(_drawProjectedEpipolarLineEssential) active landmarks: %lu\n", vecDetectedLandmarks.size( ) );
    //std::printf( "<>(_drawProjectedEpipolarLineEssential) current vector size: %lu\n", m_vecVerticesWithLandmarks.capacity( )*sizeof( std::pair< Eigen::Isometry3d, std::vector< std::pair< uint64_t, CPoint2DNormalized > > > ) );

    //ds return active landmarks
    return vecVisibleLandmarksTotal;
}

const std::vector< CLandmark > CEpipolarDetectorBRIEF::_getLandmarksGFTT( cv::Mat& p_matDisplay, const cv::Mat& p_matImageLEFT, const cv::Mat& p_matImageRIGHT, const uint32_t& p_uTileNumberBase, const Eigen::Isometry3d& p_matTransformation )
{
    //ds compute tile sizes
    const double dTileWidth( static_cast< double >( m_cSearchROI.width )/p_uTileNumberBase );
    const double dTileHeight( static_cast< double >( m_cSearchROI.height )/p_uTileNumberBase );
    const uint32_t uFeaturesPerTile( m_uLimitLandmarksPerScan/( p_uTileNumberBase*p_uTileNumberBase ) );

    //ds current reference points
    std::vector< CLandmark > vecLandmarksTracking;

    //ds create tiles
    for( uint32_t u = 0; u < p_uTileNumberBase; ++u )
    {
        for( uint32_t v = 0; v < p_uTileNumberBase; ++v )
        {
            //ds features in tile frame
            std::vector< cv::Point2f > vecFeaturesPerTile;

            //ds compute rectangle points
            const double dULU( m_iSearchUMin+u*dTileWidth );
            const double dULV( m_iSearchVMin+v*dTileHeight );
            const double dLRU( dULU+dTileWidth );
            const double dLRV( dULV+dTileHeight );
            const cv::Rect cTileROI( cv::Point2d( dULU, dULV ), cv::Point2d( dLRU, dLRV ) );

            cv::rectangle( p_matDisplay, cTileROI, CColorCodeBGR( 255, 255, 255 ) );

            //ds setup the detection mask
            cv::Mat matMask = cv::Mat::zeros( m_uImageRows, m_uImageCols, CV_8U );
            matMask( cTileROI ) = 1;

            //ds detect features inside this roi
            cv::goodFeaturesToTrack( p_matImageLEFT, vecFeaturesPerTile, uFeaturesPerTile, 0.5, m_uKeyPointSizeLimit, matMask );

            //ds for the found features
            for( const cv::Point2f& ptFeature: vecFeaturesPerTile )
            {
                //ds compute the descriptor
                std::vector< cv::KeyPoint> vecKeyPoint( 1 );
                vecKeyPoint[0] = cv::KeyPoint( ptFeature, m_uKeyPointSizeLimit );
                CDescriptor matReferenceDescriptor;
                m_cExtractorBRIEF.compute( p_matImageLEFT, vecKeyPoint, matReferenceDescriptor );

                //ds if there is a descriptor for this keypoint
                if( !vecKeyPoint.empty( ) )
                {
                    try
                    {
                        //ds triangulate the point
                        const CPoint3DCAMERA vecPointTriangulated( _getPointTriangulated( p_matImageRIGHT, vecKeyPoint[0], matReferenceDescriptor, m_cExtractorBRIEF, m_cMatcherBRIEF, m_fMatchingDistanceCutoffTriangulation ) );

                        //ds check if point is in front of camera an not more than a defined distance away
                        if( 0 < vecPointTriangulated(2) && m_dMaximumDepthMeters > vecPointTriangulated(2) )
                        {
                            //ds normalize coordinates for tracking
                            const CPoint2DInCameraFrameHomogenized vecPointNormalized( m_cCameraLEFT.getNormalHomogenized( vecKeyPoint[0] ) );

                            //ds set current reference point
                            vecLandmarksTracking.push_back( CLandmark( m_uAvailableLandmarkID, vecKeyPoint[0], matReferenceDescriptor, vecPointNormalized ) );

                            //ds compute triangulated point in world frame
                            const CPoint3DWORLD vecPointTriangulatedWorld( p_matTransformation*vecPointTriangulated );

                            //ds register landmark in total vector (only touched in here)
                            m_vecLandmarks.push_back( CLandmarkInWorldFrame( m_uAvailableLandmarkID, vecPointTriangulatedWorld ) );

                            //ds next landmark id
                            ++m_uAvailableLandmarkID;

                            //ds draw detected point
                            cv::line( p_matDisplay, ptFeature, cv::Point2f( ptFeature.x+m_uImageCols, ptFeature.y ), CColorCodeBGR( 175, 175, 175 ) );
                            cv::circle( p_matDisplay, ptFeature, 3, CColorCodeBGR( 0, 255, 0 ), -1 );
                            cv::circle( p_matDisplay, ptFeature, vecKeyPoint[0].size, CColorCodeBGR( 255, 0, 0 ), 1 );

                            //ds draw landmark in world (2d)
                            cv::circle( m_matTrajectoryXY, cv::Point2d( 50+vecPointTriangulatedWorld(0)*10, 175-vecPointTriangulatedWorld(1)*10 ), 3, CColorCodeBGR( 0, 165, 255 ), -1 );

                            //ds draw reprojections of triangulation
                            const CPoint3DHomogenized vecLandmarkHomo( vecPointTriangulated(0), vecPointTriangulated(1), vecPointTriangulated(2), 1.0 );
                            CPoint2DHomogenized vecLandmarkRIGHT( m_cCameraRIGHT.getHomogeneousProjection( vecLandmarkHomo ) );
                            cv::circle( p_matDisplay, cv::Point2i( vecLandmarkRIGHT(0)+m_uImageCols, vecLandmarkRIGHT(1) ), 3, CColorCodeBGR( 255, 0, 0 ), -1 );
                        }
                        else
                        {
                            cv::circle( p_matDisplay, ptFeature, 3, CColorCodeBGR( 0, 255, 0 ), -1 );
                            cv::circle( p_matDisplay, ptFeature, vecKeyPoint[0].size, CColorCodeBGR( 0, 0, 255 ) );

                            std::printf( "<>(_getLandmarksGFTT) could not find match for keypoint (invalid depth: %f m)\n", vecPointTriangulated(2) );
                        }
                    }
                    catch( const CExceptionNoMatchFound& p_cException )
                    {
                        cv::circle( p_matDisplay, ptFeature, 3, CColorCodeBGR( 0, 255, 0 ), -1 );
                        cv::circle( p_matDisplay, ptFeature, vecKeyPoint[0].size, CColorCodeBGR( 0, 0, 255 ) );

                        std::printf( "<>(_getLandmarksGFTT) could not find match for keypoint (matching distance to high)\n" );
                    }
                }
                else
                {
                    cv::circle( p_matDisplay, ptFeature, 3, CColorCodeBGR( 0, 255, 0 ), -1 );
                    cv::circle( p_matDisplay, ptFeature, vecKeyPoint[0].size, CColorCodeBGR( 0, 0, 255 ) );

                    std::printf( "<>(_getLandmarksGFTT) could not compute reference descriptor\n" );
                }
            }
        }
    }

    std::printf( "<>(_getLandmarksGFTT) added new landmarks: %lu\n", vecLandmarksTracking.size( ) );

    //ds return found landmarks
    return vecLandmarksTracking;
}

const CPoint3DCAMERA CEpipolarDetectorBRIEF::_getPointTriangulated( const cv::Mat& p_matImageRIGHT,
                                                                           const cv::KeyPoint& p_cKeyPoint,
                                                                           const CDescriptor& p_matReferenceDescriptor,
                                                                           const cv::DescriptorExtractor& p_cExtractor,
                                                                           const cv::DescriptorMatcher& p_cMatcher,
                                                                           const double& p_dMatchingDistanceCutoff ) const
{
    //ds buffer keypoint size
    const float& fKeyPointSize( p_cKeyPoint.size );

    //ds get keypoint to eigen space
    const CPoint2DInCameraFrame vecReference( CWrapperOpenCV::fromCVVector( p_cKeyPoint.pt ) );

    //ds right keypoint vector (check the full range)
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uImageCols );

    //ds set the keypoints
    for( uint32_t uX = 0; uX < m_uImageCols; ++uX )
    {
        vecPoolKeyPoints[uX] = cv::KeyPoint( uX, vecReference(1), fKeyPointSize );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    p_cExtractor.compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    p_cMatcher.match( p_matReferenceDescriptor, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "no match found" );
    }

    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( p_dMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds get the matching keypoint
        const CPoint2DInCameraFrame vecMatch( CWrapperOpenCV::fromCVVector( vecPoolKeyPoints[vecMatches[0].trainIdx].pt ) );

        //ds triangulate 3d point
        return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( vecReference, vecMatch, m_cCameraLEFT.m_matProjection, m_cCameraRIGHT.m_matProjection );
    }
    else
    {
        throw CExceptionNoMatchFound( "invalid match quality" );
    }
}

CPoint2DInCameraFrameHomogenized CEpipolarDetectorBRIEF::_getMatchSampleUBRIEF( cv::Mat& p_matDisplay,
                                     const cv::Mat& p_matImage,
                                     const int32_t& p_iUMinimum,
                                     const int32_t& p_iDeltaU,
                                     const Eigen::Vector3d& p_vecCoefficients,
                                     const cv::Mat& p_matReferenceDescriptor,
                                     const cv::DescriptorExtractor& p_cExtractor,
                                     const cv::DescriptorMatcher& p_cMatcher,
                                     const float& p_fKeyPointSize ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaU );

    //ds sample over U
    for( int32_t u = 0; u < p_iDeltaU; ++u )
    {
        //ds compute corresponding V coordinate
        const uint32_t uU( p_iUMinimum+u );
        const double dY( -( p_vecCoefficients(2)+p_vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( uU ) )/p_vecCoefficients(1) );
        const double dV( m_cCameraLEFT.getDenormalizedY( dY ) );

        //ds add keypoint
        vecPoolKeyPoints[u] = cv::KeyPoint( uU, dV, p_fKeyPointSize );
        cv::circle( p_matDisplay, cv::Point2i( uU, dV ), 1, CColorCodeBGR( 255, 0, 0 ), -1 );
    }

    return _getMatchBRIEF( p_matDisplay, p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_cExtractor, p_cMatcher );
}
CPoint2DInCameraFrameHomogenized CEpipolarDetectorBRIEF::_getMatchSampleVBRIEF( cv::Mat& p_matDisplay,
                                     const cv::Mat& p_matImage,
                                     const int32_t& p_iVMinimum,
                                     const int32_t& p_iDeltaV,
                                     const Eigen::Vector3d& p_vecCoefficients,
                                     const cv::Mat& p_matReferenceDescriptor,
                                     const cv::DescriptorExtractor& p_cExtractor,
                                     const cv::DescriptorMatcher& p_cMatcher,
                                     const float& p_fKeyPointSize ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaV );

    //ds sample over U
    for( int32_t v = 0; v < p_iDeltaV; ++v )
    {
        //ds compute corresponding U coordinate
        const uint32_t uV( p_iVMinimum+v );
        const double dX( -( p_vecCoefficients(2)+p_vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( uV ) )/p_vecCoefficients(0) );
        const double dU( m_cCameraLEFT.getDenormalizedX( dX ) );

        //ds add keypoint
        vecPoolKeyPoints[v] = cv::KeyPoint( dU, uV, p_fKeyPointSize );
        cv::circle( p_matDisplay, cv::Point2i( dU, uV ), 1, CColorCodeBGR( 255, 0, 0 ), -1 );
    }

    return _getMatchBRIEF( p_matDisplay, p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_cExtractor, p_cMatcher );
}

CPoint2DInCameraFrameHomogenized CEpipolarDetectorBRIEF::_getMatchBRIEF( cv::Mat& p_matDisplay,
                                                           const cv::Mat& p_matImage,
                                                           std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                                           const cv::Mat& p_matReferenceDescriptor,
                                                           const cv::DescriptorExtractor& p_cExtractor,
                                                           const cv::DescriptorMatcher& p_cMatcher ) const
{
    //ds descriptor pool
    cv::Mat matPoolDescriptors;

    //ds compute descriptors of current search area
    p_cExtractor.compute( p_matImage, p_vecPoolKeyPoints, matPoolDescriptors );

    //ds escape if we didnt find any descriptors
    if( p_vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "could not find a matching descriptor (empty KeyPoint pool)" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    p_cMatcher.match( p_matReferenceDescriptor, matPoolDescriptors, vecMatches );

    //ds escape for no matches
    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "could not find any matches (empty DMatches pool)" );
    }

    //ds buffer first match
    const cv::DMatch& cBestMatch( vecMatches[0] );

    //ds check if we are in the range (works for negative ids as well)
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( cBestMatch.trainIdx ) < p_vecPoolKeyPoints.size( ) );

    if( m_fMatchingDistanceCutoffTracking > cBestMatch.distance )
    {
        cv::circle( p_matDisplay, p_vecPoolKeyPoints[cBestMatch.trainIdx].pt, 1, CColorCodeBGR( 0, 255, 0 ), -1 );

        //ds return the match
        return CPoint2DInCameraFrameHomogenized( m_cCameraLEFT.getNormalHomogenized( p_vecPoolKeyPoints[cBestMatch.trainIdx].pt ) );
    }
    else
    {
        //ds nothing found
        //std::printf( "<>(_getMatchSampleU) dropped match (distance: %f)\n", cBestMatch.distance );
        throw CExceptionNoMatchFound( "could not find a matching descriptor (matching distance to high)" );
    }
}

void CEpipolarDetectorBRIEF::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<>(_shutDown) termination requested, detector disabled\n" );
}

void CEpipolarDetectorBRIEF::_speedUp( )
{
    ++m_iPlaybackSpeedupCounter;
    m_dFrequencyPlaybackHz += std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;
    std::printf( "<>(_speedUp) increased playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
}

void CEpipolarDetectorBRIEF::_slowDown( )
{
    m_dFrequencyPlaybackHz -= std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;

    //ds 12 fps minimum (one of 2 images 10 imu messages and 1 pose: 13-1)
    if( 12 < m_dFrequencyPlaybackHz )
    {
        std::printf( "<>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
        --m_iPlaybackSpeedupCounter;
    }
    else
    {
        m_dFrequencyPlaybackHz = 12.5;
        std::printf( "<>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
    }
}

void CEpipolarDetectorBRIEF::_updateFrameRateDisplay( const uint32_t& p_uFrameProbeRange )
{
    //ds check if we can compute the frame rate
    if( p_uFrameProbeRange == m_uFramesCurrentCycle )
    {
        //ds get time delta
        const double dDuration( ( std::chrono::duration< double >( std::chrono::system_clock::now( )-m_tmStart ) ).count( ) );

        //ds compute framerate
        m_dPreviousFrameRate = p_uFrameProbeRange/dDuration;

        //ds enable new measurement
        m_uFramesCurrentCycle = 0;
    }

    //ds check if its the first frame since the last count
    if( 0 == m_uFramesCurrentCycle )
    {
        //ds stop time
        m_tmStart = std::chrono::system_clock::now( );
    }

    //ds count frames
    ++m_uFramesCurrentCycle;
}
