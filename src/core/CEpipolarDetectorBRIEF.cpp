#include "CEpipolarDetectorBRIEF.h"

#include <opencv/highgui.h>
#include <Eigen/Core>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "configuration/CConfigurationCamera.h"

CEpipolarDetectorBRIEF::CEpipolarDetectorBRIEF( const uint32_t& p_uImageRows,
                                  const uint32_t& p_uImageCols,
                                  const bool& p_bDisplayImages,
                                  const uint32_t p_uFrequencyPlaybackHz ): m_uImageRows( p_uImageRows ),
                                                                           m_uImageCols( p_uImageCols ),
                                                                           m_uFrameCount( 0 ),
                                                                           m_cFLANNMatcher( new cv::flann::LshIndexParams( 20, 10, 2 ) ),
                                                                           m_uKeyPointSizeLimit( 10 ),
                                                                           m_iSearchUMin( m_uKeyPointSizeLimit/2 ),
                                                                           m_iSearchUMax( m_uImageCols-m_uKeyPointSizeLimit/2 ),
                                                                           m_iSearchVMin( m_uKeyPointSizeLimit/2 ),
                                                                           m_iSearchVMax( m_uImageRows-m_uKeyPointSizeLimit/2 ),
                                                                           m_cSearchROI( cv::Rect( cv::Point2i( m_iSearchUMin, m_iSearchVMin ), cv::Point2i( m_iSearchUMax, m_iSearchVMax ) ) ),
                                                                           m_fMatchingDistanceCutoff( 50.0 ),
                                                                           m_uLimitFeaturesPerScan( 500 ),
                                                                           m_uMaximumNonMatches( 10 ),
                                                                           m_uActiveLandmarksMinimum( 20 ),
                                                                           m_uAvailableLandmarkID( 0 ),
                                                                           m_bDisplayImages( p_bDisplayImages ),
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
    m_matReferenceFrameLeft    = cv::Mat::zeros( m_uImageRows, m_uImageCols, CV_8UC1 );
    m_matReferenceFrameRight   = cv::Mat::zeros( m_uImageRows, m_uImageCols, CV_8UC1 );
    m_matDisplayLowerReference = cv::Mat::zeros( m_uImageRows, 2*m_uImageCols, CV_8UC3 );

    //ds trajectory maps
    m_matTrajectoryXY = cv::Mat( 350, 350, CV_8UC3, CColorCode( 255, 255, 255 ) );
    m_matTrajectoryZ = cv::Mat( 350, 1500, CV_8UC3, CColorCode( 255, 255, 255 ) );

    //ds draw meters grid
    for( uint32_t x = 0; x < 350; x += 10 )
    {
        cv::line( m_matTrajectoryXY, cv::Point( x, 0 ),cv::Point( x, 350 ), CColorCode( 175, 175, 175 ) );
    }
    for( uint32_t x = 0; x < 1500; x += 10 )
    {
        cv::line( m_matTrajectoryZ, cv::Point( x, 0 ),cv::Point( x, 350 ), CColorCode( 175, 175, 175 ) );
    }
    for( uint32_t y = 0; y < 350; y += 10 )
    {
        cv::line( m_matTrajectoryXY, cv::Point( 0, y ),cv::Point( 350, y ), CColorCode( 175, 175, 175 ) );
        cv::line( m_matTrajectoryZ, cv::Point( 0, y ),cv::Point( 1500, y ), CColorCode( 175, 175, 175 ) );
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

    //ds pose information
    Eigen::Isometry3d m_matTransformation;
    m_matTransformation.translation( ) = p_cPose->getPosition( );
    m_matTransformation.linear( )      = p_cPose->getOrientationMatrix( );

    //ds detect features
    if( m_bDisplayImages )
    {
        //_localizeAutoSURF( matImageLeft, matImageRight, m_matTransformation );
        _localize( matImageLeft, matImageRight, m_matTransformation );
    }
}

//ds postprocessing
void CEpipolarDetectorBRIEF::dumpVerticesWithLandmarks( const std::string& p_strOutfile ) const
{
    //ds open the file
    std::ofstream cOutfile( p_strOutfile, std::ofstream::out );

    //ds loop over the vector
    for( uint32_t u = 0; u < m_vecPosesWithLandmarks.size( ); ++u )
    {
        //ds get current isometry
        const Eigen::Isometry3d matTransformation( m_vecPosesWithLandmarks[u].first );
        const Eigen::Vector3d vecPosition( matTransformation.translation( ) );

        //ds c huarg
        char chLineVertexTrack[100];
        snprintf( chLineVertexTrack, 100, "VERTEX_TRACKXYZ %u %.5f %.5f %.5f\n", u, vecPosition(0), vecPosition(1), vecPosition(2) );
        cOutfile << chLineVertexTrack;
    }

    cOutfile.close( );
}
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
    g2o::ParameterCamera* pCameraParameters = new g2o::ParameterCamera( );
    pCameraParameters->setKcam( m_cCameraLEFT.m_dFx, m_cCameraLEFT.m_dFy, m_cCameraLEFT.m_dCx, m_cCameraLEFT.m_dCy );
    pCameraParameters->setId( 0 );
    cOptimizer.addParameter( pCameraParameters );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( 0 );

    //ds add landmarks
    for( const std::pair< uint64_t, CPoint3DInWorldFrame >& prLandmark: m_vecLandmarks )
    {
        //ds set landmark vertex
        g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
        pVertexLandmark->setEstimate( prLandmark.second );
        pVertexLandmark->setId( uNextAvailableUID );

        //ds add vertex to optimizer
        cOptimizer.addVertex( pVertexLandmark );
        ++uNextAvailableUID;
    }

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPose = new g2o::VertexSE3( );
    pVertexPose->setEstimate( m_vecPosesWithLandmarks.front( ).first );
    pVertexPose->setId( uNextAvailableUID );
    cOptimizer.addVertex( pVertexPose );
    ++uNextAvailableUID;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< std::pair< Eigen::Isometry3d, std::vector< std::pair< uint64_t, CPoint2DNormalized > > > >::const_iterator pPose = m_vecPosesWithLandmarks.begin( )+1; pPose != m_vecPosesWithLandmarks.end( ); ++pPose )
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

        //ds set viewpoints
        pEdgePoseFromTo->setVertex( 0, pVertexPosePrevious );
        pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
        pEdgePoseFromTo->setMeasurement( pVertexPosePrevious->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

        //ds add to optimizer
        cOptimizer.addEdge( pEdgePoseFromTo );



        //ds check visible landmarks and add the edges for the current pose
        for( const std::pair< uint64_t, CPoint2DNormalized >& pLandmark: pPose->second )
        {
            //ds set up the edge
            g2o::EdgeProjectXYZ2UV* pEdgeLandmark = new g2o::EdgeProjectXYZ2UV( );
            //g2o::EdgePointXYZ* pEdgeLandmark = new g2o::EdgePointXYZ( );

            //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
            g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( cOptimizer.vertices( ).find( pLandmark.first )->second );

            //ds set viewpoints
            pEdgeLandmark->setVertex( 0, pVertexPoseCurrent );
            pEdgeLandmark->setVertex( 1, pVertexLandmark );

            pEdgeLandmark->setMeasurement( pLandmark.second.head< 2 >( ) );
            pEdgeLandmark->setParameterId( 0, 0 );

            //ds add to optimizer
            cOptimizer.addEdge( pEdgeLandmark );
        }
    }

    cOptimizer.save( p_strOutfile.c_str( ) );

    //ds optimize!
    //cOptimizer.initializeOptimization( );
    //cOptimizer.computeActiveErrors( );
    //cOptimizer.optimize( 1 );
}

void CEpipolarDetectorBRIEF::_localize( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation )
{
    //ds get delta to evaluate precision
    const double dTransformationDelta( CMiniVisionToolbox::getTransformationDelta( m_matPreviousTransformationLeft, p_matCurrentTransformation ) );

    //ds update references
    m_matPreviousTransformationLeft = p_matCurrentTransformation;

    //ds increment count
    ++m_uFrameCount;

    //ds log info
    //std::printf( "[%lu] transformation delta: %f\n", m_uFrameCount, dTransformationDelta );

    //ds input mats
    cv::Mat matLeft;
    cv::Mat matRight;

    //ds preprocess images
    cv::equalizeHist( p_matImageLeft, matLeft );
    cv::equalizeHist( p_matImageRight, matRight );
    m_cStereoCamera.undistortAndrectify( matLeft, matRight );

    //ds draw position on trajectory mat
    cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matCurrentTransformation.translation( )( 0 )*10, 175-p_matCurrentTransformation.translation( )( 1 )*10 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-dTransformationDelta*1000 ), 1, cv::Scalar( 0, 0, 255 ), -1 );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLeft;
    cv::Mat matDisplayRight;

    //ds get images to triple channel for colored display
    cv::cvtColor( matLeft, matDisplayLeft, cv::COLOR_GRAY2BGR );
    cv::cvtColor( matRight, matDisplayRight, cv::COLOR_GRAY2BGR );

    //ds get clean copies
    const cv::Mat matDisplayLeftClean( matDisplayLeft.clone( ) );
    const cv::Mat matDisplayRightClean( matDisplayRight.clone( ) );

    //ds draw the search space
    cv::rectangle( matDisplayLeft, cv::Point2i( m_iSearchUMin, m_iSearchVMin ), cv::Point2i( m_iSearchUMax, m_iSearchVMax ), CColorCode( 255, 255, 255 ) );


    //ds draw current lines and check how many landmarks have been redetected
    const uint64_t uActiveLandmarks( _matchProjectedEpipolarLineEssential( p_matCurrentTransformation, matDisplayLeft, matLeft, 10+dTransformationDelta*500 ) );


    //ds check if we have to detect new landmarks
    if( m_uActiveLandmarksMinimum > uActiveLandmarks )
    {
        //ds clean the lower display
        cv::hconcat( matDisplayLeftClean, matDisplayRightClean, m_matDisplayLowerReference );

        //ds detect landmarks
        m_vecScanPoints.push_back( std::pair< std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > >, Eigen::Isometry3d >( _getLandmarksGFTT( m_matDisplayLowerReference, p_matImageLeft, 3 ), p_matCurrentTransformation ) );
    }

    //ds info
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-uActiveLandmarks*2 ), 1, cv::Scalar( 255, 0, 0 ), -1 );

    //ds build display mat
    cv::Mat matDisplayUpper          = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::Mat matDisplayUpperTemporary = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpper );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpperTemporary );

    //ds show the image
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
    cv::putText( matDisplayComplete, "FPS " + std::to_string( m_dPreviousFrameRate ), cv::Point2i( m_iSearchUMin+2, m_iSearchVMin+12 ), cv::FONT_HERSHEY_PLAIN, 0.9, CColorCode( 0, 0, 255 ) );
    cv::imshow( "stereo matching", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", m_matTrajectoryXY );
    cv::imshow( "some stuff", m_matTrajectoryZ );
    int iLastKeyStroke( cv::waitKey( 1 ) );

    //ds if there was a keystroke
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
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) moved distance: %f requesting feature detection at frame: %lu\n", p_matCurrentTransformation.translation( ).squaredNorm( ), m_uFrameCount );

        //ds add a new scan vector
        m_vecScanPoints.push_back( std::pair< std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > >, Eigen::Isometry3d >( _getLandmarksGFTT( matDisplayUpper, p_matImageLeft, 3 ), p_matCurrentTransformation ) );

        cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );

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
                    m_vecScanPoints.clear( );
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
        cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matCurrentTransformation.translation( )( 0 )*10, 175-p_matCurrentTransformation.translation( )( 1 )*10 ), 5, CColorCode( 0, 255, 0 ), 1 );
        cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 5, CColorCode( 0, 255, 0 ), 1 );
    }
    else
    {
        _updateFrameRateDisplay( 10 );
    }
}

uint64_t CEpipolarDetectorBRIEF::_matchProjectedEpipolarLineEssential( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_iLineLength )
{
    //ds detected landmarks at this position
    std::vector< std::pair< uint64_t, CPoint2DNormalized > > vecDetectedLandmarks;

    //ds check all scan points we have
    for( std::vector< std::pair< std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > >, Eigen::Isometry3d > >::iterator pScanPoint = m_vecScanPoints.begin( ); pScanPoint != m_vecScanPoints.end( ); ++pScanPoint )
    {
        //ds match count
        uint64_t uCountMatches( 0 );
        uint64_t uTotalLandmarks( pScanPoint->first.size( ) );

        //ds compute essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( pScanPoint->second, p_matCurrentTransformation ) );

        //ds loop over the points for the current scan
        for( std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > >::iterator pReferencePoint = pScanPoint->first.begin( ); pReferencePoint != pScanPoint->first.end( ); ++pReferencePoint )
        {
            //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
            const Eigen::Vector3d vecCoefficients( matEssential*std::get< 3 >( *pReferencePoint ) );

            //ds compute maximum and minimum points (from top to bottom line)
            const CPoint2DNormalized& vecLastDetection( std::get< 4 >( *pReferencePoint ) );
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
                    pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
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
                        pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
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
                    pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
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
                        pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
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
                pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
                continue;
            }

            //ds keypoint vectors and descriptors to match
            std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
            vecReferenceKeyPoints[0] = std::get< 1 >( *pReferencePoint );
            cv::Mat& matReferenceDescriptor( std::get< 2 >( *pReferencePoint ) );

            //ds sample the larger range
            if( iDeltaU > iDeltaV )
            {
                try
                {
                    //ds get the match over U
                    std::get< 4 >( *pReferencePoint ) = _getMatchSampleUBRIEF( p_matDisplay, p_matImage, iUMinimum, iDeltaU, vecCoefficients, matReferenceDescriptor );
                    vecDetectedLandmarks.push_back( std::pair< uint64_t, CPoint2DNormalized >( std::get< 0 >( *pReferencePoint ), std::get< 4 >( *pReferencePoint ) ) );
                    ++uCountMatches;
                }
                catch( const CExceptionNoMatchFound& p_eException )
                {
                    //ds check if we dont have to drop the landmark yet
                    if( std::get< 5 >( *pReferencePoint ) < m_uMaximumNonMatches )
                    {
                        //ds reset reference point and mark non-match
                        std::get< 4 >( *pReferencePoint ) = vecLastDetection;
                        ++std::get< 5 >( *pReferencePoint );
                    }
                    else
                    {
                        std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark dropped\n" );
                        pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
                    }
                }
            }
            else
            {
                try
                {
                    //ds get the match over V
                    std::get< 4 >( *pReferencePoint ) = _getMatchSampleVBRIEF( p_matDisplay, p_matImage, iVForUMinimum, iDeltaV, vecCoefficients, matReferenceDescriptor );
                    vecDetectedLandmarks.push_back( std::pair< uint64_t, CPoint2DNormalized >( std::get< 0 >( *pReferencePoint ), std::get< 4 >( *pReferencePoint ) ) );
                    ++uCountMatches;
                }
                catch( const CExceptionNoMatchFound& p_eException )
                {
                    //ds check if we dont have to drop the landmark yet
                    if( std::get< 5 >( *pReferencePoint ) < m_uMaximumNonMatches )
                    {
                        //ds reset reference point and mark non-match
                        std::get< 4 >( *pReferencePoint ) = vecLastDetection;
                        ++std::get< 5 >( *pReferencePoint );
                    }
                    else
                    {
                        std::printf( "<>(_drawProjectedEpipolarLineEssential) landmark dropped\n" );
                        pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
                    }
                }
            }
        }

        //ds compute match ratio
        std::printf( "<>(_drawProjectedEpipolarLineEssential) matching ratio: %f (%lu/%lu)\n", static_cast< double >( uCountMatches )/uTotalLandmarks, uCountMatches, uTotalLandmarks );

        //ds check if we have to remove the vector
        if( pScanPoint->first.empty( ) )
        {
            pScanPoint = m_vecScanPoints.erase( pScanPoint )-1;
            std::printf( "<>(_drawProjectedEpipolarLineEssential) erased scan point\n" );
        }
    }

    //ds add to data structure
    m_vecPosesWithLandmarks.push_back( std::pair< Eigen::Isometry3d, std::vector< std::pair< uint64_t, CPoint2DNormalized > > >( p_matCurrentTransformation, vecDetectedLandmarks ) );

    //std::printf( "<>(_drawProjectedEpipolarLineEssential) active landmarks: %lu\n", vecDetectedLandmarks.size( ) );
    //std::printf( "<>(_drawProjectedEpipolarLineEssential) current vector size: %lu\n", m_vecVerticesWithLandmarks.capacity( )*sizeof( std::pair< Eigen::Isometry3d, std::vector< std::pair< uint64_t, CPoint2DNormalized > > > ) );

    //ds return number of currently active landmarks
    return vecDetectedLandmarks.size( );
}

std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > > CEpipolarDetectorBRIEF::_getLandmarksGFTT( cv::Mat& p_matDisplay, const cv::Mat& p_matImage, const uint32_t& p_uTileNumberBase )
{
    //ds draw roi
    cv::rectangle( p_matDisplay, m_cSearchROI, CColorCode( 255, 255, 255 ) );

    //ds compute tile sizes
    const double dTileWidth( static_cast< double >( m_cSearchROI.width )/p_uTileNumberBase );
    const double dTileHeight( static_cast< double >( m_cSearchROI.height )/p_uTileNumberBase );
    const uint32_t uFeaturesPerTile( m_uLimitFeaturesPerScan/( p_uTileNumberBase*p_uTileNumberBase ) );

    //ds feature vector
    std::vector< cv::KeyPoint> vecKeyPoints;

    //ds create tiles
    for( uint32_t u = 0; u < p_uTileNumberBase; ++u )
    {
        for( uint32_t v = 0; v < p_uTileNumberBase; ++v )
        {
            //ds compute rectangle points
            const double dULU( m_iSearchUMin+u*dTileWidth );
            const double dULV( m_iSearchVMin+v*dTileHeight );
            const double dLRU( dULU+dTileWidth );
            const double dLRV( dULV+dTileHeight );
            const cv::Rect cTileROI( cv::Point2d( dULU, dULV ), cv::Point2d( dLRU, dLRV ) );

            //ds features in tile frame
            std::vector< cv::Point2f > vecFeaturesShifted;

            //ds detect features inside this roi
            cv::goodFeaturesToTrack( p_matImage( cTileROI ), vecFeaturesShifted, uFeaturesPerTile, 0.5, 10 );

            //ds copy the feature information
            for( const cv::Point2f& ptFeature: vecFeaturesShifted )
            {
                vecKeyPoints.push_back( cv::KeyPoint( cv::Point2f( dULU+ptFeature.x, dULV+ptFeature.y ), m_uKeyPointSizeLimit ) );
            }

            cv::rectangle( p_matDisplay, cTileROI, CColorCode( 255, 255, 255 ) );
        }
    }

    //ds compute descriptors
    CDescriptor matReferenceDescriptors;
    m_cExtractorBRIEF.compute( p_matImage, vecKeyPoints, matReferenceDescriptors );

    std::printf( "<>(_getLandmarksGFTT) computed descriptors: %lu\n", vecKeyPoints.size( ) );

    //ds current reference points
    std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > > vecLandmarks;

    //ds register all features
    for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
    {
        //ds get the keypoint
        const cv::KeyPoint& cKeyPoint( vecKeyPoints[u] );

        //ds normalize coordinates
        const Eigen::Vector3d vecPointNormalized( m_cCameraLEFT.getNormalized( cKeyPoint ) );

        //ds set current reference point
        vecLandmarks.push_back( std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t >( m_uAvailableLandmarkID, cKeyPoint, matReferenceDescriptors.row( u ), vecPointNormalized, vecPointNormalized, 0 ) );

        //ds register landmark in total vector (only touched in here)
        m_vecLandmarks.push_back( std::pair< uint64_t, CPoint3DInWorldFrame >( m_uAvailableLandmarkID, CPoint3DInWorldFrame( vecPointNormalized ) ) );

        //ds new feature id
        ++m_uAvailableLandmarkID;

        //ds draw detected point
        cv::circle( p_matDisplay, cKeyPoint.pt, 2, CColorCode( 0, 255, 0 ), -1 );
        cv::circle( p_matDisplay, cKeyPoint.pt, cKeyPoint.size, CColorCode( 255, 0, 0 ), 1 );
    }

    std::printf( "<>(_getLandmarksGFTT) useful keypoints: %lu\n", vecLandmarks.size( ) );
    std::printf( "<>(_getLandmarksGFTT) total recorded landmarks: %lu\n", m_uAvailableLandmarkID );

    return vecLandmarks;
}

CPoint2DNormalized CEpipolarDetectorBRIEF::_getMatchSampleUBRIEF( cv::Mat& p_matDisplay,
                                     const cv::Mat& p_matImage,
                                     const int32_t& p_iUMinimum,
                                     const int32_t& p_iDeltaU,
                                     const Eigen::Vector3d& p_vecCoefficients,
                                     const cv::Mat& p_matReferenceDescriptor ) const
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
        vecPoolKeyPoints[u] = cv::KeyPoint( uU, dV, m_uKeyPointSizeLimit );
        cv::circle( p_matDisplay, cv::Point2i( uU, dV ), 1, CColorCode( 255, 0, 0 ), -1 );
    }

    return _getMatchBRIEF( p_matDisplay, p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor );
}
CPoint2DNormalized CEpipolarDetectorBRIEF::_getMatchSampleVBRIEF( cv::Mat& p_matDisplay,
                                     const cv::Mat& p_matImage,
                                     const int32_t& p_iVMinimum,
                                     const int32_t& p_iDeltaV,
                                     const Eigen::Vector3d& p_vecCoefficients,
                                     const cv::Mat& p_matReferenceDescriptor ) const
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
        vecPoolKeyPoints[v] = cv::KeyPoint( dU, uV, m_uKeyPointSizeLimit );
        cv::circle( p_matDisplay, cv::Point2i( dU, uV ), 1, CColorCode( 255, 0, 0 ), -1 );
    }

    return _getMatchBRIEF( p_matDisplay, p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor );
}

CPoint2DNormalized CEpipolarDetectorBRIEF::_getMatchBRIEF( cv::Mat& p_matDisplay,
                                                           const cv::Mat& p_matImage,
                                                           std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                                           const cv::Mat& p_matReferenceDescriptor ) const
{
    //ds descriptor pool
    cv::Mat matPoolDescriptors;

    //ds compute descriptors of current search area
    m_cExtractorBRIEF.compute( p_matImage, p_vecPoolKeyPoints, matPoolDescriptors );

    //ds escape if we didnt find any descriptors
    if( 0 == p_vecPoolKeyPoints.size( ) )
    {
        throw CExceptionNoMatchFound( "could not find a matching descriptor (empty KeyPoint pool)" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_cFLANNMatcher.match( p_matReferenceDescriptor, matPoolDescriptors, vecMatches );

    //ds buffer first match
    const cv::DMatch& cBestMatch( vecMatches[0] );

    //ds evaluate matching id
    const uint32_t uTrainIndex( cBestMatch.trainIdx );

    //ds check if we are in the range (works for negative ids as well)
    if( uTrainIndex < p_vecPoolKeyPoints.size( ) )
    {
        //ds get the matching keypoint
        cv::Point2f ptMatch( p_vecPoolKeyPoints[ uTrainIndex ].pt );

        if( m_fMatchingDistanceCutoff < cBestMatch.distance )
        {
            //ds drop the match
            //std::printf( "<>(_getMatchSampleU) dropped match (distance: %f)\n", cBestMatch.distance );

            //ds nothing found
            throw CExceptionNoMatchFound( "could not find a matching descriptor (matching distance to high)" );
        }
        else
        {
            cv::circle( p_matDisplay, ptMatch, 1, CColorCode( 0, 255, 0 ), -1 );

            //ds return the match
            return CPoint2DNormalized( m_cCameraLEFT.getNormalized( ptMatch ) );
        }
    }
    else
    {
        throw CExceptionNoMatchFound( "could not find a matching descriptor (received invalid ID from kNN matcher)" );
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
