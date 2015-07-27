#include "CEpipolarDetector.h"

#include <opencv/highgui.h>
#include <Eigen/Core>

#include "configuration/CConfigurationCamera.h"

CEpipolarDetector::CEpipolarDetector( const uint32_t& p_uImageRows,
                                  const uint32_t& p_uImageCols,
                                  const bool& p_bDisplayImages,
                                  const uint32_t p_uFrequencyPlaybackHz ): m_uImageRows( p_uImageRows ),
                                                                           m_uImageCols( p_uImageCols ),
                                                                           m_uFrameCount( 0 ),
                                                                           m_cDetectorSURF( cv::SurfFeatureDetector( 400 ) ),
                                                                           m_cExtractorSURF( cv::SurfDescriptorExtractor( 400 ) ),
                                                                           m_cFLANNMatcher( new cv::flann::LshIndexParams( 20, 10, 2 ) ),
                                                                           m_uKeyPointSizeLimit( 10 ),
                                                                           m_dMatchingDistanceCutoff( 100 ),
                                                                           m_uFeaturesCap( 100 ),
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

CEpipolarDetector::~CEpipolarDetector( )
{

}

void CEpipolarDetector::receivevDataVIWithPose( std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageLeft, std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageRight, const txt_io::CIMUMessage& p_cIMU, const std::shared_ptr< txt_io::CPoseMessage >& p_cPose )
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
        _localizeAutoBRIEF( matImageLeft, matImageRight, m_matTransformation );
    }
}

void CEpipolarDetector::_localizeAutoBRIEF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation )
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
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-dTransformationDelta*1000 ), 1, cv::Scalar( 0, 0, 255 ), -1 );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLeft;
    cv::Mat matDisplayRight;

    //ds get images to triple channel for colored display
    cv::cvtColor( matLeft, matDisplayLeft, cv::COLOR_GRAY2BGR );
    cv::cvtColor( matRight, matDisplayRight, cv::COLOR_GRAY2BGR );

    const cv::Mat matDisplayLeftClean( matDisplayLeft.clone( ) );
    const cv::Mat matDisplayRightClean( matDisplayRight.clone( ) );

    //ds build display mat
    cv::Mat matDisplayUpper          = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::Mat matDisplayUpperTemporary = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpper );



    //ds draw current lines
    _drawProjectedEpipolarLineEssentialBRIEF( p_matCurrentTransformation, matDisplayLeft, matLeft, 10+dTransformationDelta*500 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpperTemporary );


    //ds show the image
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
    cv::putText( matDisplayComplete, "FPS " + std::to_string( m_dPreviousFrameRate ), cv::Point2i( 0, 10 ), cv::FONT_HERSHEY_PLAIN, 0.9, CColorCodeBGR( 0, 0, 255 ) );
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

        //ds check current displacement
        const double dRelativeDistanceMeters( p_matCurrentTransformation.translation( ).squaredNorm( ) );

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) moved distance: %f requesting feature detection at frame: %lu\n", dRelativeDistanceMeters, m_uFrameCount );

        ///ds detect features
        std::vector< cv::Point2f > vecFeaturePoints;
        cv::goodFeaturesToTrack( p_matImageLeft, vecFeaturePoints, 100, 0.1, 2 );
        std::vector< cv::KeyPoint> vecKeyPoints( vecFeaturePoints.size( ) );

        //ds copy the feature information
        for( uint32_t u = 0; u < vecFeaturePoints.size( ); ++u )
        {
            vecKeyPoints[u] = cv::KeyPoint( vecFeaturePoints[u], m_uKeyPointSizeLimit );
        }

        //ds compute descriptors
        CDescriptor matReferenceDescriptors;
        m_cExtractorBRIEF.compute( p_matImageLeft, vecKeyPoints, matReferenceDescriptors );

        //std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) detected features: %lu\n", vecFeaturePoints.size( ) );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) computed descriptors: %lu\n", vecKeyPoints.size( ) );

        //ds current reference points
        std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > > vecReferencePoints;

        //ds register all features
        for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
        {
            //ds get the keypoint
            const cv::KeyPoint& cKeyPoint( vecKeyPoints[u] );

            const Eigen::Vector3d vecPointNormalized( m_cCameraLEFT.getNormalHomogenized( cKeyPoint ) );

            //ds set current reference point
            vecReferencePoints.push_back( std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized >( cKeyPoint, matReferenceDescriptors.row( u ), vecPointNormalized, vecPointNormalized ) );

            //ds draw detected point
            cv::circle( matDisplayUpper, cKeyPoint.pt, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
            cv::circle( matDisplayUpper, cKeyPoint.pt, cKeyPoint.size, CColorCodeBGR( 255, 0, 0 ), 1 );
        }

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) useful keypoints: %lu\n", vecReferencePoints.size( ) );

        //ds add a new scan vector
        m_vecScanPoints.push_back( std::pair< std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > >, Eigen::Isometry3d >( vecReferencePoints, p_matCurrentTransformation ) );

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
        cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matCurrentTransformation.translation( )( 0 )*10, 175-p_matCurrentTransformation.translation( )( 1 )*10 ), 5, CColorCodeBGR( 0, 255, 0 ), 1 );
        cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 5, CColorCodeBGR( 0, 255, 0 ), 1 );
    }
    else
    {
        _updateFrameRateDisplay( 10 );
    }
}

void CEpipolarDetector::_localizeAutoSURF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation )
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
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-dTransformationDelta*1000 ), 1, cv::Scalar( 0, 0, 255 ), -1 );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLeft;
    cv::Mat matDisplayRight;

    //ds get images to triple channel for colored display
    cv::cvtColor( matLeft, matDisplayLeft, cv::COLOR_GRAY2BGR );
    cv::cvtColor( matRight, matDisplayRight, cv::COLOR_GRAY2BGR );

    const cv::Mat matDisplayLeftClean( matDisplayLeft.clone( ) );
    const cv::Mat matDisplayRightClean( matDisplayRight.clone( ) );

    //ds build display mat
    cv::Mat matDisplayUpper          = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::Mat matDisplayUpperTemporary = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpper );



    //ds draw current lines
    _drawProjectedEpipolarLineEssentialSURF( p_matCurrentTransformation, matDisplayLeft, matLeft, dTransformationDelta*1000 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpperTemporary );


    //ds show the image
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
    cv::putText( matDisplayComplete, "FPS " + std::to_string( m_dPreviousFrameRate ), cv::Point2i( 0, 10 ), cv::FONT_HERSHEY_PLAIN, 0.9, CColorCodeBGR( 0, 0, 255 ) );
    cv::imshow( "stereo matching", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", m_matTrajectoryXY );
    cv::imshow( "some stuff", m_matTrajectoryZ );
    int iLastKeyStroke( cv::waitKey( 0 ) );

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

        //ds check current displacement
        const double dRelativeDistanceMeters( p_matCurrentTransformation.translation( ).squaredNorm( ) );

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) moved distance: %f requesting feature detection at frame: %lu\n", dRelativeDistanceMeters, m_uFrameCount );

        //ds detect features
        //std::vector< cv::Point2f > vecFeaturePoints;
        //cv::goodFeaturesToTrack( p_matImageLeft, vecFeaturePoints, 100, 0.1, 5 );
        //std::vector< cv::KeyPoint> vecKeyPoints( vecFeaturePoints.size( ) );

        //ds copy the features
        //for( uint32_t u = 0; u < vecFeaturePoints.size( ); ++u )
        //{
        //    vecKeyPoints[u] = cv::KeyPoint( vecFeaturePoints[u], 40 );
        //}

        std::vector< cv::KeyPoint> vecKeyPoints;
        m_cDetectorSURF.detect( p_matImageLeft, vecKeyPoints );

        //ds compute descriptors
        CDescriptor matReferenceDescriptors;
        m_cExtractorSURF.compute( p_matImageLeft, vecKeyPoints, matReferenceDescriptors );

        //std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) detected features: %lu\n", vecFeaturePoints.size( ) );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) computed descriptors: %lu\n", vecKeyPoints.size( ) );

        //ds current reference points
        std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > > vecReferencePoints;

        //ds register all features
        for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
        {
            //ds get the keypoint
            const cv::KeyPoint& cKeyPoint( vecKeyPoints[u] );

            //ds check size
            if( m_uKeyPointSizeLimit > cKeyPoint.size )
            {
                const Eigen::Vector3d vecPointNormalized( m_cCameraLEFT.getNormalHomogenized( cKeyPoint ) );

                //ds set current reference point
                vecReferencePoints.push_back( std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized >( cKeyPoint, matReferenceDescriptors.row( u ), vecPointNormalized, vecPointNormalized ) );

                //ds draw detected point
                cv::circle( matDisplayUpper, cKeyPoint.pt, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                cv::circle( matDisplayUpper, cKeyPoint.pt, cKeyPoint.size, CColorCodeBGR( 255, 0, 0 ), 1 );
            }
        }

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) useful keypoints: %lu\n", vecReferencePoints.size( ) );

        //ds add a new scan vector
        m_vecScanPoints.push_back( std::pair< std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > >, Eigen::Isometry3d >( vecReferencePoints, p_matCurrentTransformation ) );

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
        cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matCurrentTransformation.translation( )( 0 )*10, 175-p_matCurrentTransformation.translation( )( 1 )*10 ), 5, CColorCodeBGR( 0, 255, 0 ), 1 );
        cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 5, CColorCodeBGR( 0, 255, 0 ), 1 );
    }
    else
    {
        _updateFrameRateDisplay( 10 );
    }
}

void CEpipolarDetector::_drawProjectedEpipolarLineEssentialBRIEF( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_iLineLength )
{
    //ds check all scan points we have
    for( std::vector< std::pair< std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > >, Eigen::Isometry3d > >::iterator pScanPoint = m_vecScanPoints.begin( ); pScanPoint != m_vecScanPoints.end( ); ++pScanPoint )
    {
        //ds compute essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( pScanPoint->second, p_matCurrentTransformation ) );

        //ds loop over the points for the current scan
        for( std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > >::iterator pReferencePoint = pScanPoint->first.begin( ); pReferencePoint != pScanPoint->first.end( ); ++pReferencePoint )
        {
            //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
            const Eigen::Vector3d vecCoefficients( matEssential*std::get< 2 >( *pReferencePoint ) );

            //ds compute maximum and minimum points (from top to bottom line)
            const Eigen::Vector3d& vecReferenceLastDetection( std::get< 3 >( *pReferencePoint ) );
            const int32_t iULastDetection( m_cCameraLEFT.getDenormalizedX( vecReferenceLastDetection(0) ) );
            const int32_t iVLastDetection( m_cCameraLEFT.getDenormalizedY( vecReferenceLastDetection(1) ) );
            const int32_t iVLimitMaximum( std::min( iVLastDetection+p_iLineLength, static_cast< int32_t >(  m_cCameraLEFT.m_uHeightPixel ) ) );
            const int32_t iVLimitMinimum( std::max( iVLastDetection-p_iLineLength, 0 ) );

            //ds get back to pixel coordinates
            int32_t iUMaximum( std::min( iULastDetection+p_iLineLength, static_cast< int32_t >( m_cCameraLEFT.m_uWidthPixel ) ) );
            int32_t iUMinimum( std::max( iULastDetection-p_iLineLength, 0 ) );
            int32_t iVMaximum( m_cCameraLEFT.getDenormalizedY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( iUMaximum ) )/vecCoefficients(1) ) );
            int32_t iVMinimum( m_cCameraLEFT.getDenormalizedY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( iUMinimum ) )/vecCoefficients(1) ) );

            //ds negative slope (max v is also at max u)
            if( iVMaximum > iVMinimum )
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMinimum > iVMinimum )
                {
                    iVMinimum = iVLimitMinimum;
                    iUMinimum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMinimum ) )/vecCoefficients(0) );
                }
                if( iVLimitMaximum < iVMaximum )
                {
                    iVMaximum = iVLimitMaximum;
                    iUMaximum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMaximum ) )/vecCoefficients(0) );
                }
            }

            //ds positive slope (max v is at min u)
            else
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMaximum < iVMinimum )
                {
                    iVMinimum = iVLimitMaximum;
                    iUMinimum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMaximum ) )/vecCoefficients(0) );
                }
                if( iVLimitMinimum > iVMaximum )
                {
                    iVMaximum = iVLimitMinimum;
                    iUMaximum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMinimum ) )/vecCoefficients(0) );
                }

                //ds swap required for ROI generation
                std::swap( iVMinimum, iVMaximum );
            }

            //ds compute pixel ranges to sample
            const int32_t iDeltaX( iUMaximum-iUMinimum );
            const int32_t iDeltaY( iVMaximum-iVMinimum );

            //ds draw the ROI
            cv::rectangle( p_matDisplay, cv::Point2i( iUMinimum, iVMinimum ), cv::Point2i( iUMaximum, iVMaximum ), CColorCodeBGR( 255, 0, 0 ) );

            //ds sample line length
            const uint32_t uSamples( std::sqrt( iDeltaX*iDeltaX + iDeltaY*iDeltaY ) );

            //ds keypoint vectors
            std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
            vecReferenceKeyPoints[0] = std::get< 0 >( *pReferencePoint );
            std::vector< cv::KeyPoint > vecPoolKeyPoints( uSamples );
            const double& dKeyPointSize( vecReferenceKeyPoints[0].size );

            //ds compute step size for pixel sampling
            const double dStepSizeU( static_cast< double >( iDeltaX )/uSamples );

            //ds set the keypoints (by sampling over x)
            for( uint32_t u = 0; u < uSamples; ++u )
            {
                //ds compute current x and y
                const double dU( iUMinimum+u*dStepSizeU );
                const double dY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( dU ) )/vecCoefficients(1) );
                const double dV( m_cCameraLEFT.getDenormalizedY( dY ) );

                //ds add keypoint
                vecPoolKeyPoints[u] = cv::KeyPoint( dU, dV, dKeyPointSize );

                cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 255, 0, 0 ), -1 );
            }

            //std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) keypoints pool size: %lu\n", vecPoolKeyPoints.size( ) );

            //ds descriptor pool to match
            cv::Mat& matReferenceDescriptor( std::get< 1 >( *pReferencePoint ) );
            cv::Mat matPoolDescriptors;

            //ds compute descriptors of current search area
            m_cExtractorBRIEF.compute( p_matImage, vecPoolKeyPoints, matPoolDescriptors );

            //std::printf( "descriptors: rows %i cols %i %i\n", matPoolDescriptors.rows, matPoolDescriptors.cols, matPoolDescriptors.dims );

            //ds check if we could compute a descriptor
            if( 0 != matPoolDescriptors.rows )
            {
                //ds match the descriptors
                //std::vector< std::vector< cv::DMatch > > vecMatches;
                //m_cDescriptorMatcher.knnMatch( matReferenceDescriptor, matPoolDescriptors, vecMatches, 10 );
                std::vector< cv::DMatch > vecMatches;
                m_cFLANNMatcher.match( matReferenceDescriptor, matPoolDescriptors, vecMatches );

                //ds buffer first match
                //const cv::DMatch cBestMatch( vecMatches[0][0] );
                const cv::DMatch& cBestMatch( vecMatches[0] );

                //ds get the matching keypoint
                cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

                if( m_dMatchingDistanceCutoff < cBestMatch.distance )
                {
                    //ds drop the match
                    std::printf( "<CEpilinearStereoDetector>(_drawProjectedEpipolarLineEssential) matching distance: %f - dropped match\n", cBestMatch.distance );
                    pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
                }
                else
                {
                    //ds update reference
                    std::get< 3 >( *pReferencePoint ) = CPoint2DNormalized( m_cCameraLEFT.getNormalHomogenized( ptMatch ) );

                    //ds draw the match
                    cv::circle( p_matDisplay, ptMatch, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                }
            }
            else
            {
                std::printf( "<CEpilinearStereoDetector>(_drawProjectedEpipolarLineEssential) landmark no longer present\n" );
                pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
            }
        }

        //ds check if we have to remove the vector
        if( pScanPoint->first.empty( ) )
        {
            pScanPoint = m_vecScanPoints.erase( pScanPoint )-1;
            std::printf( "<CEpilinearStereoDetector>(_drawProjectedEpipolarLineEssential) erased scan point\n" );
        }
    }
}

void CEpipolarDetector::_drawProjectedEpipolarLineEssentialSURF( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_iLineLength )
{
    std::cout << "<CEpipolarDetector>(_drawProjectedEpipolarLineEssential) line length: " << p_iLineLength << std::endl;
    std::cout << "<CEpipolarDetector>(_drawProjectedEpipolarLineEssential) scan points: " << m_vecScanPoints.size( ) << std::endl;

    //ds check all scan points we have
    for( std::vector< std::pair< std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > >, Eigen::Isometry3d > >::iterator pScanPoint = m_vecScanPoints.begin( ); pScanPoint != m_vecScanPoints.end( ); ++pScanPoint )
    {
        //ds compute essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( pScanPoint->second, p_matCurrentTransformation ) );

        //ds matches
        uint32_t uMatchCount( 0 );

        //ds loop over the points for the current scan
        for( std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > >::iterator pReferencePoint = pScanPoint->first.begin( ); pReferencePoint != pScanPoint->first.end( ); ++pReferencePoint )
        {
            //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
            const Eigen::Vector3d vecCoefficients( matEssential*std::get< 2 >( *pReferencePoint ) );

            //ds compute maximum and minimum points (from top to bottom line)
            const Eigen::Vector3d& vecReferenceLastDetection( std::get< 3 >( *pReferencePoint ) );
            const int32_t iULastDetection( m_cCameraLEFT.getDenormalizedX( vecReferenceLastDetection(0) ) );
            const int32_t iVLastDetection( m_cCameraLEFT.getDenormalizedY( vecReferenceLastDetection(1) ) );
            const int32_t iVLimitMaximum( std::min( iVLastDetection+p_iLineLength, static_cast< int32_t >(  m_cCameraLEFT.m_uHeightPixel ) ) );
            const int32_t iVLimitMinimum( std::max( iVLastDetection-p_iLineLength, 0 ) );

            //ds get back to pixel coordinates
            int32_t iUMaximum( std::min( iULastDetection+p_iLineLength, static_cast< int32_t >( m_cCameraLEFT.m_uWidthPixel ) ) );
            int32_t iUMinimum( std::max( iULastDetection-p_iLineLength, 0 ) );
            int32_t iVMaximum( m_cCameraLEFT.getDenormalizedY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( iUMaximum ) )/vecCoefficients(1) ) );
            int32_t iVMinimum( m_cCameraLEFT.getDenormalizedY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( iUMinimum ) )/vecCoefficients(1) ) );

            //ds negative slope (max v is also at max u)
            if( iVMaximum > iVMinimum )
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMinimum > iVMinimum )
                {
                    iVMinimum = iVLimitMinimum;
                    iUMinimum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMinimum ) )/vecCoefficients(0) );
                }
                if( iVLimitMaximum < iVMaximum )
                {
                    iVMaximum = iVLimitMaximum;
                    iUMaximum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMaximum ) )/vecCoefficients(0) );
                }
            }

            //ds positive slope (max v is at min u)
            else
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMaximum < iVMinimum )
                {
                    iVMinimum = iVLimitMaximum;
                    iUMinimum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMaximum ) )/vecCoefficients(0) );
                }
                if( iVLimitMinimum > iVMaximum )
                {
                    iVMaximum = iVLimitMinimum;
                    iUMaximum = m_cCameraLEFT.getDenormalizedX( -( vecCoefficients(2)+vecCoefficients(1)*m_cCameraLEFT.getNormalizedY( iVLimitMinimum ) )/vecCoefficients(0) );
                }

                //ds swap required for ROI generation
                std::swap( iVMinimum, iVMaximum );
            }

            //ds compute pixel ranges to sample
            const int32_t iDeltaX( iUMaximum-iUMinimum );
            const int32_t iDeltaY( iVMaximum-iVMinimum );

            //ds compute ROI
            const int32_t iULU( std::max( iUMinimum-static_cast< int32_t >( m_uKeyPointSizeLimit ), 0 ) );
            const int32_t iULV( std::max( iVMinimum-static_cast< int32_t >( m_uKeyPointSizeLimit ), 0 ) );
            const int32_t iWidth( std::min( iDeltaX+2*m_uKeyPointSizeLimit, m_uImageCols-iULU ) );
            const int32_t iHeight( std::min( iDeltaY+2*m_uKeyPointSizeLimit, m_uImageRows-iULV ) );
            const cv::Rect cROI( iULU, iULV, iWidth, iHeight );

            //ds draw the ROI
            cv::rectangle( p_matDisplay, cv::Point2i( iULU, iULV ), cv::Point2i( iULU+iWidth, iULV+iHeight ), CColorCodeBGR( 255, 0, 0 ) );

            //ds sample line length
            const uint32_t uSamples( std::sqrt( iDeltaX*iDeltaX + iDeltaY*iDeltaY ) );

            //ds keypoint vectors
            std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
            vecReferenceKeyPoints[0] = std::get< 0 >( *pReferencePoint );
            std::vector< cv::KeyPoint > vecPoolKeyPoints( uSamples );
            const float& fKeyPointSize( vecReferenceKeyPoints[0].size );

            //ds compute step size for pixel sampling
            const double dStepSizeU( static_cast< double >( iDeltaX )/uSamples );

            //ds set the keypoints (by sampling over x)
            for( uint32_t u = 0; u < uSamples; ++u )
            {
                //ds compute current x and y
                const double dU( iUMinimum+u*dStepSizeU );
                const double dY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( dU ) )/vecCoefficients(1) );
                const double dV( m_cCameraLEFT.getDenormalizedY( dY ) );

                //ds add keypoint
                vecPoolKeyPoints[u] = cv::KeyPoint( dU-iUMinimum, dV-iVMinimum, fKeyPointSize );
                //vecPoolKeyPoints[u] = cv::KeyPoint( dU, dV, fKeyPointSize );

                cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 255, 0, 0 ), -1 );
            }

            //std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) keypoints pool size: %lu\n", vecPoolKeyPoints.size( ) );

            //ds descriptor pool to match
            cv::Mat& matReferenceDescriptor( std::get< 1 >( *pReferencePoint ) );
            cv::Mat matPoolDescriptors;

            //ds compute descriptors of current search area
            m_cExtractorSURF.compute( p_matImage( cROI ), vecPoolKeyPoints, matPoolDescriptors );

            //std::printf( "descriptors: rows %i cols %i %i\n", matPoolDescriptors.rows, matPoolDescriptors.cols, matPoolDescriptors.dims );

            //ds check if we could compute a descriptor
            if( 0 != matPoolDescriptors.rows )
            {
                //ds match the descriptors
                //std::vector< std::vector< cv::DMatch > > vecMatches;
                //m_cDescriptorMatcher.knnMatch( matReferenceDescriptor, matPoolDescriptors, vecMatches, 10 );
                std::vector< cv::DMatch > vecMatches;
                m_cFLANNMatcher.match( matReferenceDescriptor, matPoolDescriptors, vecMatches );

                //ds buffer first match
                //const cv::DMatch cBestMatch( vecMatches[0][0] );
                const cv::DMatch& cBestMatch( vecMatches[0] );

                //ds get the matching keypoint
                cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

                if( m_dMatchingDistanceCutoff < cBestMatch.distance )
                {
                    //ds drop the match
                    //std::printf( "<CEpilinearStereoDetector>(_drawProjectedEpipolarLineEssential) matching distance: %f - dropped match\n", cBestMatch.distance );
                    //pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
                }
                else
                {
                    //ds shift back
                    ptMatch.x += iUMinimum;
                    ptMatch.y += iVMinimum;

                    //ds update reference
                    std::get< 3 >( *pReferencePoint ) = CPoint2DNormalized( m_cCameraLEFT.getNormalHomogenized( ptMatch ) );

                    //ds draw the match
                    cv::circle( p_matDisplay, ptMatch, 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                    ++uMatchCount;
                }
            }
            else
            {
                std::printf( "<CEpilinearStereoDetector>(_drawProjectedEpipolarLineEssential) landmark no longer present\n" );
                pReferencePoint = pScanPoint->first.erase( pReferencePoint )-1;
            }
        }

        std::cout << "matches: " << uMatchCount << std::endl;

        //ds check if we have to remove the vector
        if( pScanPoint->first.empty( ) )
        {
            pScanPoint = m_vecScanPoints.erase( pScanPoint )-1;
            std::printf( "<CEpilinearStereoDetector>(_drawProjectedEpipolarLineEssential) erased scan point\n" );
        }
    }
}

void CEpipolarDetector::_triangulatePointSURF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, cv::Mat& p_matDisplayUpper, cv::Mat& p_matDisplayUpperTemporary, const Eigen::Isometry3d p_matCurrentTransformation )
{
    /*ds get a random color code
    const CColorCode vecColorCode( m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ) );

    //ds get the reference point locally
    const cv::Point2d ptReference( CEpipolarDetector::m_ptMouseClick.x, CEpipolarDetector::m_ptMouseClick.y );

    //ds draw current user point and descriptor radius
    cv::circle( p_matDisplayUpper, ptReference, 2, CColorCode( 0, 0, 255 ), -1 );
    cv::circle( p_matDisplayUpper, ptReference, m_uDescriptorRadius, CColorCode( 0, 0, 255 ), 2 );
    cv::circle( p_matDisplayUpperTemporary, ptReference, 2, CColorCode( 0, 0, 255 ), -1 );
    cv::circle( p_matDisplayUpperTemporary, ptReference, m_uDescriptorRadius, CColorCode( 0, 0, 255 ), 2 );

    //ds draw epipolar line
    cv::line( p_matDisplayUpper, ptReference, cv::Point( m_uImageCols+ptReference.x, ptReference.y ), vecColorCode );
    cv::line( p_matDisplayUpperTemporary, ptReference, cv::Point( m_uImageCols+ptReference.x, ptReference.y ), vecColorCode );

    //ds keypoint vector for matching
    std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
    vecReferenceKeyPoints[0] = cv::KeyPoint( ptReference, m_uDescriptorRadius );

    //ds set keypoint range for the right image
    int32_t iLeftLimit( ptReference.x - m_uImageCols/2 );

    //ds check boundaries
    if( 0 > iLeftLimit ){ iLeftLimit = 0; }

    //ds number of keypoints to check
    const uint32_t uKeyPointsPoolSize( ptReference.x-iLeftLimit );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uKeyPointsPoolSize );

    //ds set the keypoints
    for( uint32_t uX = 0; uX < uKeyPointsPoolSize; ++uX )
    {
        vecPoolKeyPoints[uX] = cv::KeyPoint( uX+iLeftLimit, ptReference.y, m_uDescriptorRadius );
    }

    std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) keypoints pool size: %lu\n", vecPoolKeyPoints.size( ) );

    CDescriptorSURF matReferenceDescriptor;
    cv::Mat matPoolDescriptors;

    //ds compute descriptors
    m_cExtractorSURF.compute( p_matImageLeft, vecReferenceKeyPoints, matReferenceDescriptor );
    m_cExtractorSURF.compute( p_matImageRight, vecPoolKeyPoints, matPoolDescriptors );

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_cDescriptorMatcher.match( matReferenceDescriptor, matPoolDescriptors, vecMatches );

    //ds get the matching keypoint
    cv::Point2f ptMatch( vecPoolKeyPoints[ vecMatches[0].trainIdx ].pt );

    std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) descriptor matching distance: %f\n", vecMatches[0].distance );
    std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) stereo offset x: %f y: %f\n", ptMatch.x-vecReferenceKeyPoints[0].pt.x, ptMatch.y-vecReferenceKeyPoints[0].pt.y );

    //ds draw the matching keypoint
    cv::circle( p_matDisplayUpper, cv::Point( m_uImageCols+ptMatch.x, ptMatch.y ), 2, CColorCode( 0, 255, 0 ), -1 );
    cv::circle( p_matDisplayUpperTemporary, cv::Point( m_uImageCols+ptMatch.x, ptMatch.y ), 2, CColorCode( 0, 255, 0 ), -1 );

    //ds triangulate 3d point
    const Eigen::Vector3d vec3DReferencePointSVDLS( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( ptReference, ptMatch, CConfigurationStereoCamera::LEFT::matProjection, CConfigurationStereoCamera::RIGHT::matProjection ) );
    const Eigen::Vector3d vec3DReferencePointQRLS( CMiniVisionToolbox::getPointStereoLinearTriangulationQRLS( ptReference, ptMatch, CConfigurationStereoCamera::LEFT::matProjection, CConfigurationStereoCamera::RIGHT::matProjection ) );
    std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) computed depth (SVD-LS): %f\n", vec3DReferencePointSVDLS(2) );
    std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) computed depth (QR-LS): %f\n", vec3DReferencePointQRLS(2) );

    //ds compute deviation
    const double dAbsoluteDeviation = std::fabs( static_cast< double >( 1.0-vec3DReferencePointSVDLS(2)/vec3DReferencePointQRLS(2) ) )*100.0;
    std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) absolute deviation: %f %%\n", dAbsoluteDeviation );

    //ds add the coordinates to the reference structure
    m_vecReferencePoints.push_back( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptorSURF, Eigen::Isometry3d, CColorCode, CPoint2DInCameraFrame >( CPoint2DInCameraFrame( ptReference.x, ptReference.y ),
                                                                                                                                                             vec3DReferencePointSVDLS,
                                                                                                                                                             vecReferenceKeyPoints[0],
                                                                                                                                                             matReferenceDescriptor,
                                                                                                                                                             p_matCurrentTransformation,
                                                                                                                                                             vecColorCode,
                                                                                                                                                             CPoint2DInCameraFrame( ptReference.x, ptReference.y ) ) );*/
}

void CEpipolarDetector::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<CEpilinearStereoDetector>(_shutDown) termination requested, detector disabled\n" );
}

void CEpipolarDetector::_speedUp( )
{
    ++m_iPlaybackSpeedupCounter;
    m_dFrequencyPlaybackHz += std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;
    std::printf( "<CEpilinearStereoDetector>(_speedUp) increased playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
}

void CEpipolarDetector::_slowDown( )
{
    m_dFrequencyPlaybackHz -= std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;

    //ds 12 fps minimum (one of 2 images 10 imu messages and 1 pose: 13-1)
    if( 12 < m_dFrequencyPlaybackHz )
    {
        std::printf( "<CEpilinearStereoDetector>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
        --m_iPlaybackSpeedupCounter;
    }
    else
    {
        m_dFrequencyPlaybackHz = 12.5;
        std::printf( "<CEpilinearStereoDetector>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
    }
}

void CEpipolarDetector::_updateFrameRateDisplay( const uint32_t& p_uFrameProbeRange )
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
