#include "CStereoDetector.h"

#include <opencv/highgui.h>
#include <Eigen/Core>

#include "configuration/CConfigurationStereoCamera.h"
#include "configuration/CConfigurationCamera.h"

//ds UGLY statics
cv::Point2i CStereoDetector::m_ptMouseClick = cv::Point2i( 0, 0 );
bool CStereoDetector::m_bRightClicked = false;

CStereoDetector::CStereoDetector( const uint32_t& p_uImageRows,
                                  const uint32_t& p_uImageCols,
                                  const bool& p_bDisplayImages,
                                  const uint32_t p_uFrequencyPlaybackHz ): m_uImageRows( p_uImageRows ),
                                                                           m_uImageCols( p_uImageCols ),
                                                                           m_prSize( cv::Size( p_uImageCols,  p_uImageRows ) ),
                                                                           m_matIntrinsicLEFT( CWrapperOpenCV::fromCVMatrix< double, 3, 3 >( CConfigurationStereoCamera::LEFT::matIntrinsic ) ),
                                                                           m_matIntrinsicRIGHT( CWrapperOpenCV::fromCVMatrix< double, 3, 3 >( CConfigurationStereoCamera::RIGHT::matIntrinsic ) ),
                                                                           m_matProjectionLEFT( CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( CConfigurationStereoCamera::LEFT::matProjection ) ),
                                                                           m_matProjectionRIGHT( CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( CConfigurationStereoCamera::RIGHT::matProjection ) ),
                                                                           m_matMLEFT( m_matProjectionLEFT.block< 3, 3 >( 0, 0 ) ),
                                                                           m_vecTLEFT( m_matProjectionLEFT.block< 3, 1 >( 0, 3 ) ),
                                                                           m_uFrameCount( 0 ),
                                                                           m_cDetectorSURF( cv::SurfFeatureDetector( 400 ) ),
                                                                           m_cExtractorSURF( cv::SurfDescriptorExtractor( 400 ) ),
                                                                           m_dMatchingDistanceCutoff( 0.5 ),
                                                                           m_uFeaturesCap( 100 ),
                                                                           m_uDescriptorRadius( 40 ),
                                                                           m_uDescriptorCenterPixelOffset( (m_uDescriptorRadius-1)/2 ),
                                                                           m_rectROI( cv::Rect( m_uDescriptorCenterPixelOffset, m_uDescriptorCenterPixelOffset, m_uImageCols-2*m_uDescriptorCenterPixelOffset, m_uImageRows-2*m_uDescriptorCenterPixelOffset ) ),
                                                                           m_bDisplayImages( p_bDisplayImages ),
                                                                           m_bIsShutdownRequested( false ),
                                                                           m_dFrequencyPlaybackHz( p_uFrequencyPlaybackHz ),
                                                                           m_uFrequencyPlaybackDeltaHz( 50 ),
                                                                           m_iPlaybackSpeedupCounter( 0 ),
                                                                           m_cRandomGenerator( 1337 ),
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

    //ds initialize reference point holder
    m_vecReferencePoints.clear( );

    //ds initialize the window
    cv::namedWindow( "stereo matching", cv::WINDOW_AUTOSIZE );

    //ds set mouse callback
    cv::setMouseCallback( "stereo matching", CStereoDetector::_catchMouseClick, 0 );

    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) instance allocated\n" );
}

CStereoDetector::~CStereoDetector( )
{

}

void CStereoDetector::receivevDataVIWithPose( std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageLeft, std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageRight, const txt_io::CIMUMessage& p_cIMU, const std::shared_ptr< txt_io::CPoseMessage >& p_cPose )
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
        _localizeManual( matImageLeft, matImageRight, m_matTransformation );
        //_localizeAuto( matImageLeft, matImageRight, m_matTransformation );
    }
}

void CStereoDetector::_localizeManual( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation )
{
    //ds get delta to evaluate precision
    const double dTransformationDelta( CMiniVisionToolbox::getTransformationDelta( m_matPreviousTransformationLeft, p_matCurrentTransformation ) );

    //ds log info
    std::printf( "[%lu] transformation delta: %f\n", m_uFrameCount, dTransformationDelta );

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
    _drawProjectedEpipolarLineEssential( p_matCurrentTransformation, matDisplayLeft, matLeft, dTransformationDelta*3 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpperTemporary );



    //ds show the image
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
    cv::imshow( "stereo matching", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", m_matTrajectoryXY );
    cv::imshow( "some stuff", m_matTrajectoryZ );
    int iLastKeyStroke( cv::waitKey( 1 ) );

    //ds if there was a keystroke
    if( -1 != iLastKeyStroke )
    {
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

        //ds check current displacement
        const double dRelativeDistanceMeters( p_matCurrentTransformation.translation( ).squaredNorm( ) );

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) moved distance: %f requesting user input at frame: %lu\n", dRelativeDistanceMeters, m_uFrameCount );

        //ds enable mouse click
        CStereoDetector::m_ptMouseClick.x = -1;
        CStereoDetector::m_ptMouseClick.y = -1;
        CStereoDetector::m_bRightClicked = false;
        cv::rectangle( matDisplayUpperTemporary, m_rectROI, CColorCode( 255, 255, 255 ) );
        cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );

        //ds modification detection
        bool bWasThereAModification( false );

        //ds record mouseclicks until space is pressed
        while( CConfigurationOpenCV::KeyStroke::iSpace != iLastKeyStroke )
        {
            if( CStereoDetector::m_ptMouseClick.inside( m_rectROI ) )
            {
                //ds caught mouse click
                std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) ---------------------------------------------------------------\n" );
                std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) selected point: ( %u, %u )\n", CStereoDetector::m_ptMouseClick.x, CStereoDetector::m_ptMouseClick.y );

                //ds handle the marked point
                _triangulatePointSURF( matLeft, matRight, matDisplayUpper, matDisplayUpperTemporary, p_matCurrentTransformation );
                //_triangulatePointDepthSampling( matLeft, matRight, matDisplayUpper, matDisplayUpperTemporary, p_matCurrentTransformation );

                //ds update image
                cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
                cv::imshow( "stereo matching", matDisplayComplete );

                //ds reset mouse
                CStereoDetector::m_ptMouseClick.x = -1;
                CStereoDetector::m_ptMouseClick.y = -1;

                //ds modified
                bWasThereAModification = true;
            }
            else if( -1 != CStereoDetector::m_ptMouseClick.x && -1 != CStereoDetector::m_ptMouseClick.y )
            {
                //ds clicked somewhere else, just reset
                CStereoDetector::m_ptMouseClick.x = -1;
                CStereoDetector::m_ptMouseClick.y = -1;
            }

            //ds if there was a right click
            if( CStereoDetector::m_bRightClicked )
            {
                //ds we have to clean up everything
                m_vecReferencePoints.clear( );

                std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) cleared reference points\n" );

                //ds reset and remember
                CStereoDetector::m_bRightClicked = false;

                //ds update image
                cv::hconcat( matDisplayLeftClean, matDisplayRightClean, matDisplayUpper );
                cv::hconcat( matDisplayLeftClean, matDisplayRightClean, matDisplayUpperTemporary );

                //ds redraw selectable rectangle
                cv::rectangle( matDisplayUpper, m_rectROI, CColorCode( 255, 255, 255 ) );
                cv::rectangle( matDisplayUpperTemporary, m_rectROI, CColorCode( 255, 255, 255 ) );
                cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
                cv::imshow( "stereo matching", matDisplayComplete );

                //ds modified
                bWasThereAModification = true;
            }

            //ds check keys
            iLastKeyStroke = cv::waitKey( 1 );

            //ds evaluate again
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
                default:
                {
                    break;
                }
            }
        }

        //ds check if new points have to be analyzed
        if( bWasThereAModification )
        {
            //ds get a copy of the upper to the lower display
            m_matDisplayLowerReference = matDisplayUpper.clone( );
            cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );

            //ds mark position of user input (persistently)
            cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matCurrentTransformation.translation( )( 0 )*10, 175-p_matCurrentTransformation.translation( )( 1 )*10 ), 5, CColorCode( 0, 255, 0 ), 1 );
            cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 5, CColorCode( 0, 255, 0 ), 1 );
        }
        else
        {
            std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) no changes made, keeping reference points\n" );
        }
    }

    //ds update references
    m_matReferenceFrameLeft  = p_matImageLeft;
    m_matReferenceFrameRight = p_matImageRight;
    m_matPreviousTransformationLeft  = p_matCurrentTransformation;

    //ds increment count
    ++m_uFrameCount;
}

void CStereoDetector::_localizeAuto( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation )
{
    //ds get delta to evaluate precision
    const double dTransformationDelta( CMiniVisionToolbox::getTransformationDelta( m_matPreviousTransformationLeft, p_matCurrentTransformation ) );

    //ds log info
    std::printf( "[%lu] transformation delta: %f\n", m_uFrameCount, dTransformationDelta );

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
    _drawProjectedEpipolarLineEssential( p_matCurrentTransformation, matDisplayLeft, matLeft, dTransformationDelta*3 );
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpperTemporary );



    //ds show the image
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
    cv::imshow( "stereo matching", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", m_matTrajectoryXY );
    cv::imshow( "some stuff", m_matTrajectoryZ );
    int iLastKeyStroke( cv::waitKey( 1 ) );

    //ds if there was a keystroke
    if( -1 != iLastKeyStroke )
    {
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

        //ds we have to clean up everything
        m_vecReferencePoints.clear( );

        //ds update image
        cv::hconcat( matDisplayLeftClean, matDisplayRightClean, matDisplayUpper );
        cv::hconcat( matDisplayLeftClean, matDisplayRightClean, matDisplayUpperTemporary );

        //ds redraw
        cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );

        //ds check current displacement
        const double dRelativeDistanceMeters( p_matCurrentTransformation.translation( ).squaredNorm( ) );

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) moved distance: %f requesting feature detection at frame: %lu\n", dRelativeDistanceMeters, m_uFrameCount );

        //ds detect features
        std::vector< cv::KeyPoint> vecKeyPoints;
        m_cDetectorSURF.detect( p_matImageLeft, vecKeyPoints );

        //ds compute descriptors
        CDescriptor matReferenceDescriptors;
        m_cExtractorSURF.compute( p_matImageLeft, vecKeyPoints, matReferenceDescriptors );

        std::printf( "dim: %i %i %i", matReferenceDescriptors.rows, matReferenceDescriptors.cols, matReferenceDescriptors.dims );

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) detected features: %lu\n", vecKeyPoints.size( ) );

        //ds register all features
        for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
        {
            //ds get a random color code
            const CColorCode vecColorCode( m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ) );

            //ds get the keypoint
            const cv::KeyPoint cKeyPoint( vecKeyPoints[u] );

            //ds add the point to structure
            m_vecReferencePoints.push_back( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptor, Eigen::Isometry3d, CColorCode, CPoint2DInCameraFrame >( CWrapperOpenCV::fromCVVector( cKeyPoint.pt ),
                                                                                                                                                                     Eigen::Vector3d( 0, 0, 0 ),
                                                                                                                                                                     cKeyPoint,
                                                                                                                                                                     matReferenceDescriptors.row( u ),
                                                                                                                                                                     p_matCurrentTransformation,
                                                                                                                                                                     vecColorCode,
                                                                                                                                                                     CWrapperOpenCV::fromCVVector( cKeyPoint.pt ) ) );

            //ds draw detected point
            cv::circle( matDisplayUpper, cKeyPoint.pt, 2, vecColorCode, -1 );
        }

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

        //ds mark position of user input (persistently)
        cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matCurrentTransformation.translation( )( 0 )*10, 175-p_matCurrentTransformation.translation( )( 1 )*10 ), 5, CColorCode( 0, 255, 0 ), 1 );
        cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 5, CColorCode( 0, 255, 0 ), 1 );
    }

    //ds update references
    m_matReferenceFrameLeft  = p_matImageLeft;
    m_matReferenceFrameRight = p_matImageRight;
    m_matPreviousTransformationLeft  = p_matCurrentTransformation;

    //ds increment count
    ++m_uFrameCount;
}

void CStereoDetector::_drawProjectedEpipolarLineEssential( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const double& p_dLineLength )
{
    //ds for all the registered points
    for( std::vector< std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptor, Eigen::Isometry3d, CColorCode, CPoint2DInCameraFrame > >::iterator vecReferencePoint = m_vecReferencePoints.begin( ); vecReferencePoint != m_vecReferencePoints.end( ); ++vecReferencePoint )
    {

    //for( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptorSURF, Eigen::Isometry3d, CColorCode, CPoint2DInCameraFrame >& vecReferencePoint: m_vecReferencePoints )
    //{

        //ds get essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( std::get< 4 >( *vecReferencePoint ), p_matCurrentTransformation ) );

        //ds get normalized reference point
        const Eigen::Vector3d vecReference( m_cCameraLEFT.getNormalized( std::get< 0 >( *vecReferencePoint ) ) );

        //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
        const Eigen::Vector3d vecCoefficients( matEssential*vecReference );

        //std::printf( "<CSimpleFeatureDetector>(_drawEpisubsequentLine) curve equation: f(x) = %fx + %f\n", -vecCoefficients(0)/vecCoefficients(1), -vecCoefficients(2)/vecCoefficients(1) );

        //ds compute maximum and minimum points (from top to bottom line)
        const Eigen::Vector3d vecReferenceLastDetection( m_cCameraLEFT.getNormalized( std::get< 6 >( *vecReferencePoint ) ) );
        const double dLimitXMinimum( vecReferenceLastDetection(0)-p_dLineLength );
        const double dLimitXMaximum( vecReferenceLastDetection(0)+p_dLineLength );
        double dYMinimum( vecReferenceLastDetection(1)-p_dLineLength );//m_cCameraLEFT.m_prRangeHeightNormalized.first );
        double dYMaximum( vecReferenceLastDetection(1)+p_dLineLength );//m_cCameraLEFT.m_prRangeHeightNormalized.second );
        double dXMinimum( -( vecCoefficients(2)+vecCoefficients(1)*dYMinimum )/vecCoefficients(0) );
        double dXMaximum( -( vecCoefficients(2)+vecCoefficients(1)*dYMaximum )/vecCoefficients(0) );

        //ds shift the points to the visible range and recompute y
        if( dLimitXMinimum > dXMinimum )
        {
            dXMinimum = dLimitXMinimum;
            dYMinimum = -( vecCoefficients(2)+vecCoefficients(0)*dXMinimum )/vecCoefficients(1);
        }
        else if( dLimitXMaximum < dXMinimum )
        {
            dXMinimum = dLimitXMaximum;
            dYMinimum = -( vecCoefficients(2)+vecCoefficients(0)*dXMinimum )/vecCoefficients(1);
        }
        if( dLimitXMinimum > dXMaximum )
        {
            dXMaximum = dLimitXMinimum;
            dYMinimum = -( vecCoefficients(2)+vecCoefficients(0)*dXMaximum )/vecCoefficients(1);
        }
        else if( dLimitXMaximum < dXMaximum )
        {
            dXMaximum = dLimitXMaximum;
            dYMinimum = -( vecCoefficients(2)+vecCoefficients(0)*dXMaximum )/vecCoefficients(1);
        }

        //ds swap for consistency as we are looping from x=0 to x=max
        if( dXMinimum > dXMaximum )
        {
            std::swap( dXMinimum, dXMaximum );
            std::swap( dYMinimum, dYMaximum );
        }

        //ds draw the line
        //cv::line( p_matDisplay, cv::Point2i( dXforYMinimum, dYforYMinimum ), cv::Point2i( dXforYMaximum, dYforYMaximum ), std::get< 4 >( vecReferencePoint ) );

        //ds get back to pixel coordinates
        const double dUMaximum( m_cCameraLEFT.getDenormalizedX( dXMaximum ) );
        const double dUMinimum( m_cCameraLEFT.getDenormalizedX( dXMinimum ) );
        const double dVMaximum( m_cCameraLEFT.getDenormalizedY( dYMaximum ) );
        const double dVMinimum( m_cCameraLEFT.getDenormalizedY( dYMinimum ) );

        //ds compute pixel ranges to sample
        const double dDeltaX( dUMaximum-dUMinimum );
        const double dDeltaY( dVMaximum-dVMinimum );

        //ds sample line length
        const uint32_t uSamples( std::lround( std::sqrt( dDeltaX*dDeltaX + dDeltaY*dDeltaY ) ) );

        //ds keypoint vectors
        std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
        vecReferenceKeyPoints[0] = std::get< 2 >( *vecReferencePoint );
        std::vector< cv::KeyPoint > vecPoolKeyPoints( uSamples );

        //ds color code
        const CColorCode cColorLine( std::get< 5 >( *vecReferencePoint ) );

        //ds compute step size for pixel sampling
        const double dStepSizeU( dDeltaX/uSamples );

        //ds set the keypoints (by sampling over x)
        for( uint32_t u = 0; u < uSamples; ++u )
        {
            //ds compute current x and y
            const double dU( dUMinimum+u*dStepSizeU );
            const double dY( -( vecCoefficients(2)+vecCoefficients(0)*m_cCameraLEFT.getNormalizedX( dU ) )/vecCoefficients(1) );
            const double dV( m_cCameraLEFT.getDenormalizedY( dY ) );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dU, dV, m_uDescriptorRadius );

            cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, cColorLine, -1 );
        }

        std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) keypoints pool size: %lu\n", vecPoolKeyPoints.size( ) );

        //ds descriptor pool to match
        cv::Mat matReferenceDescriptor( std::get< 3 >( *vecReferencePoint ) );
        cv::Mat matPoolDescriptors;

        //ds compute descriptors of current image
        m_cDetectorSURF.compute( p_matImage, vecPoolKeyPoints, matPoolDescriptors );

        //ds check if we could compute a descriptor
        if( 0 != matPoolDescriptors.rows )
        {
            //ds match the descriptors
            //std::vector< std::vector< cv::DMatch > > vecMatches;
            //m_cDescriptorMatcher.knnMatch( matReferenceDescriptor, matPoolDescriptors, vecMatches, 10 );
            std::vector< cv::DMatch > vecMatches;
            m_cDescriptorMatcher.match( matReferenceDescriptor, matPoolDescriptors, vecMatches );

            //ds buffer first match
            //const cv::DMatch cBestMatch( vecMatches[0][0] );
            const cv::DMatch cBestMatch( vecMatches[0] );

            //ds get the matching keypoint
            cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

            if( m_dMatchingDistanceCutoff < cBestMatch.distance )
            {
                //ds drop the match
                std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) matching distance: %f - dropped match\n", cBestMatch.distance );
                //m_vecReferencePoints.erase( vecReferencePoint );
            }
            else
            {
                //ds get the matching keypoint
                cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

                //ds update reference
                std::get< 6 >( *vecReferencePoint ) = CPoint2DInCameraFrame( ptMatch.x, ptMatch.y );
                //std::get< 4 >( vecReferencePoint ) = p_matCurrentTransformation;

                //ds draw the match
                cv::circle( p_matDisplay, ptMatch, 2, CColorCode( 0, 255, 0 ), -1 );
                cv::circle( p_matDisplay, ptMatch, m_uDescriptorRadius, CColorCode( 0, 255, 0 ), 1 );
            }
        }
        else
        {
            std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) landmark no longer present\n" );
            m_vecReferencePoints.erase( vecReferencePoint );
        }
    }
}

void CStereoDetector::_triangulatePointSURF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, cv::Mat& p_matDisplayUpper, cv::Mat& p_matDisplayUpperTemporary, const Eigen::Isometry3d p_matCurrentTransformation )
{
    //ds get a random color code
    const CColorCode vecColorCode( m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ) );

    //ds get the reference point locally
    const cv::Point2d ptReference( CStereoDetector::m_ptMouseClick.x, CStereoDetector::m_ptMouseClick.y );

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

    CDescriptor matReferenceDescriptor;
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
    m_vecReferencePoints.push_back( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptor, Eigen::Isometry3d, CColorCode, CPoint2DInCameraFrame >( CPoint2DInCameraFrame( ptReference.x, ptReference.y ),
                                                                                                                                                             vec3DReferencePointSVDLS,
                                                                                                                                                             vecReferenceKeyPoints[0],
                                                                                                                                                             matReferenceDescriptor,
                                                                                                                                                             p_matCurrentTransformation,
                                                                                                                                                             vecColorCode,
                                                                                                                                                             CPoint2DInCameraFrame( ptReference.x, ptReference.y ) ) );
}

void CStereoDetector::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<CEpilinearStereoDetector>(_shutDown) termination requested, detector disabled\n" );
}

void CStereoDetector::_speedUp( )
{
    ++m_iPlaybackSpeedupCounter;
    m_dFrequencyPlaybackHz += std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;
    std::printf( "<CEpilinearStereoDetector>(_speedUp) increased playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
}

void CStereoDetector::_slowDown( )
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

void CStereoDetector::_catchMouseClick( int p_iEventType, int p_iX, int p_iY, int p_iFlags, void* p_hUserdata )
{
    //ds if click is enabled
    if( -1 == CStereoDetector::m_ptMouseClick.x && -1 == CStereoDetector::m_ptMouseClick.y )
    {
        if( p_iEventType == cv::EVENT_LBUTTONDOWN )
        {
            //ds set coordinates (corrected)
            CStereoDetector::m_ptMouseClick.x = p_iX;
            CStereoDetector::m_ptMouseClick.y = p_iY-2;
        }
        else if( p_iEventType == cv::EVENT_RBUTTONDOWN )
        {
            //ds catch right click
            CStereoDetector::m_bRightClicked = true;
        }
    }
}
