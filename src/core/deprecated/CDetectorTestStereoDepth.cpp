#include "CDetectorTestStereoDepth.h"

#include <opencv/highgui.h>

#include "configuration/CConfigurationStereoCamera.h"
#include "configuration/CConfigurationCamera.h"
#include "exceptions/CExceptionNoMatchFound.h"
#include "utility/CMiniTimer.h"

//ds UGLY statics
cv::Point2i CDetectorTestStereoDepth::m_ptMouseClick = cv::Point2i( 0, 0 );
bool CDetectorTestStereoDepth::m_bRightClicked = false;

CDetectorTestStereoDepth::CDetectorTestStereoDepth( const uint32_t& p_uImageRows,
                                  const uint32_t& p_uImageCols,
                                  const bool& p_bDisplayImages,
                                  const uint32_t p_uFrequencyPlaybackHz ): m_uImageRows( p_uImageRows ),
                                                                           m_uImageCols( p_uImageCols ),
                                                                           m_uFrameCount( 0 ),
                                                                           m_pExtractorBRIEF( std::make_shared< cv::BriefDescriptorExtractor >( 64 ) ),
                                                                           m_cDetectorSURF( cv::SurfFeatureDetector( 400 ) ),
                                                                           m_cExtractorSURF( cv::SurfDescriptorExtractor( 400 ) ),
                                                                           m_pMatcherBRIEF( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 20, 10, 2 ) ) ),
                                                                           m_fMatchingDistanceCutoffBRIEF( 100.0 ),
                                                                           m_fMatchingDistanceCutoffSURF( 0.5 ),
                                                                           m_uFeaturesCap( 200 ),
                                                                           m_uKeyPointSizeLimit( 10 ),
                                                                           m_uDescriptorCenterPixelOffset( (m_uKeyPointSizeLimit-1)/2 ),
                                                                           m_rectROI( cv::Rect( m_uDescriptorCenterPixelOffset, m_uDescriptorCenterPixelOffset, m_uImageCols-2*m_uDescriptorCenterPixelOffset, m_uImageRows-2*m_uDescriptorCenterPixelOffset ) ),
                                                                           m_bDisplayImages( p_bDisplayImages ),
                                                                           m_bIsShutdownRequested( false ),
                                                                           m_dFrequencyPlaybackHz( p_uFrequencyPlaybackHz ),
                                                                           m_uFrequencyPlaybackDeltaHz( 50 ),
                                                                           m_iPlaybackSpeedupCounter( 0 ),
                                                                           m_cRandomGenerator( 1337 ),
                                                                           m_pCameraLEFT( std::make_shared< CPinholeCamera >( CConfigurationCamera::LEFT::cPinholeCamera ) ),
                                                                           m_pCameraRIGHT( std::make_shared< CPinholeCamera >( CConfigurationCamera::RIGHT::cPinholeCamera ) ),
                                                                           m_pStereoCamera( std::make_shared< CStereoCamera >( m_pCameraLEFT, m_pCameraRIGHT ) ),
                                                                           m_cDetectorMonoGFTT( m_pCameraLEFT, m_uFeaturesCap ),
                                                                           m_cTriangulator( m_pStereoCamera, m_pExtractorBRIEF, m_pMatcherBRIEF, 50.0 )
{
    //ds initialize reference frames with black images
    m_matReferenceFrameLeft    = cv::Mat::zeros( m_uImageRows, m_uImageCols, CV_8UC1 );
    m_matReferenceFrameRight   = cv::Mat::zeros( m_uImageRows, m_uImageCols, CV_8UC1 );
    m_matDisplayLowerReference = cv::Mat::zeros( m_uImageRows, 2*m_uImageCols, CV_8UC3 );

    //ds initialize the window
    cv::namedWindow( "stereo matching", cv::WINDOW_AUTOSIZE );

    //ds set mouse callback
    cv::setMouseCallback( "stereo matching", CDetectorTestStereoDepth::_catchMouseClick, 0 );

    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) instance allocated\n" );
}

CDetectorTestStereoDepth::~CDetectorTestStereoDepth( )
{

}

void CDetectorTestStereoDepth::receivevDataVIWithPose( const std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageLeft, const std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageRight, const txt_io::CIMUMessage& p_cIMU, const std::shared_ptr< txt_io::CPoseMessage >& p_cPose )
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
    }
}

void CDetectorTestStereoDepth::_localizeManual( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation )
{
    //ds input mats
    cv::Mat matLeft;
    cv::Mat matRight;

    //ds preprocess images
    cv::equalizeHist( p_matImageLeft, matLeft );
    cv::equalizeHist( p_matImageRight, matRight );
    m_pStereoCamera->undistortAndrectify( matLeft, matRight );

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
    cv::hconcat( matDisplayLeft, matDisplayRight, matDisplayUpper );

    //ds show the image
    cv::Mat matDisplayComplete = cv::Mat( 2*m_uImageRows, 2*m_uImageCols, CV_8UC3 );
    cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
    cv::imshow( "stereo matching", matDisplayComplete );
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

        std::printf( "<>() +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" );
        std::printf( "<>() moved distance: %f requesting user input at frame: %lu\n", dRelativeDistanceMeters, m_uFrameCount );

        //ds enable mouse click
        CDetectorTestStereoDepth::m_ptMouseClick.x = -1;
        CDetectorTestStereoDepth::m_ptMouseClick.y = -1;
        CDetectorTestStereoDepth::m_bRightClicked  = false;
        cv::rectangle( matDisplayUpper, m_rectROI, CColorCodeBGR( 255, 255, 255 ) );
        cv::circle( matDisplayUpper, cv::Point2i( m_pCameraLEFT->m_dCx, m_pCameraLEFT->m_dCy), 3, CColorCodeBGR( 0, 0, 255 ), -1 );
        cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );

        //ds selected method
        int iSelectedMethod( 0 );

        std::vector< cv::KeyPoint > vecDetectedPoints;

        //ds record mouseclicks until space is pressed
        while( CConfigurationOpenCV::KeyStroke::iSpace != iLastKeyStroke )
        {
            //ds method switch
            if( CConfigurationOpenCV::KeyStroke::iNum1 == iLastKeyStroke )
            {
                iSelectedMethod = 0;
                std::printf( "<>(_getLandmarksGFTT) selected method %i\n", iSelectedMethod );
            }
            if( CConfigurationOpenCV::KeyStroke::iNum2 == iLastKeyStroke )
            {
                iSelectedMethod = 1;
                std::printf( "<>(_getLandmarksGFTT) selected method %i\n", iSelectedMethod );
            }
            if( CConfigurationOpenCV::KeyStroke::iNum3 == iLastKeyStroke )
            {
                iSelectedMethod = 2;
                std::printf( "<>(_getLandmarksGFTT) selected method %i\n", iSelectedMethod );
            }

            if( CDetectorTestStereoDepth::m_ptMouseClick.inside( m_rectROI ) )
            {
                //ds performance measuring
                const uint64_t uToken( CMiniTimer::tic( ) );
                uint64_t uCountMatches( 0 );

                //ds caught mouse click
                std::printf( "<>(_getLandmarksGFTT) ---------------------------------------------------------------\n" );
                std::printf( "<>(_getLandmarksGFTT) 2D LEFT (selected): ( %u, %u )\n", CDetectorTestStereoDepth::m_ptMouseClick.x, CDetectorTestStereoDepth::m_ptMouseClick.y );

                //ds draw mask for detected points
                cv::Mat matMask( cv::Mat( m_uImageRows, m_uImageCols, CV_8UC1, cv::Scalar ( 255 ) ) );

                //ds draw black circles for existing landmark positions into the mask
                for( const cv::KeyPoint& cLandmarkInCameraFrame: vecDetectedPoints )
                {
                    cv::circle( matMask, cLandmarkInCameraFrame.pt, 10, cv::Scalar ( 0 ), -1 );
                }

                cv::imshow( "mask", matMask );

                //ds get keypoints
                const std::shared_ptr< std::vector< cv::KeyPoint > > vecKeyPoints( m_cDetectorMonoGFTT.detectKeyPointsTilewise( p_matImageLeft, matMask ) );

                //m_pDetectorGFTT->detect( p_matImageLeft, vecKeyPoints );
                //cv::FAST( p_matImageLeft, vecKeyPoints, 15.0 );

                //ds compute descriptors
                CDescriptor matReferenceDescriptors;
                m_pExtractorBRIEF->compute( matLeft, *vecKeyPoints, matReferenceDescriptors );

                const uint64_t uCountFeatures( vecKeyPoints->size( ) );

                for( uint32_t u = 0; u < vecKeyPoints->size( ); ++u )
                {
                    //ds current point
                    const cv::KeyPoint cKeyPoint( vecKeyPoints->at(u) );
                    cv::Point2f ptCurrent( cKeyPoint.pt );
                    const CPoint2DInCameraFrame vecLandmark2D( CWrapperOpenCV::fromCVVector( ptCurrent ) );

                    try
                    {
                        //ds point to triangulate
                        CPoint3DCAMERA vecPointTriangulated;

                        switch( iSelectedMethod )
                        {
                            case 0:
                            {
                                vecPointTriangulated = m_cTriangulator.getPointTriangulatedFull( matRight, cKeyPoint, matReferenceDescriptors.row(u) );
                                break;
                            }
                            case 1:
                            {
                                vecPointTriangulated = m_cTriangulator.getPointTriangulatedLimited( matRight, cKeyPoint, matReferenceDescriptors.row(u) );
                                break;
                            }
                            case 2:
                            {
                                vecPointTriangulated = m_cTriangulator.getPointTriangulatedAdaptive( matRight, cKeyPoint, matReferenceDescriptors.row(u) );
                                break;
                            }
                            default:
                            {
                                assert( false );
                                break;
                            }
                        }

                        //ds check if point is in front of camera an not more than a defined distance away
                        if( 1.0 < vecPointTriangulated(2) && 1000.0 > vecPointTriangulated(2) )
                        {
                            //ds add points to detected
                            vecDetectedPoints.push_back( cKeyPoint );

                            //ds positive count
                            ++uCountMatches;

                            //ds normalize coordinates for tracking
                            const CPoint2DInCameraFrameHomogenized vecPointNormalized( m_pCameraLEFT->getHomogenized( cKeyPoint ) );

                            //ds compute triangulated point in world frame
                            const CPoint3DWORLD vecPointTriangulatedWorld( p_matCurrentTransformation*vecPointTriangulated );

                            //ds draw detected point
                            cv::line( matDisplayUpper, ptCurrent, cv::Point2f( ptCurrent.x+m_uImageCols, ptCurrent.y ), CColorCodeBGR( 175, 175, 175 ) );
                            cv::circle( matDisplayUpper, ptCurrent, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                            cv::circle( matDisplayUpper, ptCurrent, cKeyPoint.size, CColorCodeBGR( 255, 0, 0 ), 1 );

                            //ds draw reprojections of triangulation
                            const CPoint3DHomogenized vecLandmarkHomo( vecPointTriangulated(0), vecPointTriangulated(1), vecPointTriangulated(2), 1.0 );
                            CPoint2DHomogenized vecLandmarkRIGHT( m_pCameraRIGHT->getHomogeneousProjection( vecLandmarkHomo ) );
                            cv::circle( matDisplayUpper, cv::Point2i( vecLandmarkRIGHT(0)+m_uImageCols, vecLandmarkRIGHT(1) ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
                        }
                        else
                        {
                            cv::circle( matDisplayUpper, ptCurrent, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                            cv::circle( matDisplayUpper, ptCurrent, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

                            std::printf( "<>(_getLandmarksGFTT) could not find match for keypoint (invalid depth: %f m)\n", vecPointTriangulated(2) );
                        }
                    }
                    catch( const CExceptionNoMatchFound& p_cException )
                    {
                        cv::circle( matDisplayUpper, ptCurrent, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                        cv::circle( matDisplayUpper, ptCurrent, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

                        std::printf( "<>(_getLandmarksGFTT) could not find match for keypoint (%s)\n", p_cException.what( ) );
                    }
                }

                //ds update image
                cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
                cv::imshow( "stereo matching", matDisplayComplete );

                //ds update lower reference
                m_matDisplayLowerReference = matDisplayUpper.clone( );

                //ds duration
                const double dComputationCost( CMiniTimer::toc( uToken ) );

                std::printf( "<>(_getLandmarksGFTT) landmarks detected: %lu\n", uCountFeatures );
                std::printf( "<>(_getLandmarksGFTT) landmarks triangulated: %lu\n", uCountMatches );
                std::printf( "<>(_getLandmarksGFTT) triangulation ratio: %f\n", static_cast< double >( uCountMatches )/uCountFeatures );
                std::printf( "<>(_getLandmarksGFTT) computation cost: %f s\n", dComputationCost );
                std::printf( "<>(_getLandmarksGFTT) algorithm benefit: [%f]\n", uCountMatches/dComputationCost );
                std::printf( "<>(_getLandmarksGFTT) ---------------------------------------------------------------\n" );

                //ds reset mouse
                CDetectorTestStereoDepth::m_ptMouseClick.x = -1;
                CDetectorTestStereoDepth::m_ptMouseClick.y = -1;
            }
            else if( -1 != CDetectorTestStereoDepth::m_ptMouseClick.x && -1 != CDetectorTestStereoDepth::m_ptMouseClick.y )
            {
                //ds clicked somewhere else, just reset
                CDetectorTestStereoDepth::m_ptMouseClick.x = -1;
                CDetectorTestStereoDepth::m_ptMouseClick.y = -1;
            }

            //ds if there was a right click
            if( CDetectorTestStereoDepth::m_bRightClicked )
            {
                std::printf( "<>() cleared reference points\n" );

                //ds reset and remember
                CDetectorTestStereoDepth::m_bRightClicked = false;

                //ds update image
                cv::hconcat( matDisplayLeftClean, matDisplayRightClean, matDisplayUpper );

                //ds redraw selectable rectangle
                cv::rectangle( matDisplayUpper, m_rectROI, CColorCodeBGR( 255, 255, 255 ) );
                cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );
                cv::imshow( "stereo matching", matDisplayComplete );
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
    }

    //ds update references
    m_matReferenceFrameLeft  = p_matImageLeft;
    m_matReferenceFrameRight = p_matImageRight;
    m_matPreviousTransformationLeft  = p_matCurrentTransformation;

    //ds increment count
    ++m_uFrameCount;
}

void CDetectorTestStereoDepth::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<CEpilinearStereoDetector>(_shutDown) termination requested, detector disabled\n" );
}

void CDetectorTestStereoDepth::_speedUp( )
{
    ++m_iPlaybackSpeedupCounter;
    m_dFrequencyPlaybackHz += std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;
    std::printf( "<CEpilinearStereoDetector>(_speedUp) increased playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
}

void CDetectorTestStereoDepth::_slowDown( )
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

void CDetectorTestStereoDepth::_catchMouseClick( int p_iEventType, int p_iX, int p_iY, int p_iFlags, void* p_hUserdata )
{
    //ds if click is enabled
    if( -1 == CDetectorTestStereoDepth::m_ptMouseClick.x && -1 == CDetectorTestStereoDepth::m_ptMouseClick.y )
    {
        if( p_iEventType == cv::EVENT_LBUTTONDOWN )
        {
            //ds set coordinates (corrected)
            CDetectorTestStereoDepth::m_ptMouseClick.x = p_iX;
            CDetectorTestStereoDepth::m_ptMouseClick.y = p_iY-2;
        }
        else if( p_iEventType == cv::EVENT_RBUTTONDOWN )
        {
            //ds catch right click
            CDetectorTestStereoDepth::m_bRightClicked = true;
        }
    }
}
