#include "CNaiveStereoDetector.h"

#include <opencv/highgui.h>
#include <Eigen/Core>

#include "configuration/CConfigurationStereoCamera.h"

//ds UGLY statics
cv::Point2i CNaiveStereoDetector::m_ptMouseClick = cv::Point2i( 0, 0 );
bool CNaiveStereoDetector::m_bRightClicked = false;

CNaiveStereoDetector::CNaiveStereoDetector( const uint32_t& p_uImageRows,
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
                                                                   m_dMatchingDistanceCutoff( 0.5 ),
                                                                   m_uFeaturesCap( 100 ),
                                                                   m_uDescriptorRadius( 40 ),
                                                                   m_uDescriptorCenterPixelOffset( (m_uDescriptorRadius-1)/2 ),
                                                                   m_rectROI( cv::Rect( m_uDescriptorCenterPixelOffset, m_uDescriptorCenterPixelOffset, m_uImageCols-2*m_uDescriptorCenterPixelOffset, m_uImageRows-2*m_uDescriptorCenterPixelOffset ) ),
                                                                   m_iExponentDepthMaximum( 3 ),
                                                                   m_dExponentStepSize( 0.01 ),
                                                                   m_iExponentDepthMinimum( std::ceil( std::log( m_dExponentStepSize ) ) ),
                                                                   m_iSamplingLowerLimit( m_iExponentDepthMinimum/m_dExponentStepSize ),
                                                                   m_iSamplingUpperLimit( m_iExponentDepthMaximum/m_dExponentStepSize ),
                                                                   m_uSamplingRange( m_iSamplingUpperLimit-m_iSamplingLowerLimit ),
                                                                   m_bDisplayImages( p_bDisplayImages ),
                                                                   m_bIsShutdownRequested( false ),
                                                                   m_dFrequencyPlaybackHz( p_uFrequencyPlaybackHz ),
                                                                   m_uFrequencyPlaybackDeltaHz( 50 ),
                                                                   m_iPlaybackSpeedupCounter( 0 ),
                                                                   m_cRandomGenerator( 1337 )
{
    //ds initialize reference frames with black images
    m_matReferenceFrameLeft    = cv::Mat::zeros( m_uImageRows, m_uImageCols, CV_8UC1 );
    m_matReferenceFrameRight   = cv::Mat::zeros( m_uImageRows, m_uImageCols, CV_8UC1 );
    m_matDisplayLowerReference = cv::Mat::zeros( m_uImageRows, 2*m_uImageCols, CV_8UC3 );

    //ds set up stereo transforms
    m_matTransformLEFTtoIMU.linear()        = Eigen::Quaterniond( -0.00333631563313, 0.00154028789643, -0.0114620263178, 0.999927556608 ).matrix( );
    m_matTransformLEFTtoIMU.translation( )  = Eigen::Vector3d( 0.0666914200614, 0.0038316133947, -0.0101029245794 );
    m_matTransformRIGHTtoIMU.linear( )      = Eigen::Quaterniond( -0.00186686047363, 6.55239757426e-05, -0.00862255915657, 0.999961080249 ).matrix( );
    m_matTransformRIGHTtoIMU.translation( ) = Eigen::Vector3d( -0.0434705406089, 0.00417949317011, -0.00942355850866 );
    m_matTransformLEFTtoRIGHT               = m_matTransformRIGHTtoIMU.inverse( )*m_matTransformLEFTtoIMU;

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

    //ds compute undistorted images
    cv::initUndistortRectifyMap( CConfigurationStereoCamera::LEFT::matIntrinsic, CConfigurationStereoCamera::LEFT::vecDistortionCoefficientsCV, CConfigurationStereoCamera::LEFT::matRectification, CConfigurationStereoCamera::LEFT::matProjection , m_prSize, CV_16SC2, m_arrMapsLEFT[0], m_arrMapsLEFT[1] );
    cv::initUndistortRectifyMap( CConfigurationStereoCamera::RIGHT::matIntrinsic, CConfigurationStereoCamera::RIGHT::vecDistortionCoefficientsCV, CConfigurationStereoCamera::RIGHT::matRectification, CConfigurationStereoCamera::RIGHT::matProjection , m_prSize, CV_16SC2, m_arrMapsRIGHT[0], m_arrMapsRIGHT[1] );

    //ds initialize the window
    cv::namedWindow( "stereo matching", cv::WINDOW_AUTOSIZE );

    //ds set mouse callback
    cv::setMouseCallback( "stereo matching", CNaiveStereoDetector::_catchMouseClick, 0 );

    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) depth sampling - upper limit: %f exp(%i)\n", std::exp( m_iExponentDepthMaximum ),m_iExponentDepthMaximum );
    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) depth sampling - lower limit: %f exp(%i)\n", std::exp( m_iExponentDepthMinimum ),m_iExponentDepthMinimum );
    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) depth sampling - samples: %f\n", static_cast< double >( m_iExponentDepthMaximum-m_iExponentDepthMinimum )/m_dExponentStepSize );
    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) depth sampling - sampling range: %li to %li\n", m_iSamplingLowerLimit, m_iSamplingUpperLimit );
    std::printf( "<CEpilinearStereoDetector>(CEpilinearStereoDetector) instance allocated\n" );
}

CNaiveStereoDetector::~CNaiveStereoDetector( )
{

}

void CNaiveStereoDetector::receivevDataVI( txt_io::PinholeImageMessage& p_cImageLeft, txt_io::PinholeImageMessage& p_cImageRight, const txt_io::CIMUMessage& p_cIMU )
{
    //ds get images into opencv format
    const cv::Mat matImageLeft( p_cImageLeft.image( ) );
    const cv::Mat matImageRight( p_cImageRight.image( ) );

    //ds detect features
    //_manualLine( matImageLeft, matImageRight );
    _detectFeaturesCorner( matImageLeft, matImageRight );
}

void CNaiveStereoDetector::receivevDataVIWithPose( std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageLeft, std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageRight, const txt_io::CIMUMessage& p_cIMU, const std::shared_ptr< txt_io::CPoseMessage >& p_cPose )
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
        _localize( matImageLeft, matImageRight, m_matTransformation );
    }
}

void CNaiveStereoDetector::_localize( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation )
{
    //ds get delta to evaluate precision
    const double dTransformationDelta( CMiniVisionToolbox::getTransformationDelta( m_matTransformationLeft, p_matCurrentTransformation ) );

    //ds log info
    std::printf( "[%lu] transformation delta: %f m\n", m_uFrameCount, dTransformationDelta );

    //ds input mats
    cv::Mat matLeft;
    cv::Mat matRight;

    //ds normalize monochrome input
    cv::equalizeHist( p_matImageLeft, matLeft );
    cv::equalizeHist( p_matImageRight, matRight );

    //ds remap
    cv::remap( matLeft, matLeft, m_arrMapsLEFT[0], m_arrMapsLEFT[1], cv::INTER_LINEAR );
    cv::remap( matRight, matRight, m_arrMapsRIGHT[0], m_arrMapsRIGHT[1], cv::INTER_LINEAR );

    //ds draw position on trajectory mat
    cv::circle( m_matTrajectoryXY, cv::Point2d( 50+p_matCurrentTransformation.translation( )( 0 )*10, 175-p_matCurrentTransformation.translation( )( 1 )*10 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_matCurrentTransformation.translation( )( 2 )*100 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-dTransformationDelta*100 ), 1, cv::Scalar( 0, 0, 255 ), -1 );

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
    //_drawProjectedEpipolarLineFundamental( p_matCurrentTransformation, matDisplayLeft, matLeft );
    _drawProjectedEpipolarLineEssential1( p_matCurrentTransformation, matDisplayLeft, matLeft );
    //_drawProjectedEpipolarLineEssential2( p_matCurrentTransformation, matDisplayLeft, matLeft );
    //_drawProjectedEpipolarLineDepthSampling( p_matCurrentTransformation, matDisplayLeft );
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
                break;
            }
        }

        //ds reset keystroke
        iLastKeyStroke = -1;

        //ds check current displacement
        const double dRelativeDistanceMeters( p_matCurrentTransformation.translation( ).squaredNorm( ) );

        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" );
        std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) moved distance: %f requesting user input at frame: %lu\n", dRelativeDistanceMeters, m_uFrameCount );

        //ds enable mouse click
        CNaiveStereoDetector::m_ptMouseClick.x = -1;
        CNaiveStereoDetector::m_ptMouseClick.y = -1;
        CNaiveStereoDetector::m_bRightClicked = false;
        cv::rectangle( matDisplayUpperTemporary, m_rectROI, CColorCode( 255, 255, 255 ) );
        cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
        cv::imshow( "stereo matching", matDisplayComplete );

        //ds modification detection
        bool bWasThereAModification( false );

        //ds record mouseclicks until space is pressed
        while( CConfigurationOpenCV::KeyStroke::iSpace != iLastKeyStroke )
        {
            if( CNaiveStereoDetector::m_ptMouseClick.inside( m_rectROI ) )
            {
                //ds caught mouse click
                std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) ---------------------------------------------------------------\n" );
                std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) selected point: ( %u, %u )\n", CNaiveStereoDetector::m_ptMouseClick.x, CNaiveStereoDetector::m_ptMouseClick.y );

                //ds handle the marked point
                _triangulatePointSURF( matLeft, matRight, matDisplayUpper, matDisplayUpperTemporary, p_matCurrentTransformation );
                //_triangulatePointDepthSampling( matLeft, matRight, matDisplayUpper, matDisplayUpperTemporary, p_matCurrentTransformation );

                //ds update image
                cv::vconcat( matDisplayUpperTemporary, m_matDisplayLowerReference, matDisplayComplete );
                cv::imshow( "stereo matching", matDisplayComplete );

                //ds reset mouse
                CNaiveStereoDetector::m_ptMouseClick.x = -1;
                CNaiveStereoDetector::m_ptMouseClick.y = -1;

                //ds modified
                bWasThereAModification = true;
            }
            else if( -1 != CNaiveStereoDetector::m_ptMouseClick.x && -1 != CNaiveStereoDetector::m_ptMouseClick.y )
            {
                //ds clicked somewhere else, just reset
                CNaiveStereoDetector::m_ptMouseClick.x = -1;
                CNaiveStereoDetector::m_ptMouseClick.y = -1;
            }

            //ds if there was a right click
            if( CNaiveStereoDetector::m_bRightClicked )
            {
                //ds we have to clean up everything
                m_vecReferencePoints.clear( );

                std::printf( "<CEpilinearStereoDetector>(_drawEpisubsequentLine) cleared reference points\n" );

                //ds reset and remember
                CNaiveStereoDetector::m_bRightClicked = false;

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
    m_matTransformationLeft  = p_matCurrentTransformation;

    //ds increment count
    ++m_uFrameCount;
}

void CNaiveStereoDetector::_drawProjectedEpipolarLineFundamental( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage )
{
    //ds for all the registered points
    for( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptorSURF, Eigen::Isometry3d, CColorCode > vecReferencePoint: m_vecReferencePoints )
    {
        //ds get fundamental matrixp_isoCurrentTransformation
        const Eigen::Matrix3d matFundamental( CMiniVisionToolbox::getFundamental( std::get< 4 >( vecReferencePoint ), p_matCurrentTransformation, m_matIntrinsicLEFT ) );

        //ds get 2d point
        const CPoint2DInCameraFrame vec2DReferencePoint( std::get< 0 >( vecReferencePoint ) );

        //ds compute the projection of the point (line) in the current frame
        const Eigen::Vector3d vecCoefficients( matFundamental*Eigen::Vector3d( vec2DReferencePoint(0), vec2DReferencePoint(1), 1.0 ) );

        //std::printf( "<CSimpleFeatureDetector>(_drawEpisubsequentLine) curve equation: f(x) = %fx + %f\n", -vecCoefficients(0)/vecCoefficients(1), -vecCoefficients(2)/vecCoefficients(1) );

        //ds compute maximum and minimum points (from top to bottom line)
        double dXforYMinimum( -vecCoefficients(2)/vecCoefficients(0) );
        double dXforYMaximum( -( vecCoefficients(2)+vecCoefficients(1)*m_uImageRows )/vecCoefficients(0) );
        double dYforYMinimum( 0.0 );
        double dYforYMaximum( m_uImageRows );

        //ds shift the points to 0 max range
        if( 0 > dXforYMinimum )
        {
            //ds we have to recompute Y
            dXforYMinimum = 0;
            dYforYMinimum     = -vecCoefficients(2)/vecCoefficients(1);
        }
        else if( m_uImageCols < static_cast< uint32_t >( std::lround( dXforYMinimum ) ) )
        {
            dXforYMinimum = m_uImageCols;
            dYforYMinimum     = -( vecCoefficients(2)+vecCoefficients(0)*m_uImageCols )/vecCoefficients(1);
        }
        if( 0 > dXforYMaximum )
        {
            dXforYMaximum = 0;
            dYforYMaximum     = -vecCoefficients(2)/vecCoefficients(1);
        }
        else if( m_uImageCols < static_cast< uint32_t >( std::lround( dXforYMaximum ) ) )
        {
            dXforYMaximum = m_uImageCols;
            dYforYMaximum     = -( vecCoefficients(2)+vecCoefficients(0)*m_uImageCols )/vecCoefficients(1);
        }

        //ds swap for consistency as we are looping from x=0 to x=max
        if( dXforYMinimum > dXforYMaximum )
        {
            std::swap( dXforYMinimum, dXforYMaximum );
            std::swap( dYforYMinimum, dYforYMaximum );
        }

        //ds draw the line
        //cv::line( p_matDisplay, cv::Point2i( dXforYMinimum, dYforYMinimum ), cv::Point2i( dXforYMaximum, dYforYMaximum ), std::get< 4 >( vecReferencePoint ) );

        //ds number of keypoints to check
        //const uint32_t uDeltaX( uXforYMaximum-uXforYMinimum );
        //const uint32_t uDeltaY( uYforYMaximum-uYforYMinimum );
        const double dDeltaX( dXforYMaximum-dXforYMinimum );
        const double dDeltaY( dYforYMaximum-dYforYMinimum );

        //ds line length
        const uint32_t uSamples( std::lround( std::sqrt( dDeltaX*dDeltaX + dDeltaY*dDeltaY ) ) );

        //ds compute step size for sampling
        const double dStepSizeX( dDeltaX/uSamples );

        //ds keypoint vectors
        std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
        vecReferenceKeyPoints[0] = std::get< 2 >( vecReferencePoint );
        std::vector< cv::KeyPoint > vecPoolKeyPoints( uSamples );

        //ds set the keypoints (by sampling over x)
        for( uint32_t u = 0; u < uSamples; ++u )
        {
            //ds compute current x and y
            const double dX( dXforYMinimum+u*dStepSizeX );
            const double dY( -( vecCoefficients(2)+vecCoefficients(0)*dX )/vecCoefficients(1) );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dX, dY, m_uDescriptorRadius );

            cv::circle( p_matDisplay, cv::Point2i( dX, dY ), 1, CColorCode( 255, 0, 0 ), -1 );
        }

        std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) keypoints pool size: %lu\n", vecPoolKeyPoints.size( ) );

        //ds descriptor pool to match
        cv::Mat matReferenceDescriptor( std::get< 3 >( vecReferencePoint ) );
        cv::Mat matPoolDescriptors;

        //ds compute descriptors
        m_cDetectorSURF.compute( p_matImage, vecPoolKeyPoints, matPoolDescriptors );

        //ds match the descriptors
        std::vector< std::vector< cv::DMatch > > vecMatches;
        m_cDescriptorMatcher.knnMatch( matReferenceDescriptor, matPoolDescriptors, vecMatches, 10 );

        //ds buffer first match
        const cv::DMatch cBestMatch( vecMatches[0][0] );

        //ds get the matching keypoint
        cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

        if( m_dMatchingDistanceCutoff < cBestMatch.distance )
        {
            //ds drop the match
            std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) matching distance: %f - dropped match\n", cBestMatch.distance );
        }
        else
        {
            //ds get the matching keypoint
            cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

            //ds draw the match
            cv::circle( p_matDisplay, ptMatch, 2, CColorCode( 0, 255, 0 ), -1 );
            cv::circle( p_matDisplay, ptMatch, m_uDescriptorRadius, CColorCode( 0, 255, 0 ), 1 );
        }
    }
}

void CNaiveStereoDetector::_drawProjectedEpipolarLineEssential1( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage )
{
    //ds for all the registered points
    for( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptorSURF, Eigen::Isometry3d, CColorCode > vecReferencePoint: m_vecReferencePoints )
    {
        //ds get essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( std::get< 4 >( vecReferencePoint ), p_matCurrentTransformation ) );

        //ds get 2d point
        const CPoint2DInCameraFrame vec2DReferencePoint( std::get< 0 >( vecReferencePoint ) );

        //ds compute the projection of the point (line) in the current frame
        const Eigen::Vector3d vecCoefficients( matEssential*CMiniVisionToolbox::getNormalized( vec2DReferencePoint,
                                                                                               CConfigurationStereoCamera::LEFT::dFx,
                                                                                               CConfigurationStereoCamera::LEFT::dFy,
                                                                                               CConfigurationStereoCamera::LEFT::dCx,
                                                                                               CConfigurationStereoCamera::LEFT::dCy ) );

        //std::printf( "<CSimpleFeatureDetector>(_drawEpisubsequentLine) curve equation: f(x) = %fx + %f\n", -vecCoefficients(0)/vecCoefficients(1), -vecCoefficients(2)/vecCoefficients(1) );

        //ds compute maximum and minimum points (from top to bottom line)
        double dXforYMinimum( CMiniVisionToolbox::getDenormalized( -vecCoefficients(2)/vecCoefficients(0), CConfigurationStereoCamera::LEFT::dFx, CConfigurationStereoCamera::LEFT::dCx ) );
        double dXforYMaximum( CMiniVisionToolbox::getDenormalized( -( vecCoefficients(2)+vecCoefficients(1) )/vecCoefficients(0), CConfigurationStereoCamera::LEFT::dFx, CConfigurationStereoCamera::LEFT::dCx ) );
        double dYforYMinimum( 0.0 );
        double dYforYMaximum( 480.0 );

        //ds shift the points to 0 max range
        if( 0 > dXforYMinimum )
        {
            //ds we have to recompute Y
            dXforYMinimum = 0;
            dYforYMinimum = CMiniVisionToolbox::getDenormalized( -vecCoefficients(2)/vecCoefficients(1), CConfigurationStereoCamera::LEFT::dFy, CConfigurationStereoCamera::LEFT::dCy );
        }
        else if( m_uImageCols < static_cast< uint32_t >( std::lround( dXforYMinimum ) ) )
        {
            dXforYMinimum = 752.0;
            dYforYMinimum = CMiniVisionToolbox::getDenormalized( -( vecCoefficients(2)+vecCoefficients(0) )/vecCoefficients(1), CConfigurationStereoCamera::LEFT::dFy, CConfigurationStereoCamera::LEFT::dCy );
        }
        if( 0 > dXforYMaximum )
        {
            dXforYMaximum = 0;
            dYforYMaximum = CMiniVisionToolbox::getDenormalized( -vecCoefficients(2)/vecCoefficients(1), CConfigurationStereoCamera::LEFT::dFy, CConfigurationStereoCamera::LEFT::dCy );
        }
        else if( m_uImageCols < static_cast< uint32_t >( std::lround( dXforYMaximum ) ) )
        {
            dXforYMaximum = 752.0;
            dYforYMaximum = CMiniVisionToolbox::getDenormalized( -( vecCoefficients(2)+vecCoefficients(0) )/vecCoefficients(1), CConfigurationStereoCamera::LEFT::dFy, CConfigurationStereoCamera::LEFT::dCy );
        }

        //ds swap for consistency as we are looping from x=0 to x=max
        if( dXforYMinimum > dXforYMaximum )
        {
            std::swap( dXforYMinimum, dXforYMaximum );
            std::swap( dYforYMinimum, dYforYMaximum );
        }

        //ds draw the line
        //cv::line( p_matDisplay, cv::Point2i( dXforYMinimum, dYforYMinimum ), cv::Point2i( dXforYMaximum, dYforYMaximum ), std::get< 4 >( vecReferencePoint ) );

        //ds number of keypoints to check
        //const uint32_t uDeltaX( uXforYMaximum-uXforYMinimum );
        //const uint32_t uDeltaY( uYforYMaximum-uYforYMinimum );
        const double dDeltaX( dXforYMaximum-dXforYMinimum );
        const double dDeltaY( dYforYMaximum-dYforYMinimum );

        //ds line length
        const uint32_t uSamples( std::lround( std::sqrt( dDeltaX*dDeltaX + dDeltaY*dDeltaY ) ) );

        //ds compute step size for sampling
        const double dStepSizeX( dDeltaX/uSamples );

        //ds keypoint vectors
        std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
        vecReferenceKeyPoints[0] = std::get< 2 >( vecReferencePoint );
        std::vector< cv::KeyPoint > vecPoolKeyPoints( uSamples );

        //ds set the keypoints (by sampling over x)
        for( uint32_t u = 0; u < uSamples; ++u )
        {
            //ds compute current x and y
            const double dX( dXforYMinimum+u*dStepSizeX );
            const double dY( CMiniVisionToolbox::getDenormalized( -( vecCoefficients(2)+vecCoefficients(0)*CMiniVisionToolbox::getNormalized( dX, CConfigurationStereoCamera::LEFT::dFx, CConfigurationStereoCamera::LEFT::dCx ) )/vecCoefficients(1), CConfigurationStereoCamera::LEFT::dFy, CConfigurationStereoCamera::LEFT::dCy ) );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dX, dY, m_uDescriptorRadius );

            cv::circle( p_matDisplay, cv::Point2i( dX, dY ), 1, CColorCode( 0, 0, 255 ), -1 );
        }

        std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) keypoints pool size: %lu\n", vecPoolKeyPoints.size( ) );

        //ds descriptor pool to match
        cv::Mat matReferenceDescriptor( std::get< 3 >( vecReferencePoint ) );
        cv::Mat matPoolDescriptors;

        //ds compute descriptors
        m_cDetectorSURF.compute( p_matImage, vecPoolKeyPoints, matPoolDescriptors );

        //ds match the descriptors
        std::vector< std::vector< cv::DMatch > > vecMatches;
        m_cDescriptorMatcher.knnMatch( matReferenceDescriptor, matPoolDescriptors, vecMatches, 10 );

        //ds buffer first match
        const cv::DMatch cBestMatch( vecMatches[0][0] );

        //ds get the matching keypoint
        cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

        if( m_dMatchingDistanceCutoff < cBestMatch.distance )
        {
            //ds drop the match
            std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) matching distance: %f - dropped match\n", cBestMatch.distance );
        }
        else
        {
            //ds get the matching keypoint
            cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

            //ds draw the match
            cv::circle( p_matDisplay, ptMatch, 2, CColorCode( 0, 255, 0 ), -1 );
            cv::circle( p_matDisplay, ptMatch, m_uDescriptorRadius, CColorCode( 0, 255, 0 ), 1 );
        }
    }
}

void CNaiveStereoDetector::_drawProjectedEpipolarLineEssential2( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage )
{
    //ds for all the registered points
    for( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptorSURF, Eigen::Isometry3d, CColorCode > vecReferencePoint: m_vecReferencePoints )
    {
        //ds get essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( std::get< 4 >( vecReferencePoint ), p_matCurrentTransformation ) );

        //ds get 2d point
        const CPoint2DInCameraFrame vec2DReferencePoint( std::get< 0 >( vecReferencePoint ) );

        //ds compute the projection of the point (line) in the current frame
        const Eigen::Vector3d vecCoefficients( matEssential*Eigen::Vector3d( vec2DReferencePoint(0)/752.0, vec2DReferencePoint(1)/480.0, 1.0 ) );

        //std::printf( "<CSimpleFeatureDetector>(_drawEpisubsequentLine) curve equation: f(x) = %fx + %f\n", -vecCoefficients(0)/vecCoefficients(1), -vecCoefficients(2)/vecCoefficients(1) );

        //ds compute maximum and minimum points (from top to bottom line)
        double dXforYMinimum( -vecCoefficients(2)/vecCoefficients(0) );
        double dXforYMaximum( -( vecCoefficients(2)+vecCoefficients(1) )/vecCoefficients(0) );
        double dYforYMinimum( 0.0 );
        double dYforYMaximum( 1.0 );

        //ds shift the points to 0 max range
        if( 0 > dXforYMinimum )
        {
            //ds we have to recompute Y
            dXforYMinimum = 0;
            dYforYMinimum = -vecCoefficients(2)/vecCoefficients(1);
        }
        else if( m_uImageCols < static_cast< uint32_t >( std::lround( dXforYMinimum ) ) )
        {
            dXforYMinimum = 1.0;
            dYforYMinimum = -( vecCoefficients(2)+vecCoefficients(0) )/vecCoefficients(1);
        }
        if( 0 > dXforYMaximum )
        {
            dXforYMaximum = 0;
            dYforYMaximum = -vecCoefficients(2)/vecCoefficients(1);
        }
        else if( m_uImageCols < static_cast< uint32_t >( std::lround( dXforYMaximum ) ) )
        {
            dXforYMaximum = 1.0;
            dYforYMaximum = -( vecCoefficients(2)+vecCoefficients(0) )/vecCoefficients(1);
        }

        //ds swap for consistency as we are looping from x=0 to x=max
        if( dXforYMinimum > dXforYMaximum )
        {
            std::swap( dXforYMinimum, dXforYMaximum );
            std::swap( dYforYMinimum, dYforYMaximum );
        }

        //ds draw the line
        //cv::line( p_matDisplay, cv::Point2i( dXforYMinimum, dYforYMinimum ), cv::Point2i( dXforYMaximum, dYforYMaximum ), std::get< 4 >( vecReferencePoint ) );

        //ds number of keypoints to check
        //const uint32_t uDeltaX( uXforYMaximum-uXforYMinimum );
        //const uint32_t uDeltaY( uYforYMaximum-uYforYMinimum );
        const double dDeltaX( dXforYMaximum-dXforYMinimum );
        const double dDeltaY( dYforYMaximum-dYforYMinimum );

        //ds line length
        const uint32_t uSamples( std::lround( std::sqrt( dDeltaX*dDeltaX*752*752 + dDeltaY*dDeltaY*480*480 ) ) );

        //ds compute step size for sampling
        const double dStepSizeX( dDeltaX/uSamples );

        //ds keypoint vectors
        std::vector< cv::KeyPoint > vecReferenceKeyPoints( 1 );
        vecReferenceKeyPoints[0] = std::get< 2 >( vecReferencePoint );
        std::vector< cv::KeyPoint > vecPoolKeyPoints( uSamples );

        //ds set the keypoints (by sampling over x)
        for( uint32_t u = 0; u < uSamples; ++u )
        {
            //ds compute current x and y
            const double dX( dXforYMinimum+u*dStepSizeX );
            const double dY( -( vecCoefficients(2)+vecCoefficients(0)*dX )/vecCoefficients(1) );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dX*752, dY*480, m_uDescriptorRadius );

            cv::circle( p_matDisplay, cv::Point2i( dX*752, dY*480 ), 1, CColorCode( 0, 255, 0 ), -1 );
        }

        std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) keypoints pool size: %lu\n", vecPoolKeyPoints.size( ) );

        //ds descriptor pool to match
        cv::Mat matReferenceDescriptor( std::get< 3 >( vecReferencePoint ) );
        cv::Mat matPoolDescriptors;

        //ds compute descriptors
        m_cDetectorSURF.compute( p_matImage, vecPoolKeyPoints, matPoolDescriptors );

        //ds match the descriptors
        std::vector< std::vector< cv::DMatch > > vecMatches;
        m_cDescriptorMatcher.knnMatch( matReferenceDescriptor, matPoolDescriptors, vecMatches, 10 );

        //ds buffer first match
        const cv::DMatch cBestMatch( vecMatches[0][0] );

        //ds get the matching keypoint
        cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

        if( m_dMatchingDistanceCutoff < cBestMatch.distance )
        {
            //ds drop the match
            std::printf( "<CEpilinearStereoDetector>(_triangulatePointSURF) matching distance: %f - dropped match\n", cBestMatch.distance );
        }
        else
        {
            //ds get the matching keypoint
            cv::Point2f ptMatch( vecPoolKeyPoints[ cBestMatch.trainIdx ].pt );

            //ds draw the match
            cv::circle( p_matDisplay, ptMatch, 2, CColorCode( 0, 255, 0 ), -1 );
            cv::circle( p_matDisplay, ptMatch, m_uDescriptorRadius, CColorCode( 0, 255, 0 ), 1 );
        }
    }
}

void CNaiveStereoDetector::_drawProjectedEpipolarLineDepthSampling( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay )
{
    //ds for all the registered points
    for( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptorSURF, Eigen::Isometry3d, CColorCode > vecReferencePoint: m_vecReferencePoints )
    {
        //ds get transformation
        const Eigen::Isometry3d matTransformation( p_matCurrentTransformation*( std::get< 4 >( vecReferencePoint ) ).inverse( ) );

        //ds get 2d point
        const CPoint2DInCameraFrame vec2DReferencePoint( std::get< 0 >( vecReferencePoint ) );

        //ds sample depth
        for( int iExponent = m_iSamplingLowerLimit; iExponent < m_iSamplingUpperLimit; ++iExponent )
        {
            //ds current depth value
            const double dZ( std::exp( iExponent*m_dExponentStepSize ) );

            //ds compute X and Y value
            const double dX = dZ/( CConfigurationStereoCamera::LEFT::dFx )*( vec2DReferencePoint(0)-CConfigurationStereoCamera::LEFT::dCx );
            const double dY = dZ/( CConfigurationStereoCamera::LEFT::dFy )*( vec2DReferencePoint(1)-CConfigurationStereoCamera::LEFT::dCy );

            //ds transform the point to the other camera frame in world
            const Eigen::Vector4d vecPointWORLDinRIGHT( matTransformation*Eigen::Vector4d( dX, dY, dZ, 1.0 ) );

            //ds project point back into camera frame
            Eigen::Vector3d vecPointReprojected( m_matProjectionLEFT*vecPointWORLDinRIGHT );

            //ds get point coordinates
            const double dU( vecPointReprojected(0)/vecPointReprojected(2) );
            const double dV( vecPointReprojected(1)/vecPointReprojected(2) );

            //ds go to pixel space
            const cv::Point2i ptPointProjected( std::lround( dU ), std::lround( dV ) );

            //ds check if the point is in the valid range
            if( ptPointProjected.inside( m_rectROI ) )
            {
                cv::circle( p_matDisplay, ptPointProjected, 1, CColorCode( 0, 0, 0 ), -1 );
            }
        }
    }
}

void CNaiveStereoDetector::_triangulatePointSURF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, cv::Mat& p_matDisplayUpper, cv::Mat& p_matDisplayUpperTemporary, const Eigen::Isometry3d p_matCurrentTransformation )
{
    //ds get a random color code
    const CColorCode vecColorCode( m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ) );

    //ds get the reference point locally
    const cv::Point2d ptReference( CNaiveStereoDetector::m_ptMouseClick.x, CNaiveStereoDetector::m_ptMouseClick.y );

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
    m_cDetectorSURF.compute( p_matImageLeft, vecReferenceKeyPoints, matReferenceDescriptor );
    m_cDetectorSURF.compute( p_matImageRight, vecPoolKeyPoints, matPoolDescriptors );

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
    m_vecReferencePoints.push_back( std::tuple< CPoint2DInCameraFrame, CPoint3DInCameraFrame, cv::KeyPoint, CDescriptorSURF, Eigen::Isometry3d, CColorCode >( CPoint2DInCameraFrame( ptReference.x, ptReference.y ),
                                                                                                                                                             vec3DReferencePointSVDLS,
                                                                                                                                                             vecReferenceKeyPoints[0],
                                                                                                                                                             matReferenceDescriptor,
                                                                                                                                                             p_matCurrentTransformation,
                                                                                                                                                             vecColorCode ) );
}

//ds TODO remove static_casts
void CNaiveStereoDetector::_triangulatePointDepthSampling( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, cv::Mat& p_matDisplayUpper, cv::Mat& p_matDisplayUpperTemporary, const Eigen::Isometry3d p_matCurrentTransformation )
{
    //ds get a random color code
    const CColorCode vecColorCode( m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ) );

    //ds draw mouseclick
    cv::circle( p_matDisplayUpperTemporary, CNaiveStereoDetector::m_ptMouseClick, 2, CColorCode( 0, 0, 255 ), -1 );

    //ds draw horizontal epipolar line (for comparison)
    cv::line( p_matDisplayUpperTemporary, CNaiveStereoDetector::m_ptMouseClick, cv::Point( m_uImageCols+CNaiveStereoDetector::m_ptMouseClick.x, CNaiveStereoDetector::m_ptMouseClick.y ), vecColorCode );

    //ds reference roi
    cv::Rect cROILEFT( CNaiveStereoDetector::m_ptMouseClick.x-m_uDescriptorCenterPixelOffset, CNaiveStereoDetector::m_ptMouseClick.y-m_uDescriptorCenterPixelOffset, m_uDescriptorRadius, m_uDescriptorRadius );

    //ds draw the roi
    cv::rectangle( p_matDisplayUpperTemporary, cROILEFT, CColorCode( 0, 0, 255 ) );

    //ds gaussian weighting
    //const double dSigma( 0.5 );

    //ds current descriptor
    //const Eigen::Matrix< double, uDescriptorRadius, uDescriptorRadius > matDescriptorLEFT( CMiniVisionToolbox::getGaussianWeighted< uDescriptorRadius, uDescriptorRadius >( CWrapperOpenCV::fromCVMatrix< uint8_t, uDescriptorRadius, uDescriptorRadius >( p_matImageLeft( cROILEFT ) ), dSigma ) );

    //ds distorted point
    const Eigen::Vector2d vecPointDistortedLEFT( CNaiveStereoDetector::m_ptMouseClick.x/752.0, CNaiveStereoDetector::m_ptMouseClick.y/480.0 );

    //ds compute undistorted point
    //const Eigen::Vector2d vecPointUndistoredLEFT( CMiniVisionToolbox::getPointUndistortedPlumbBob( CWrapperOpenCV::fromCVVector< int, 2 >( CEpilinearStereoDetector::m_ptMouseClick ), m_matProjectionLEFT, CConfigurationStereoCamera::LEFT::vecDistortionCoefficients ) );

    //std::printf( "distorted: %i %i\n",CEpilinearStereoDetector::m_ptMouseClick.x, CEpilinearStereoDetector::m_ptMouseClick.y );
    //std::printf( "undistorted: %f %f \n", vecPointUndistoredLEFT(0), vecPointUndistoredLEFT(1) );

    //ds optimization denominator (the smaller the more aggressive)
    //uint32_t uOptimizationDenominator( 2 );

    //ds matching point
    cv::Point2i ptMatch( 0, 0 );

    //ds sample depth
    for( int iExponent = m_iSamplingLowerLimit; iExponent < m_iSamplingUpperLimit; ++iExponent )
    {
        //ds current depth value
        const double dZ( std::exp( iExponent*m_dExponentStepSize ) );

        //ds get world coordinates in left camera frame
        const double dX = dZ*vecPointDistortedLEFT(0);//dZ/( CConfigurationStereoCamera::LEFT::dFx )*( vecPointDistortedLEFT(0)-CConfigurationStereoCamera::LEFT::dCx );
        const double dY = dZ*vecPointDistortedLEFT(1);//dZ/( CConfigurationStereoCamera::LEFT::dFy )*( vecPointDistortedLEFT(1)-CConfigurationStereoCamera::LEFT::dCy );

        //ds transform the point to the right camera frame
        const Eigen::Vector4d vecPointWORLDinRIGHT( m_matTransformLEFTtoRIGHT*Eigen::Vector4d( dX, dY, dZ, 1.0 ) );

        //ds project point into other camera frame
        Eigen::Vector3d vecPointRIGHTScaled( m_matProjectionRIGHT*vecPointWORLDinRIGHT );

        //ds normalize the point
        vecPointRIGHTScaled /= vecPointRIGHTScaled(2);

        /*ds normalize pixel coordinates
        const Eigen::Vector2d vecPointDistortedRIGHT( vecPointRIGHTScaled(0)/752.0, vecPointRIGHTScaled(1)/480.0 );

        //ds get distorted version
        //const Eigen::Vector2d vecPointUndistortedRIGHT( CMiniVisionToolbox::getPointDistortedPlumbBob( vecPointRIGHT, CConfigurationStereoCamera::RIGHT::vecPrincipalPoint, CConfigurationStereoCamera::RIGHT::vecDistortionCoefficients ) );
        const Eigen::Vector2d vecPointUndistortedRIGHT( CMiniVisionToolbox::getPointUndistortedPlumbBob( vecPointDistortedRIGHT, m_matProjectionRIGHT, CConfigurationStereoCamera::RIGHT::vecDistortionCoefficients ) );

        std::printf( "vecPointUndistortedRIGHT %f %f\n", vecPointUndistortedRIGHT(0), vecPointUndistortedRIGHT(1) );

        //ds get normalized plane coordinates
        const int32_t iU( std::lround( static_cast< double >( vecPointUndistortedRIGHT(0) ) ) );
        const int32_t iV( std::lround( static_cast< double >( vecPointUndistortedRIGHT(1) ) ) );

        //ds check for optimization (skip processing steps)
        if( 0 > iU )
        {
            //ds skip steps
            iExponent += m_uSamplingRange/( uOptimizationDenominator*uOptimizationDenominator );

            //ds diminishing returns
            ++uOptimizationDenominator;
        }
        else
        {
            //ds check if we can process the matching
            if( m_uDescriptorCenterPixelOffset < static_cast< unsigned int >( iU ) && m_uImageCols-m_uDescriptorCenterPixelOffset > static_cast< unsigned int >( iU ) &&
                m_uDescriptorCenterPixelOffset < static_cast< unsigned int >( iV ) && m_uImageRows-m_uDescriptorCenterPixelOffset > static_cast< unsigned int >( iV ) )
            {
                cv::circle( p_matDisplayUpperTemporary, cv::Point( m_uImageCols+iU, iV ), 1, CColorCode( 255, 0, 0 ), -1 );
                */
                /*ds current roi
                cv::Rect cROIRIGHT( iU-m_uDescriptorCenterPixelOffset, iV-m_uDescriptorCenterPixelOffset, uDescriptorRadius, uDescriptorRadius );

                //ds pick it from the matrix
                const Eigen::Matrix< double, uDescriptorRadius, uDescriptorRadius > matDescriptorRIGHT( CMiniVisionToolbox::getGaussianWeighted< uDescriptorRadius, uDescriptorRadius >( CWrapperOpenCV::fromCVMatrix< uint8_t, uDescriptorRadius, uDescriptorRadius >( p_matImageRight( cROIRIGHT ) ), dSigma ) );

                //ds get the sum of differences
                const double dCurrentMatch( _getMatchingDistance( matDescriptorLEFT, matDescriptorRIGHT ) );

                if( dMatchingDistance >= dCurrentMatch )
                {
                    std::printf( "sum %f\n", dCurrentMatch );
                    dMatchingDistance = dCurrentMatch;
                    ptMatch = cv::Point( iU, iV );
                }*//*
            }
        }*/
    }

    cv::Rect cROIRIGHTDraw( m_uImageCols+ptMatch.x-m_uDescriptorCenterPixelOffset, ptMatch.y-m_uDescriptorCenterPixelOffset, m_uDescriptorRadius, m_uDescriptorRadius );
    cv::rectangle( p_matDisplayUpperTemporary, cROIRIGHTDraw, CColorCode( 0, 255, 0 ) );
}

void CNaiveStereoDetector::_triangulatePointDepthSamplingLinear( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, cv::Mat& p_matDisplayUpper, cv::Mat& p_matDisplayUpperTemporary, const Eigen::Isometry3d p_matCurrentTransformation )
{
    //ds get a random color code
    const CColorCode vecColorCode( m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ), m_cRandomGenerator.uniform( 0, 255 ) );

    //ds draw mouseclick
    cv::circle( p_matDisplayUpperTemporary, CNaiveStereoDetector::m_ptMouseClick, 2, CColorCode( 0, 0, 255 ), -1 );

    //ds draw horizontal epipolar line (for comparison)
    cv::line( p_matDisplayUpperTemporary, CNaiveStereoDetector::m_ptMouseClick, cv::Point( m_uImageCols+CNaiveStereoDetector::m_ptMouseClick.x, CNaiveStereoDetector::m_ptMouseClick.y ), vecColorCode );

    //ds get coordinates of current point
    //const double dU( CEpilinearStereoDetector::m_ptMouseClick.x );
    //const double dV( CEpilinearStereoDetector::m_ptMouseClick.y );

    //ds compute A matrix
    //Eigen::Matrix< double, 2, 3 > matA;
    //matA.block( 0, 0, 1, 3 ) = dU*m_matMLEFT.row(2)-m_matMLEFT.row(0);
    //matA.block( 1, 0, 1, 3 ) = dV*m_matMLEFT.row(2)-m_matMLEFT.row(1);

    //ds sample depth
    for( int iExponent = m_iSamplingLowerLimit; iExponent < m_iSamplingUpperLimit; ++iExponent )
    {
        //ds current depth value
        const double dZ( std::exp( iExponent*m_dExponentStepSize ) );

        //ds compute RHS
        //const Eigen::Vector2d vecRHS( static_cast< double >( matA(0,2)*dZ - m_vecTLEFT(0) + dU*m_vecTLEFT(2) ), static_cast< double >( matA(1,2)*dZ - m_vecTLEFT(1) + dV*m_vecTLEFT(2) ) );

        //ds solve the system
        //const Eigen::Vector2d vecPointWORLDinLEFT( matA.block( 0, 0, 2, 2 ).fullPivHouseholderQr( ).solve( -vecRHS ) );

        //ds get world coordinates
        const double dX = dZ/(CConfigurationStereoCamera::LEFT::dFx)*( CNaiveStereoDetector::m_ptMouseClick.x-CConfigurationStereoCamera::LEFT::dCx );
        const double dY = dZ/(CConfigurationStereoCamera::LEFT::dFy)*( CNaiveStereoDetector::m_ptMouseClick.y-CConfigurationStereoCamera::LEFT::dCy );

        //ds transform the point to the right camera frame
        const Eigen::Vector4d vecPointWORLDinRIGHT( m_matTransformLEFTtoRIGHT*Eigen::Vector4d( dX, dY, dZ, 1.0 ) );

        //ds project point into other camera frame
        const Eigen::Vector3d vecPointRIGHT( m_matProjectionRIGHT*vecPointWORLDinRIGHT );

        //ds get normalized plane coordinates
        const int32_t iU( std::lround( static_cast< double >( vecPointRIGHT(0)/vecPointRIGHT(2) ) ) );
        const int32_t iV( std::lround( static_cast< double >( vecPointRIGHT(1)/vecPointRIGHT(2) ) ) );

        //ds draw it
        if( 0 <= iU )
        {
            cv::circle( p_matDisplayUpperTemporary, cv::Point( m_uImageCols+iU, iV ), 1, CColorCode( 255, 0, 0 ), 1 );
        }


        //ds get world coordinates
        //const double dX = dZ/(CConfigurationStereoCamera::LEFT::dFx)*( CEpilinearStereoDetector::m_ptMouseClick.x-CConfigurationStereoCamera::LEFT::dCx );
        //const double dY = dZ/(CConfigurationStereoCamera::LEFT::dFy)*( CEpilinearStereoDetector::m_ptMouseClick.y-CConfigurationStereoCamera::LEFT::dCy );

        //std::printf( "%f, %f, %f\n", dX, dY, dZ );

        //ds reproject point
        //Eigen::Vector3d vecPointRIGHT( m_matProjectionRIGHT*Eigen::Vector4d( dX, dY, dZ, 1.0 ) );

        //ds normalize
        //vecPointRIGHT /= vecPointRIGHT(2);

        //ds compute pixel coordinates in the right frame
        //const double iU( CConfigurationStereoCamera::RIGHT::dCx+CConfigurationStereoCamera::RIGHT::dFx/dZ*dX );
        //const double iV( CConfigurationStereoCamera::RIGHT::dCy+CConfigurationStereoCamera::RIGHT::dFy/dZ*dY );

        //std::printf( "%f, %f\n", vecPointRIGHT(0), vecPointRIGHT(1) );*/

        /*const double dZ1 = ( dU*m_matMLEFT(2,2)-m_matMLEFT(0,2) )*dZ;
        const double dZ2 = ( dV*m_matMLEFT(2,2)-m_matMLEFT(1,2) )*dZ;

        //ds compute RHS
        const Eigen::Vector2d vecRHS( dU*m_vecTLEFT(2)-m_vecTLEFT(0)+dZ1, dV*m_vecTLEFT(2)-m_vecTLEFT(1)+dZ2 );

        Eigen::Matrix2d matA;

        matA << dU*m_matMLEFT(2,0)-m_matMLEFT(0,0), dU*m_matMLEFT(2,1)-m_matMLEFT(0,1), dV*m_matMLEFT(2,0)-m_matMLEFT(1,0), dV*m_matMLEFT(2,1)-m_matMLEFT(1,1);

        //ds compute point in world frame (homogeneous)
        const Eigen::Vector2d vecPointHomo( matA.fullPivHouseholderQr( ).solve( -vecRHS ) );

        //ds reproject point
        Eigen::Vector3d vecPointRIGHT( m_matProjectionRIGHT*Eigen::Vector4d( vecPointHomo(0), vecPointHomo(1), dZ, 1.0 ) );

        //ds normalize
        vecPointRIGHT /= vecPointRIGHT(2);

        //ds get integer coordinates
        const int32_t iU( std::round( vecPointRIGHT(0) ) );
        const int32_t iV( std::round( vecPointRIGHT(1) ) );

        std::printf( "projected point ( %i, %i ) to ( %i, %i )\n", CEpilinearStereoDetector::m_ptMouseClick.x, CEpilinearStereoDetector::m_ptMouseClick.y, iU, iV );
        cv::circle( p_matDisplayUpperTemporary, cv::Point( m_uImageCols+iU, iV ), 1, CColorCode( 0, 0, 255 ), 1 );*/
    }
}

void CNaiveStereoDetector::_detectFeaturesCorner( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight )
{
    //ds input mats
    cv::Mat matLeft;
    cv::Mat matRight;

    //ds normalize monochrome input
    cv::equalizeHist( p_matImageLeft, matLeft );
    cv::equalizeHist( p_matImageRight, matRight );

    //ds detected keypoints
    std::vector< cv::KeyPoint > vecKeyPointsLeft;
    std::vector< cv::KeyPoint > vecKeyPointsRight;

    //ds configuration
    const double dQualityLevel = 0.01;
    const double dMinimumDistance = 20.0;
    const uint32_t uBlockSize = 10;
    const bool bUseHarrisCorners = false;
    const double dK = 0.04;
    const double dMaximumVerticalDistortion = 25.0;

    //ds allocate a features detector
    cv::GoodFeaturesToTrackDetector cDetectorGFTT( m_uFeaturesCap, dQualityLevel, dMinimumDistance, uBlockSize, bUseHarrisCorners, dK );

    //ds detect features and create keypoints
    cDetectorGFTT.detect( matLeft, vecKeyPointsLeft );
    cDetectorGFTT.detect( matRight, vecKeyPointsRight );

    //ds get a descriptor extractor
    cv::OrbDescriptorExtractor cExtractorORB;

    cv::Mat matDescriptorsLeft;
    cv::Mat matDescriptorsRight;

    //ds compute descriptor images
    cExtractorORB.compute( matLeft, vecKeyPointsLeft, matDescriptorsLeft );
    cExtractorORB.compute( matRight, vecKeyPointsRight, matDescriptorsRight );

    //ds get a bruteforce matcher
    cv::BFMatcher cMatcher;
    std::vector< cv::DMatch > vecMatches;
    cMatcher.match( matDescriptorsLeft, matDescriptorsRight, vecMatches );

    std::vector< cv::DMatch > vecGoodMatches;

    //ds filter the matches for vertical skews
    for( cv::DMatch cMatch: vecMatches )
    {
        //ds check if the match is vertically plausible (thanks to stereo vision)
        if( dMaximumVerticalDistortion > std::fabs( vecKeyPointsLeft[cMatch.queryIdx].pt.y - vecKeyPointsRight[cMatch.trainIdx].pt.y ) )
        {
            //ds add the match
            vecGoodMatches.push_back( cMatch );
        }
    }

    //ds if display is desired
    if( m_bDisplayImages )
    {
        //ds get images into triple channel mats (display only)
        cv::Mat matDisplayLeft;
        cv::Mat matDisplayRight;

        //ds get images to triple channel for colored display
        cv::cvtColor( p_matImageLeft, matDisplayLeft, cv::COLOR_GRAY2BGR );
        cv::cvtColor( p_matImageRight, matDisplayRight, cv::COLOR_GRAY2BGR );

        //ds good matches
        std::vector< cv::DMatch > vecMatchesFiltered;

        //ds draw the matches
        for( cv::DMatch cMatch: vecGoodMatches )
        {
            cv::circle( matDisplayLeft, vecKeyPointsLeft[cMatch.queryIdx].pt, 5, cv::Scalar( 0, 255, 0 ), 1 );
            cv::circle( matDisplayRight, vecKeyPointsRight[cMatch.trainIdx].pt, 5, cv::Scalar( 0, 255, 0 ), 1 );
        }

        //ds display mat
        cv::Mat matDisplay = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
        cv::hconcat( matDisplayLeft, matDisplayRight, matDisplay );

        //ds draw matched feature connection
        for( cv::DMatch cMatch: vecGoodMatches )
        {
            cv::line( matDisplay, vecKeyPointsLeft[cMatch.queryIdx].pt, cv::Point2f( m_uImageCols + vecKeyPointsRight[cMatch.trainIdx].pt.x, vecKeyPointsRight[cMatch.trainIdx].pt.y ) , cv::Scalar( 255, 0, 0 ) );
        }

        //ds show the image
        cv::imshow( "stereo matching", matDisplay );

        //ds display time
        cv::waitKey( 1 );
    }

    //ds update references
    m_matReferenceFrameLeft  = p_matImageLeft;
    m_matReferenceFrameRight = p_matImageRight;

    //ds increment count
    ++m_uFrameCount;
}

void CNaiveStereoDetector::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<CEpilinearStereoDetector>(_shutDown) termination requested, detector disabled\n" );
}

void CNaiveStereoDetector::_speedUp( )
{
    ++m_iPlaybackSpeedupCounter;
    m_dFrequencyPlaybackHz += std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;
    std::printf( "<CEpilinearStereoDetector>(_speedUp) increased playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
}

void CNaiveStereoDetector::_slowDown( )
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

void CNaiveStereoDetector::_catchMouseClick( int p_iEventType, int p_iX, int p_iY, int p_iFlags, void* p_hUserdata )
{
    //ds if click is enabled
    if( -1 == CNaiveStereoDetector::m_ptMouseClick.x && -1 == CNaiveStereoDetector::m_ptMouseClick.y )
    {
        if( p_iEventType == cv::EVENT_LBUTTONDOWN )
        {
            //ds set coordinates (corrected)
            CNaiveStereoDetector::m_ptMouseClick.x = p_iX;
            CNaiveStereoDetector::m_ptMouseClick.y = p_iY-2;
        }
        else if( p_iEventType == cv::EVENT_RBUTTONDOWN )
        {
            //ds catch right click
            CNaiveStereoDetector::m_bRightClicked = true;
        }
    }
}
