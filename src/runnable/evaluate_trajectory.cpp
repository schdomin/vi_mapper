#include <iostream>
#include <fstream>
#include <map>
#include <math.h>

//ds custom
#include "utility/CLogger.h"


//ds KITTI snippet
inline double rotationError( const Eigen::Matrix3d& p_matRotationDelta );

int32_t main( int32_t argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds arguments: ground_truth vi_mapper_trajectory_file
    if( 3 != argc )
    {
        std::printf( "(main) please specify the arguments <trajectory_ground_truth> <trajectory_vi_mapper>\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds files
    std::string strInfileReference = argv[1];
    std::string strInfileTest      = argv[2];

    //ds open files
    std::ifstream ifReference( strInfileReference, std::ifstream::in );
    std::ifstream ifTest( strInfileTest, std::ifstream::in );

    //ds on failure
    if( !ifReference.good( ) || !ifTest.good( ) )
    {
        std::printf( "(main) unable to open trajectory files\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds log configuration
    CLogger::openBox( );
    std::printf( "(main) strInfileReference := '%s'\n", strInfileReference.c_str( ) );
    std::printf( "(main) strInfileTest      := '%s'\n", strInfileTest.c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds pose maps (for easy comparison)
    std::map< const UIDFrame, const Eigen::Isometry3d > mapPosesReference;
    std::map< const UIDFrame, const Eigen::Isometry3d > mapPosesTest;

    //ds frame counters
    UIDFrame uFrameCountReference = 0;
    UIDFrame uFrameCountTest      = 0;

    //ds parse both files
    try
    {
        //ds reference first
        while( ifReference.good( ) )
        {
            //ds line buffer
            std::string strLineBuffer;

            //ds read one line
            std::getline( ifReference, strLineBuffer );

            //ds if we read something
            if( !strLineBuffer.empty( ) )
            {
                //ds get it to a stringstream
                std::istringstream issLine( strLineBuffer );

                //ds information fields (KITTI format)
                Eigen::Matrix4d matTransformationRAW( Eigen::Matrix4d::Identity( ) );
                for( uint8_t u = 0; u < 3; ++u )
                {
                    for( uint8_t v = 0; v < 4; ++v )
                    {
                        issLine >> matTransformationRAW(u,v);
                    }
                }

                //ds set transformation matrix
                const Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationRAW );

                //ds save to map
                mapPosesReference.insert( std::make_pair( uFrameCountReference, matTransformationLEFTtoWORLD ) );
                ++uFrameCountReference;
            }
        }

        std::printf( "(main) successfully parsed: %s [entries: %lu]\n", strInfileReference.c_str( ), mapPosesReference.size( ) );

        //ds then test
        while( ifTest.good( ) )
        {
            //ds line buffer
            std::string strLineBuffer;

            //ds read one line
            std::getline( ifTest, strLineBuffer );

            //ds if we read something
            if( !strLineBuffer.empty( ) )
            {
                //ds get it to a stringstream
                std::istringstream issLine( strLineBuffer );

                //ds information fields (KITTI format)
                Eigen::Matrix4d matTransformationRAW( Eigen::Matrix4d::Identity( ) );
                for( uint8_t u = 0; u < 3; ++u )
                {
                    for( uint8_t v = 0; v < 4; ++v )
                    {
                        issLine >> matTransformationRAW(u,v);
                    }
                }

                //ds set transformation matrix
                const Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationRAW );

                //ds save to map
                mapPosesTest.insert( std::make_pair( uFrameCountTest, matTransformationLEFTtoWORLD ) );
                ++uFrameCountTest;
            }
        }

        std::printf( "(main) successfully parsed: %s [entries: %lu]\n", strInfileTest.c_str( ), mapPosesTest.size( ) );
    }
    catch( const std::exception& p_cException )
    {
        //ds halt on any exception
        std::printf( "(main) ERROR: unable to parse trajectory files, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds consistency
    if( uFrameCountTest > uFrameCountReference )
    {
        std::printf( "(main) ERROR: test frame count (%lu) has to be lower than reference frame count (%lu)\n", uFrameCountTest, uFrameCountReference );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds stats
    double dTranslationErrorL1MetersTotal   = 0.0;
    double dTranslationTotalMeters          = 0.0;
    double dRotationErrorTotal              = 0.0;
    double dTranslationErrorL1RelativeTotal = 0.0;
    double dRotationErrorRelativeTotal      = 0.0;

    Eigen::Isometry3d matTransformationLEFTtoWORLDPrevious( Eigen::Matrix4d::Identity( ) );

    //ds start framewise evaluation
    for( UIDFrame uFrameCheck = 0; uFrameCheck < uFrameCountTest; ++uFrameCheck )
    {
        //ds fetch both transformations
        const Eigen::Isometry3d matTransformationLEFTtoWORLDReference( mapPosesReference.at( uFrameCheck ) );
        const Eigen::Isometry3d matTransformationLEFTtoWORLDTest( mapPosesTest.at( uFrameCheck ) );

        //ds translations
        const double dTranslationL1Meters      = ( matTransformationLEFTtoWORLDReference.translation( )-matTransformationLEFTtoWORLDPrevious.translation( ) ).norm( );
        const double dTranslationErrorL1Meters = ( matTransformationLEFTtoWORLDReference.translation( )-matTransformationLEFTtoWORLDTest.translation( ) ).norm( );

        //ds totals
        if( 0.0 != dTranslationL1Meters )
        {
            dTranslationErrorL1MetersTotal   += dTranslationErrorL1Meters;
            dTranslationErrorL1RelativeTotal += dTranslationErrorL1Meters/dTranslationL1Meters;
        }

        //ds rotation error
        const double dRotation      = rotationError( matTransformationLEFTtoWORLDReference.linear( )-matTransformationLEFTtoWORLDPrevious.linear( ) );
        const double dRotationError = rotationError( matTransformationLEFTtoWORLDReference.linear( )-matTransformationLEFTtoWORLDTest.linear( ) );

        //ds totals
        if( 0.0 != dRotation )
        {
            dRotationErrorTotal         += dRotationError;
            dRotationErrorRelativeTotal += dRotationError/dRotation;
        }

        //ds accumulate total translation
        dTranslationTotalMeters += dTranslationL1Meters;
        matTransformationLEFTtoWORLDPrevious = matTransformationLEFTtoWORLDReference;
    }

    CLogger::openBox( );
    std::printf( "(main) total trajectory length         = %f\n", dTranslationTotalMeters );
    std::printf( "(main) translation error (L1) total    = %f\n", dTranslationErrorL1MetersTotal );
    std::printf( "(main) translation error (L1) average  = %f\n", dTranslationErrorL1MetersTotal/uFrameCountTest );
    std::printf( "(main) translation error (L1) relative = %f\n", dTranslationErrorL1RelativeTotal/uFrameCountTest );
    std::printf( "(main) rotation error total            = %f\n", dRotationErrorTotal );
    std::printf( "(main) rotation error average          = %f\n", dRotationErrorTotal/uFrameCountTest );
    std::printf( "(main) rotation error relative         = %f\n", dRotationErrorRelativeTotal/uFrameCountTest );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds done
    ifReference.close( );
    ifTest.close( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

inline double rotationError( const Eigen::Matrix3d& p_matRotationDelta )
{
    double a = p_matRotationDelta(0,0);
    double b = p_matRotationDelta(1,1);
    double c = p_matRotationDelta(2,2);

    //ds if not perfect case
    if( 0 != a && 0 != b && 0 != c )
    {
        double d = 0.5*( a + b + c - 1.0 );
        return acos( std::max( std::min( d, 1.0 ), -1.0 ) );
    }
    else
    {
        return 0.0;
    }
}
