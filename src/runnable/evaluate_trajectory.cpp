#include <iostream>
#include <fstream>
#include <map>
#include <math.h>

//ds custom
#include "utility/CLogger.h"
#include "configuration/CConfigurationCameraKITTI.h"



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
        return 1;
    }

    //ds log configuration
    CLogger::openBox( );
    std::printf( "(main) strInfileReference := '%s'\n", strInfileReference.c_str( ) );
    std::printf( "(main) strInfileTest      := '%s'\n", strInfileTest.c_str( ) );
    CLogger::closeBox( );

    //ds pose maps (for easy comparison)
    std::map< const UIDFrame, Eigen::Isometry3d > mapPosesReference;
    std::map< const UIDFrame, const Eigen::Isometry3d > mapPosesTest;

    //ds frame counters
    UIDFrame uFrameCountReference = 0;
    std::vector< UIDFrame > vecFrameTest( 0 );

    //ds parse both files
    try
    {
        //ds to our WORLD
        const Eigen::Isometry3d matTransformationLEFTtoWORLDInitial( Eigen::Matrix4d( CConfigurationCameraKITTI::matTransformationIntialLEFTtoWORLD ) );

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
                //mapPosesReference.insert( std::make_pair( uFrameCountReference, matTransformationLEFTtoWORLDInitial*matTransformationLEFTtoWORLD ) );
                mapPosesReference.insert( std::make_pair( uFrameCountReference, matTransformationLEFTtoWORLD.inverse( ) ) );
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

                //ds parse frame id
                UIDFrame uID = 0;
                issLine >> uID;

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
                mapPosesTest.insert( std::make_pair( uID, matTransformationLEFTtoWORLD.inverse( ) ) );
                vecFrameTest.push_back( uID );
            }
        }

        //ds synchronize initial values
        mapPosesTest.insert( std::make_pair( 0, matTransformationLEFTtoWORLDInitial.inverse( )*mapPosesReference[0] ) );

        std::printf( "(main) successfully parsed: %s [entries: %lu]\n", strInfileTest.c_str( ), mapPosesTest.size( ) );

        //std::cout << mapPosesReference[0].matrix( ) << std::endl;
        //std::cout << mapPosesTest[0].matrix( ) << std::endl;

        /*ds buffer first vectors
        const Eigen::Vector3d vecTranslationReference( mapPosesReference[vecFrameTest.front( )].translation( ).normalized( ) );
        const Eigen::Vector3d vecTranslationTest( mapPosesTest[vecFrameTest.front( )].translation( ).normalized( ) );

        //ds compute rotation matrix
        const Eigen::Vector3d vecCross( vecTranslationReference.cross( vecTranslationTest ) );
        const double dSine   = vecCross.norm( );
        const double dCosine = vecTranslationReference.transpose( )*vecTranslationTest;

        //ds skew matrix
        Eigen::Matrix3d matSkew;

        //ds formula: http://en.wikipedia.org/wiki/Cross_product#Skew-symmetric_matrix
        matSkew <<          0.0, -vecCross(2),  vecCross(1),
                    vecCross(2),          0.0, -vecCross(0),
                   -vecCross(1),  vecCross(0),          0.0;

        //ds get transformation matrix
        Eigen::Isometry3d matTransformationReferenceToTest( Eigen::Matrix4d::Identity( ) );
        matTransformationReferenceToTest.linear( ) = Eigen::Matrix3d::Identity( ) + matSkew + ( 1-dCosine )/( dSine*dSine )*matSkew*matSkew;

        //ds adjust reference input to our coordinate frame
        for( std::map< const UIDFrame, Eigen::Isometry3d >::size_type uID = 0; uID < mapPosesReference.size( ); ++uID )
        {
            mapPosesReference[uID] = matTransformationReferenceToTest*mapPosesReference[uID];
        }*/
    }
    catch( const std::exception& p_cException )
    {
        //ds halt on any exception
        std::printf( "(main) ERROR: unable to parse trajectory files, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds test frames
    const UIDFrame uFrameCountTest = vecFrameTest.size( );

    //ds consistency
    if( uFrameCountTest > uFrameCountReference )
    {
        std::printf( "(main) ERROR: test frame count (%lu) has to be lower than reference frame count (%lu)\n", uFrameCountTest, uFrameCountReference );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds stats
    double dTranslationErrorL1MetersTotal   = 0.0;
    double dTranslationTotalMetersReference = 0.0;
    double dTranslationTotalMetersTest      = 0.0;
    double dRotationErrorTotal              = 0.0;
    double dTranslationErrorL1RelativeTotal = 0.0;
    double dRotationErrorRelativeTotal      = 0.0;

    try
    {
        //ds previous ID
        UIDFrame uIDFramePrevious = 0;

        //ds start framewise evaluation
        for( const UIDFrame uIDFrameTest: vecFrameTest )
        {
            //ds compute relative transformations
            const Eigen::Isometry3d matTransformationRelativeReference( mapPosesReference[uIDFrameTest]*mapPosesReference[uIDFramePrevious].inverse( ) );
            const Eigen::Isometry3d matTransformationRelativeTest( mapPosesTest[uIDFrameTest]*mapPosesTest[uIDFramePrevious].inverse( ) );

            //ds ignore z value
            //matTransformationLEFTtoWORLDReference.translation( ).z( ) = 0.0;
            //matTransformationLEFTtoWORLDTest.translation( ).z( ) = 0.0;

            //ds translations
            const double dTranslationL1Meters      = matTransformationRelativeReference.translation( ).norm( );
            const double dTranslationErrorL1Meters = ( matTransformationRelativeReference.translation( )-matTransformationRelativeTest.translation( ) ).norm( );

            //ds mark big errors
            if( 1.0 < dTranslationErrorL1Meters/dTranslationL1Meters )
            {
                std::printf( "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" );
            }

            std::printf( "(main) [%06lu][%06lu] error: %5.2f/%5.2f (%7.2f, %7.2f, %7.2f) > (%7.2f, %7.2f, %7.2f)\n",
                         uIDFramePrevious, uIDFrameTest, dTranslationErrorL1Meters, dTranslationL1Meters,
                         matTransformationRelativeReference.translation( ).x( ), matTransformationRelativeReference.translation( ).y( ), matTransformationRelativeReference.translation( ).z( ),
                         matTransformationRelativeTest.translation( ).x( ), matTransformationRelativeTest.translation( ).y( ), matTransformationRelativeTest.translation( ).z( ) );

            //ds mark big errors
            if( 1.0 < dTranslationErrorL1Meters/dTranslationL1Meters )
            {
                std::printf( "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" );
            }

            //ds totals
            if( 0.0 != dTranslationL1Meters )
            {
                dTranslationErrorL1MetersTotal   += dTranslationErrorL1Meters;
                dTranslationErrorL1RelativeTotal += dTranslationErrorL1Meters/dTranslationL1Meters;
            }

            //ds rotation error
            const double dRotation      = rotationError( matTransformationRelativeReference.linear( ) );
            const double dRotationError = rotationError( matTransformationRelativeReference.linear( )-matTransformationRelativeTest.linear( ) );

            //ds totals
            if( 0.0 != dRotation )
            {
                dRotationErrorTotal         += dRotationError;
                dRotationErrorRelativeTotal += dRotationError/dRotation;
            }

            //ds accumulate total translation
            dTranslationTotalMetersReference += dTranslationL1Meters;
            dTranslationTotalMetersTest      += matTransformationRelativeTest.translation( ).norm( );

            //ds update previous
            uIDFramePrevious = uIDFrameTest;
        }
    }
    catch( const std::exception& p_cException )
    {
        //ds halt on any exception
        std::printf( "(main) ERROR: unable to parse imported trajectory vectors, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    CLogger::openBox( );
    std::printf( "(main) RESULT SUMMARY:\n\n" );
    std::printf( "(main) total trajectory length         = %f m\n", dTranslationTotalMetersReference );
    std::printf( "(main) computed trajectory length      = %f m\n\n", dTranslationTotalMetersTest );
    std::printf( "(main) translation error (L1) total    = %f m\n", dTranslationErrorL1MetersTotal );
    std::printf( "(main) translation error (L1) average  = %f m\n", dTranslationErrorL1MetersTotal/uFrameCountTest );
    std::printf( "(main) translation error (L1) relative = %f \n", dTranslationErrorL1RelativeTotal/uFrameCountTest );
    std::printf( "(main) relative translation precision  = [%f]\n\n", 1.0-dTranslationErrorL1RelativeTotal/uFrameCountTest );
    std::printf( "(main) rotation error total            = %f rad\n", dRotationErrorTotal );
    std::printf( "(main) rotation error average          = %f rad\n", dRotationErrorTotal/uFrameCountTest );
    std::printf( "(main) rotation error relative         = %f\n", dRotationErrorRelativeTotal/uFrameCountTest );
    CLogger::closeBox( );

    //ds done
    ifReference.close( );
    ifTest.close( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
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
