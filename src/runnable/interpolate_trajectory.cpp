#include <iostream>
#include <fstream>

//ds custom
#include "utility/CLogger.h"

int32_t main( int32_t argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds arguments: ground_truth vi_mapper_trajectory_file
    if( 19 != argc )
    {
        std::printf( "(main) please specify the argument <trajectory_vi_mapper> <final_frame_count> <initial rotation>\n" );
        std::printf( "(main) please specify the argument <trajectory_vi_mapper> <final_frame_count> 1 0 0 0 0 0 -1 0 0 1 0 0 0 0 0 1\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds transformation to kitti
    Eigen::Isometry3d matTransformationToKITTI( Eigen::Matrix4d::Identity( ) );
    for( uint8_t a = 0; a < 4; ++a )
    {
        for( uint8_t b = 0; b < 4; ++b )
        {
            matTransformationToKITTI(a,b) = atof( argv[3+4*a+b] );
        }
    }

    //ds files
    const std::string strInfileTest = argv[1];
    const std::vector< Eigen::Matrix4d >::size_type uFinalFrameCount = std::atoi( argv[2] );

    //ds outfile name
    const std::string strOutfileTest = strInfileTest+".interpolated.txt";

    //ds open files
    std::ifstream ifTest( strInfileTest, std::ifstream::in );

    //ds on failure
    if( !ifTest.good( ) )
    {
        std::printf( "(main) unable to open trajectory files\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds log configuration
    CLogger::openBox( );
    std::printf( "(main) strInfileTest  := '%s'\n", strInfileTest.c_str( ) );
    std::printf( "(main) strOutfileTest := '%s'\n", strOutfileTest.c_str( ) );
    std::printf( "(main) set transformation matrix: \n\n" );
    std::cout << matTransformationToKITTI.matrix( ) << std::endl;
    CLogger::closeBox( );
    getchar( );

    //ds interpolated vector
    std::vector< Eigen::Isometry3d > vecPosesInterpolated( 0 );
    vecPosesInterpolated.push_back( Eigen::Isometry3d( Eigen::Matrix4d::Identity( ) ) );

    //ds parse both files
    try
    {
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

                //ds parse current frame id
                UIDFrame uIDCurrent = 0;
                issLine >> uIDCurrent;

                //ds information fields (KITTI format)
                Eigen::Matrix4d matTransformationRAW( Eigen::Matrix4d::Identity( ) );
                for( uint8_t u = 0; u < 3; ++u )
                {
                    for( uint8_t v = 0; v < 4; ++v )
                    {
                        issLine >> matTransformationRAW(u,v);
                    }
                }

                //ds compute for kitti system
                const Eigen::Isometry3d matTransformationNOWtoWORLD( matTransformationToKITTI*matTransformationRAW );
                const Eigen::Quaterniond vecQuaternionNOWtoWORLD( matTransformationNOWtoWORLD.linear( ) );

                //ds previous matrix
                const Eigen::Isometry3d matTransformationPREVIOUStoWORLD( vecPosesInterpolated.back( ) );
                const Eigen::Quaterniond vecQuaternionPREVIOUStoWORLD( matTransformationPREVIOUStoWORLD.linear( ) );

                //ds previous id
                const UIDFrame uIDPrevious = vecPosesInterpolated.size( )-1;

                //ds compute last to now delta
                const Eigen::Vector3d vecTranslation = matTransformationNOWtoWORLD.translation( )-matTransformationPREVIOUStoWORLD.translation( );

                //ds frame delta
                const UIDFrame uFrameDelta = uIDCurrent-uIDPrevious;

                //ds interpolate (trajectory only)
                for( UIDFrame u = 1; u < uFrameDelta; ++u )
                {
                    //ds portion
                    const double dPortion = static_cast< double >( u )/uFrameDelta;

                    std::printf( "(main) creating frame ID: %lu (factor: %3.1f)\n", uIDPrevious+u, dPortion );

                    Eigen::Isometry3d matTransformationCURRENTtoWORLD( matTransformationPREVIOUStoWORLD );

                    //ds compute translation
                    matTransformationCURRENTtoWORLD.translation( ) += dPortion*vecTranslation;

                    //ds compute interpolated quaternion
                    Eigen::Quaterniond vecQuaternionCURRENTtoWORLD = vecQuaternionPREVIOUStoWORLD.slerp( dPortion, vecQuaternionNOWtoWORLD );

                    //ds set it
                    matTransformationCURRENTtoWORLD.linear( ) = Eigen::Matrix3d( vecQuaternionCURRENTtoWORLD );

                    vecPosesInterpolated.push_back( matTransformationCURRENTtoWORLD );
                }

                //ds synchronization
                assert( uIDCurrent == static_cast< UIDFrame >( vecPosesInterpolated.size( ) ) );

                std::printf( "(main) setting reference frame ID: %lu\n", vecPosesInterpolated.size( ) );
                vecPosesInterpolated.push_back( matTransformationNOWtoWORLD );
            }
        }
    }
    catch( const std::exception& p_cException )
    {
        //ds halt on any exception
        std::printf( "(main) ERROR: unable to parse trajectory files, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds close it
    ifTest.close( );

    //ds check if limit not reached yet
    if( uFinalFrameCount > vecPosesInterpolated.size( ) )
    {
        //ds matrices
        const Eigen::Isometry3d matTransformationNOWtoWORLD( vecPosesInterpolated.back( ) );
        const Eigen::Quaterniond vecQuaternionNOWtoWORLD( matTransformationNOWtoWORLD.linear( ) );
        const Eigen::Isometry3d matTransformationPREVIOUStoWORLD( vecPosesInterpolated[vecPosesInterpolated.size( )-2] );
        const Eigen::Quaterniond vecQuaternionPREVIOUStoWORLD( matTransformationPREVIOUStoWORLD.linear( ) );

        //ds previous id
        const UIDFrame uIDPrevious = vecPosesInterpolated.size( )-1;

        //ds compute last to now delta
        const Eigen::Vector3d vecTranslation = matTransformationNOWtoWORLD.translation( )-matTransformationPREVIOUStoWORLD.translation( );

        //ds frame delta
        const UIDFrame uFrameDelta = uFinalFrameCount-uIDPrevious;

        //ds interpolate (trajectory only)
        for( UIDFrame u = 1; u < uFrameDelta; ++u )
        {
            //ds portion
            const double dPortion = static_cast< double >( u )/uFrameDelta;

            std::printf( "(main) creating frame ID: %lu (factor: %3.1f)\n", uIDPrevious+u, dPortion );

            Eigen::Isometry3d matTransformationCURRENTtoWORLD( matTransformationNOWtoWORLD );

            //ds compute translation
            matTransformationCURRENTtoWORLD.translation( ) += dPortion*vecTranslation;

            //ds compute interpolated quaternion
            Eigen::Quaterniond vecQuaternionCURRENTtoWORLD = vecQuaternionPREVIOUStoWORLD.slerp( dPortion, vecQuaternionNOWtoWORLD );

            //ds set it
            matTransformationCURRENTtoWORLD.linear( ) = Eigen::Matrix3d( vecQuaternionCURRENTtoWORLD );

            vecPosesInterpolated.push_back( matTransformationCURRENTtoWORLD );
        }
    }

    assert( uFinalFrameCount == vecPosesInterpolated.size( ) );

    //ds write transforms to file
    std::ofstream ofTest( strOutfileTest, std::ifstream::out );
    for( const Eigen::Isometry3d matTranformationLEFTtoWORLD: vecPosesInterpolated )
    {
        for( uint8_t u = 0; u < 3; ++u )
        {
            for( uint8_t v = 0; v < 4; ++v )
            {
                ofTest << matTranformationLEFTtoWORLD(u,v) << " ";
            }
        }
        ofTest << "\n";
    }
    ofTest.close( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    return 0;
}
