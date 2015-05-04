#ifndef CWRAPPEROPENCV_H_
#define CWRAPPEROPENCV_H_

#include <Eigen/Core>
#include <opencv/cv.h>

class CWrapperOpenCV
{

//ds methods
public:

    template< typename tType, uint32_t uRows, uint32_t uCols > static const Eigen::Matrix< tType, uRows, uCols > fromCVMatrix( const cv::Mat& p_matCV )
    {
        //ds allocate eigen matrix
        Eigen::Matrix< tType, uRows, uCols > matEigen;

        //ds fill the matrix (column major)
        for( uint32_t u = 0; u < uRows; ++u )
        {
            for( uint32_t v = 0; v < uCols; ++v )
            {
                matEigen( u, v ) = p_matCV.at< tType >( u, v );
            }
        }

        return matEigen;
    }

    template< typename tType, uint32_t uRows > static const Eigen::Matrix< tType, uRows, 1 > fromCVVector( const cv::Vec< tType, uRows > p_vecCV )
    {
        //ds allocate eigen matrix
        Eigen::Matrix< tType, uRows, 1 > vecEigen;

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < uRows; ++u )
        {
            vecEigen( u ) = p_vecCV( u );
        }

        return vecEigen;
    }

    template< typename tType, uint32_t uRows > static const cv::Vec< tType, uRows > toCVVector( const Eigen::Matrix< tType, uRows, 1 > p_vecEigen )
    {
        //ds allocate cv vector
        cv::Vec< tType, uRows > vecCV;

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < uRows; ++u )
        {
            vecCV( u ) = p_vecEigen( u );
        }

        return vecCV;
    }

};

#endif //#define CWRAPPEROPENCV_H_
