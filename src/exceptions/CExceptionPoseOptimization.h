#ifndef CEXCEPTIONPOSEOPTIMIZATION_H_
#define CEXCEPTIONPOSEOPTIMIZATION_H_

#include <exception>

class CExceptionPoseOptimization: public std::exception
{

public:

    CExceptionPoseOptimization( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionPoseOptimization( )
    {

    }

private:

    const std::string m_strExceptionDescription;

public:

    virtual const char* what( ) const throw( )
    {
        return m_strExceptionDescription.c_str( );
    }
};

#endif //CEXCEPTIONPOSEOPTIMIZATION_H_
