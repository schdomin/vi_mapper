#ifndef CEXCEPTIONDETECTIONFAILED_H_
#define CEXCEPTIONDETECTIONFAILED_H_

#include <exception>

class CExceptionDetectionFailed: public std::exception
{

public:

    CExceptionDetectionFailed( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionDetectionFailed( )
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

#endif //CEXCEPTIONDETECTIONFAILED_H_
