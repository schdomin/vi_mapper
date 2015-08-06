#ifndef CEXCEPTIONINVALIDFILE_H
#define CEXCEPTIONINVALIDFILE_H

#include <exception>

class CExceptionInvalidFile: public std::exception
{

public:

    CExceptionInvalidFile( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionInvalidFile( )
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

#endif //CEXCEPTIONINVALIDFILE_H
