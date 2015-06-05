#ifndef CEXCEPTIONNOMATCHFOUNDINTERNAL_H
#define CEXCEPTIONNOMATCHFOUNDINTERNAL_H

#include <exception>

class CExceptionNoMatchFoundInternal: public std::exception
{

public:

    CExceptionNoMatchFoundInternal( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionNoMatchFoundInternal( )
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

#endif //CEXCEPTIONNOMATCHFOUNDINTERNAL_H
