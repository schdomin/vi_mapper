#ifndef CEXCEPTIONNOMATCHFOUND_H_
#define CEXCEPTIONNOMATCHFOUND_H_

#include <exception>

class CExceptionNoMatchFound: public std::exception
{

public:

    CExceptionNoMatchFound( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionNoMatchFound( )
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

#endif //CEXCEPTIONNOMATCHFOUND_H_
