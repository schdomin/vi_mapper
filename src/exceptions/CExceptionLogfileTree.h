#ifndef CEXCEPTIONLOGFILETREE_H
#define CEXCEPTIONLOGFILETREE_H

#include <exception>

class CExceptionLogfileTree: public std::exception
{

public:

    CExceptionLogfileTree( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionLogfileTree( )
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

#endif //CEXCEPTIONLOGFILETREE_H
