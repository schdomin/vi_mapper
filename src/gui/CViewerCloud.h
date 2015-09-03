#ifndef CVIEWERCLOUD_H
#define CVIEWERCLOUD_H

#include <QGLViewer/qglviewer.h>

//ds custom
#include "types/TypesCloud.h"

class CViewerCloud: public QGLViewer
{

public:

    CViewerCloud( const std::shared_ptr< const std::vector< CMatchCloud > > p_vecMatches,
                  const Eigen::Isometry3d& p_matTransformationQUERYtoREFERENCEInitial,
                  const Eigen::Isometry3d& p_matTransformationQUERYtoREFERENCEFinal );
    ~CViewerCloud( );

protected:

    virtual void draw( );
    virtual void init( );
    virtual void keyPressEvent( QKeyEvent* p_pEvent );
    virtual QString helpString( ) const;

private:

    std::shared_ptr< const std::vector< CMatchCloud > > m_vecMatches = 0;
    Eigen::Isometry3d m_matTransformationQUERYtoREFERENCEInitial;
    Eigen::Isometry3d m_matTransformationQUERYtoREFERENCEFinal;
    bool m_bDrawOptimizedPoints = false;

};

#endif //CVIEWERCLOUD_H
