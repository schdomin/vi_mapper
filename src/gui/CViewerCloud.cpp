#include "CViewerCloud.h"

#include <QKeyEvent>

CViewerCloud::CViewerCloud( const std::shared_ptr< const std::vector< CMatchCloud > > p_vecMatches,
                            const Eigen::Isometry3d& p_matTransformationQUERYtoREFERENCEInitial,
                            const Eigen::Isometry3d& p_matTransformationQUERYtoREFERENCEFinal )
{
    m_vecMatches                               = p_vecMatches;
    m_matTransformationQUERYtoREFERENCEInitial = p_matTransformationQUERYtoREFERENCEInitial;
    m_matTransformationQUERYtoREFERENCEFinal   = p_matTransformationQUERYtoREFERENCEFinal;

    setWindowTitle( "CViewerCloud" );
    showMaximized( );
}
CViewerCloud::~CViewerCloud( )
{

}

void CViewerCloud::draw( )
{
    glPushAttrib( GL_ENABLE_BIT );
    glDisable( GL_LIGHTING );
    glPointSize( 5.0 );

    //ds draw WORLD coordinate frame
    glColor3f( 1.0, 1.0, 1.0 );
    glLineWidth( 1.0 );
    drawAxis( 0.5 );

    //ds set line width and point size for landmarks
    glLineWidth( 1.0 );
    glPointSize( 2.5 );

    //ds draw reference cloud in blue
    if( 0 != m_vecMatches )
    {
        for( const CMatchCloud& cPoint: *m_vecMatches )
        {
            //ds reference camera point
            const CPoint3DCAMERA vecREFERENCE( cPoint.cPointReference.vecPointXYZCAMERA );
            const CPoint3DCAMERA vecQUERYInitial( m_matTransformationQUERYtoREFERENCEInitial*cPoint.cPointQuery.vecPointXYZCAMERA );

            //ds draw the line
            glColor3f( 0.5, 0.5, 0.5 );
            glBegin( GL_LINES );
            glVertex3f( vecREFERENCE.x( ), vecREFERENCE.y( ), vecREFERENCE.z( ) );
            glVertex3f( vecQUERYInitial.x( ), vecQUERYInitial.y( ), vecQUERYInitial.z( ) );
            glEnd( );

            glBegin( GL_POINTS );
            glColor3f( 0.25, 0.25, 1.0 );
            glVertex3f( vecREFERENCE.x( ), vecREFERENCE.y( ), vecREFERENCE.z( ) );
            glColor3f( 1.0, 0.0, 0.0 );
            glVertex3f( vecQUERYInitial.x( ), vecQUERYInitial.y( ), vecQUERYInitial.z( ) );
            glEnd( );

            if( m_bDrawOptimizedPoints )
            {
                const CPoint3DCAMERA vecQUERYFinal( m_matTransformationQUERYtoREFERENCEFinal*cPoint.cPointQuery.vecPointXYZCAMERA );

                //ds draw the line
                glColor3f( 1.0, 0.0, 0.0 );
                glBegin( GL_LINES );
                glVertex3f( vecQUERYInitial.x( ), vecQUERYInitial.y( ), vecQUERYInitial.z( ) );
                glVertex3f( vecQUERYFinal.x( ), vecQUERYFinal.y( ), vecQUERYFinal.z( ) );
                glEnd( );

                glBegin( GL_POINTS );
                glColor3f( 0.0, 1.0, 0.0 );
                glVertex3f( vecQUERYFinal.x( ), vecQUERYFinal.y( ), vecQUERYFinal.z( ) );
                glEnd( );
            }
        }
    }

    glPopAttrib( ); //GL_ENABLE_BIT
}

//virtual void keyPressEvent( QKeyEvent* p_pEvent );
void CViewerCloud::init( )
{
    setSceneRadius( 25.0 );
}

void CViewerCloud::keyPressEvent( QKeyEvent* p_pEvent )
{
    //ds evaluate key pressed
    switch( p_pEvent->key( ) )
    {
        case Qt::Key_Space:
        {
            //ds switch drawing mode
            if( m_bDrawOptimizedPoints )
            {
                m_bDrawOptimizedPoints = false;
            }
            else
            {
                m_bDrawOptimizedPoints = true;
            }
            draw( );
            updateGL( );
            break;
        }

        default:
        {
            QGLViewer::keyPressEvent( p_pEvent );
        }
    }
}

QString CViewerCloud::helpString( ) const
{
    return "TODO";
}
