#include "CViewerScene.h"

#include <QKeyEvent>
#include <QGLViewer/manipulatedFrame.h>
#include </usr/include/GL/freeglut_std.h>

void CViewerScene::draw()
{
    //ds draw WORLD coordinate frame
    glColor3f( 1.0, 1.0, 1.0 );
    glLineWidth( 1.0 );
    drawAxis( 0.5 );

    //ds previous frame (always at origin)
    qglviewer::Frame cFramePrevious;

    //ds draw all keyframes
    for( qglviewer::Frame cFrame: m_vecKeyFrames )
    {
        //ds positions
        const qglviewer::Vec vecPositionPrevious( cFramePrevious.position( ) );
        const qglviewer::Vec vecPositionNow( cFrame.position( ) );

        //ds draw the line between the previous and current keyframe
        glColor3f( 0.25, 0.25, 1.0 );
        glLineWidth( 2.5 );
        glPushAttrib( GL_ENABLE_BIT );
        glDisable( GL_LIGHTING );
        glBegin( GL_LINES );
        glVertex3f( vecPositionPrevious.x, vecPositionPrevious.y, vecPositionPrevious.z );
        glVertex3f( vecPositionNow.x, vecPositionNow.y, vecPositionNow.z );
        glEnd( );
        glPopAttrib( );

        //ds orientation
        glColor3f( 1.0, 1.0, 1.0 );
        glLineWidth( 1.0 );
        glPushMatrix( );
        glMultMatrixd( cFrame.matrix( ) );
        drawAxis( 0.1 );
        glPopMatrix( );

        //ds update previous
        cFramePrevious = cFrame;
    }

    //ds draw all landmarks
    for( const std::pair< UIDLandmark, CLandmarkInScene >& cLandmarkInScene: m_mapLandmarks )
    {
        glColor3f( 0.5, 0.5, 0.5 );
        glLineWidth( 1.0 );
        glPushMatrix( );
        glTranslatef( cLandmarkInScene.second.vecPositionXYZOptimized.x, cLandmarkInScene.second.vecPositionXYZOptimized.y, cLandmarkInScene.second.vecPositionXYZOptimized.z );
        glutSolidSphere( 0.05, 8, 8 );
        glPopMatrix( );
        glFlush( );
    }
}

void CViewerScene::init( )
{
    //ds initialize
    m_vecKeyFrames.clear( );
    m_mapLandmarks.clear( );
    restoreStateFromFile( );
    setSceneRadius( 25.0 );

    //ds UGLY HACK
    char fakeParam[] = "fake";
    char *fakeargv[] = { fakeParam, NULL };
    int fakeargc = 1;
    glutInit( &fakeargc, fakeargv );
}

QString CViewerScene::helpString( ) const
{
  QString text("<h2>HUBBA BABA</h2>");
  text += "A <i>KeyFrameInterpolator</i> holds an interpolated path defined by key frames. ";
  text += "It can then smoothly make its associed frame follow that path. Key frames can interactively be manipulated, even ";
  text += "during interpolation.<br><br>";
  text += "Note that the camera holds 12 such keyFrameInterpolators, binded to F1-F12. Press <b>Alt+Fx</b> to define new key ";
  text += "frames, and then press <b>Fx</b> to make the camera follow the path. Press <b>C</b> to visualize these paths.<br><br>";
  text += "<b>+/-</b> changes the interpolation speed. Negative values are allowed.<br><br>";
  text += "<b>Return</b> starts-stops the interpolation.<br><br>";
  text += "Use the left and right arrows to change the manipulated KeyFrame. ";
  text += "Press <b>Control</b> to move it or simply hover over it.";
  return text;
}

/*void CViewerScene::keyPressEvent( QKeyEvent* p_pEvent )
{
    switch( p_pEvent->key( ) )
    {
        default:
        {
            QGLViewer::close( );
            break;
        }
    }
}*/
