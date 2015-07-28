#ifndef CVIEWERSCENE_H
#define CVIEWERSCENE_H

#include <QGLViewer/qglviewer.h>
#include <vector>
#include <map>

//ds custom
#include "types/Typedefs.h"

class CViewerScene: public QGLViewer
{

    //ds landmark type
    struct CLandmarkInScene
    {
        CLandmarkInScene( const UIDLandmark& p_uID,
                          const CPoint3DInWorldFrame& p_vecPositionXYZOriginal,
                          const CPoint3DInWorldFrame& p_vecPositionXYZOptimized ): uID( p_uID ),
                                                                             vecPositionXYZOriginal( p_vecPositionXYZOriginal.x( ), p_vecPositionXYZOriginal.y( ), p_vecPositionXYZOriginal.z( ) ),
                                                                             vecPositionXYZOptimized( p_vecPositionXYZOptimized.x( ), p_vecPositionXYZOptimized.y( ), p_vecPositionXYZOptimized.z( ) )
        {
            //ds nothing to do
        }

        const UIDLandmark uID;
        const qglviewer::Vec vecPositionXYZOriginal;
        qglviewer::Vec vecPositionXYZOptimized;
    };

protected:

    virtual void draw( );
    //virtual void keyPressEvent( QKeyEvent* p_pEvent );
    virtual void init( );
    virtual QString helpString( ) const;

public:

    void addKeyFrame( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< const CMeasurementLandmark* > > p_pLandmarks )
    {
        //ds position
        const CPoint3DInWorldFrame vecPosition( p_matTransformationLEFTtoWORLD.translation( ) );
        const Eigen::Quaterniond vecQuaternion( p_matTransformationLEFTtoWORLD.linear( ) );

        //ds setup the new frame
        qglviewer::Frame cFrameNew;
        cFrameNew.setPosition( vecPosition.x( ), vecPosition.y( ), vecPosition.z( ) );
        cFrameNew.setOrientation( vecQuaternion.x( ), vecQuaternion.y( ), vecQuaternion.z( ), vecQuaternion.w( ) );

        //ds add it to the vector
        m_vecKeyFrames.push_back( cFrameNew );

        //ds add/update the landmarks
        for( const CMeasurementLandmark* pLandmark: *p_pLandmarks )
        {
            try
            {
                //ds check if landmark is present
                CLandmarkInScene cLandmarkInScene( m_mapLandmarks.at( pLandmark->uID ) );

                //ds update optimized position
                cLandmarkInScene.vecPositionXYZOptimized = qglviewer::Vec( pLandmark->vecPointXYZWORLDOptimized.x( ), pLandmark->vecPointXYZWORLDOptimized.y( ), pLandmark->vecPointXYZWORLDOptimized.z( ) );
            }
            catch( const std::out_of_range& p_cException )
            {
                //ds add the landmark
                m_mapLandmarks.insert( std::pair< UIDLandmark, CLandmarkInScene >( pLandmark->uID, CLandmarkInScene( pLandmark->uID, pLandmark->vecPointXYZWORLD, pLandmark->vecPointXYZWORLDOptimized ) ) );
            }
        }

        //ds force redraw
        draw( );
    }

private:

    std::vector< qglviewer::Frame > m_vecKeyFrames;
    std::map< UIDLandmark, CLandmarkInScene > m_mapLandmarks;

};

#endif //CVIEWERSCENE_H
