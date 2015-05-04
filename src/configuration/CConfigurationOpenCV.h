#ifndef CCONFIGURATIONOPENCV_H_
#define CCONFIGURATIONOPENCV_H_

#include <opencv/cv.h>

class CConfigurationOpenCV
{

//ds fields
public:

    //ds keystrokes
    static struct KeyStroke
    {
        constexpr static int iEscape      = 537919515; //1074790427; //537919515;
        constexpr static int iNumpadPlus  = 537984939;
        constexpr static int iNumpadMinus = 537984941;
        constexpr static int iSpace       = 537919520; //1074790432; //537919520;
    } KeyStroke;

};

#endif //#define CCONFIGURATIONOPENCV_H_
