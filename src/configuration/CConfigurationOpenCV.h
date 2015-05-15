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
        constexpr static int iEscape      = 1048603; //1074790427; //537919515;
        constexpr static int iNumpadPlus  = 1114027; //1074855851; //537984939;
        constexpr static int iNumpadMinus = 1114029; //1074855853; //537984941;
        constexpr static int iSpace       = 1048608; //1074790432; //537919520;
        constexpr static int iBackspace   = 1113864;
    } KeyStroke;

};

#endif //#define CCONFIGURATIONOPENCV_H_
