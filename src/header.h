#include <time.h>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"


/* ----------------------------------------------------------------
 *  * Cellular Automata Constant Variables
 *   * ----------------------------------------------------------------*/
static const float EPSINON = 0.000001;
static const float ERROR_VALUE = -9999;
static const int   ERROR_COORD = -9999;
static const float DISPERSION = 5.0;

/*
 * static const int _classWater = 1;
 * static const int _classUrban = 2;
 * static const int _classBarren = 3;
 * static const int _classForest = 4;
 * static const int _classShrubland = 5;
 * static const int _classWoody = 6;
 * static const int _classHerbaceous = 7;
 * static const int _classCultivated = 8;
 * static const int _classWetland = 9;
 * */

static const float _coeffConst = 6.4640;
static const float _coeffElev = 43.5404;
static const float _coeffSlope = 1.9150;
static const float _coeffDist2CityCtr = 41.3441;
static const float _coeffDist2Trnsprt = 12.5878;
/*
 float _aLandUseCoeffs[9] = {0.0,     // water
                                         0.0,     // urban
                                         -9.8655, // barren
                                         -8.7469, // forest
                                         -9.2688, // shrubland
                                         -8.0321, // woody
                                         -9.1693, // herbaceous
                                         -8.9420, // cultivated
                                         -9.4500  // wetland
                                        };
*/
/* ----------------------------------------------------------------
 *  * Random() returns a pseudo-random real number uniformly
 *   * distributed between 0.0 and 1.0.
 *    * ----------------------------------------------------------------*/
#define MODULUS    2147483647 /* DON'T CHANGE THIS VALUE                  */
#define MULTIPLIER 48271      /* DON'T CHANGE THIS VALUE                  */
#define CHECK      399268537  /* DON'T CHANGE THIS VALUE                  */
#define STREAMS    256        /* # of streams, DON'T CHANGE THIS VALUE    */
#define A256       22925      /* jump multiplier, DON'T CHANGE THIS VALUE */
#define DEFAULT    123456789  /* initial seed, use 0 < DEFAULT < MODULUS  */
#define BLOCK_SIZE 32
//static long seed[STREAMS] = {DEFAULT};  /* current state of each stream   */
static int  stream        = 0;          /* stream index, 0 is the default */

