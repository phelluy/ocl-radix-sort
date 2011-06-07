// C++ class for sorting integer list in OpenCL
// copyright Philippe Helluy, Université de Strasbourg, France, 2011, helluy@math.unistra.fr
// licensed under the GNU Lesser General Public License see http://www.gnu.org/copyleft/lesser.html
// if you find this software usefull you can cite the following work in your reports or articles:
// Philippe HELLUY, A portable implementation of the radix sort algorithm in OpenCL, 2011.
// http://hal.archives-ouvertes.fr/hal-00596730
// global parameters for the CLRadixSort class
// they are included in the class AND in the OpenCL kernels
///////////////////////////////////////////////////////
// these parameters can be changed
#define _ITEMS  64 // number of items in a group
#define _GROUPS 32 // the number of virtual processors is _ITEMS * _GROUPS
#define  _HISTOSPLIT 512 // number of splits of the histogram
#define _TOTALBITS 28  // number of bits for the integer in the list (max=32)
#define _BITS 4  // number of bits in the radix
#define _SMALLBITS 1  //  number of bits in the small radix (=1 for a split algorithm)
#define _BLOCKSIZE 512  // size of the sorted blocks in the Satish algorithm
// max size of the sorted vector
// it has to be divisible by  _ITEMS * _GROUPS
// (for other sizes, pad the list with big values)
//#define _N (_ITEMS * _GROUPS * 16)  
#define _N (1<<23)  // maximal size of the list  
#define VERBOSE 1
#define TRANSPOSE  // transpose the initial vector (faster memory access)
//#define PERMUT  // store the final permutation
////////////////////////////////////////////////////////
#define SATISH

// the following parameters are computed from the previous
#define _RADIX (1 << _BITS) //  radix  = 2^_BITS
#define _SMALLRADIX (1 << _SMALLBITS) // small radix for the local sort (generally =2)
#define _PASS (_TOTALBITS/_BITS) // number of needed passes to sort the list
#define _HISTOSIZE (_ITEMS * _GROUPS * _RADIX ) // size of the histogram
// maximal value of integers for the sort to be correct
#define _MAXINT (1 << _TOTALBITS)

