// C++ class for sorting integer list in OpenCL
// copyright Philippe Helluy, Université de Strasbourg, France, 2011, helluy@math.unistra.fr
// licensed under the GNU Lesser General Public License see http://www.gnu.org/copyleft/lesser.html
// if you find this software usefull you can cite the following work in your reports or articles:
// Philippe HELLUY, A portable implementation of the radix sort algorithm in OpenCL, HAL 2011.
// The algorithm is the radix sort algorithm
// Marcho Zagha and Guy E. Blelloch. “Radix Sort For Vector Multiprocessor.”
// in: Conference on High Performance Networking and Computing, pp. 712-721, 1991.
// each integer is made of _TOTALBITS bits. The radix is made of _BITS bits. The sort is made of
// several passes, each consisting in sorting against a group of bits corresponding to the radix.
// _TOTALBITS/_BITS passes are needed.
// The sorting parameters can be changed in "CLRadixSortParam.hpp"
// compilation for Mac:
//g++ CLRadixSort.cpp CLRadixSortMain.cpp -framework opencl -Wall
// compilation for Linux:
//g++ CLRadixSort.cpp CLRadixSortMain.cpp -lOpenCL -Wall

#ifndef _CLRADIXSORT
#define _CLRADIXSORT

#include "CLRadixSortParam.hpp"


#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif 

typedef cl_uint uint;


#include <string>
#include<fstream>
#include<iostream>
#include<assert.h>
#include<math.h>
#include <stdlib.h>

using namespace std;


class CLRadixSort{


friend ostream &operator<<(ostream &os, CLRadixSort &r);

public:
  CLRadixSort(cl_context Context,
	      cl_device_id NumDevice,
	      cl_command_queue CommandQueue);
  
  CLRadixSort() {};
  ~CLRadixSort();
  

  // this function allows to change the size of the sorted vector
  void Resize(int nn);

  // this function treats the array d_Keys on the GPU
  // and return the sorting permutation in the array d_Permut
  void Sort();

  // get the data from the GPU (for debugging)
  void RecupGPU(void);

  // put the data on the host in the GPU
  void Host2GPU(void);

  // check that the sort is successfull (for debugging)
  void Check(void);

  // sort a set of particles (for debugging)
  void PICSorting(void);

  // transpose the list for faster memeory access
  // (improve coalescence)
  void Transpose(int nbrow,int nbcol);

  // compute the histograms for one pass
  void Histogram(uint pass);
  // scan the histograms
  void ScanHistogram(void);
  // scan the histograms
  void Reorder(uint pass);


  cl_context Context;             // OpenCL context
  cl_device_id NumDevice;         // OpenCL Device
  cl_command_queue CommandQueue;     // OpenCL command queue 
  cl_program Program;                // OpenCL program
  uint h_Histograms[_RADIX * _GROUPS * _ITEMS]; // histograms on the cpu
  cl_mem d_Histograms;                   // histograms on the GPU

  // sum of the local histograms
  uint h_globsum[_HISTOSPLIT];
  cl_mem d_globsum;
  cl_mem d_temp;  // in case where the sum is not needed

  // list of keys
  uint nkeys; // actual number of keys
  uint nkeys_rounded; // next multiple of _ITEMS*_GROUPS
  uint h_checkKeys[_N]; // a copy for check
  uint h_Keys[_N];
  cl_mem d_inKeys;
  cl_mem d_outKeys;

  // permutation
  uint h_Permut[_N];
  cl_mem d_inPermut;
  cl_mem d_outPermut;

   // OpenCL kernels
  cl_kernel ckTranspose; // transpose the initial list
  cl_kernel ckHistogram;  // compute histograms
  cl_kernel ckScanHistogram; // scan local histogram
  cl_kernel ckPasteHistogram; // paste local histograms
  cl_kernel ckReorder; // final reordering

  // timers
  float histo_time,scan_time,reorder_time,sort_time,transpose_time;

};


float corput(int n,int k1,int k2);

#endif
