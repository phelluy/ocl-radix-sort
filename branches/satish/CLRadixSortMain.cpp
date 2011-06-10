// C++ class for sorting integer list in OpenCL
// copyright Philippe Helluy, Université de Strasbourg, France, 2011, helluy@math.unistra.fr
// licensed under the GNU Lesser General Public License see http://www.gnu.org/copyleft/lesser.html
// if you find this software usefull you can cite the following work in your reports or articles:
// Philippe HELLUY, A portable implementation of the radix sort algorithm in OpenCL, HAL 2011.
#include "CLRadixSort.hpp"

#include<iostream>
#include<fstream>
#include<assert.h>
#include <algorithm>
#include <vector>
#include <time.h>


using namespace std; 


int main(void){

  // OpenCL init

  cl_device_id* Devices;   // devoces array 

  char exten[1000]; // list of extensions to opencl language

  cl_uint NbPlatforms;
  cl_context Context;
  cl_uint NbDevices;   // number of devices of the gpu
  cl_uint NumDevice;   // device id
  cl_uint NumPlatform; // platform id
  cl_device_type DeviceType;   
  cl_command_queue CommandQueue; 


  cl_int status;

  cout <<endl<< "Test CLRadixSort class..."<<endl<<endl;

  // lecture du nombre de plateformes open cl
  status = clGetPlatformIDs(0, NULL, &NbPlatforms);
  assert (status == CL_SUCCESS);

  assert(NbPlatforms > 0);

  // allocation du tableaux des plateformes
  cl_platform_id* Platforms = new cl_platform_id[NbPlatforms];

  // remplissage
  status = clGetPlatformIDs(NbPlatforms, Platforms, NULL);
  assert (status == CL_SUCCESS);

  // affichage
  cout << "Available platforms:"<<endl;



  char pbuf[2000];
  for (int i = 0; i < (int)NbPlatforms; ++i) { 
    cout <<"Platform "<<i+1<<":"<<endl;
  // affichages divers
    status = clGetPlatformInfo(Platforms[i],
			       CL_PLATFORM_NAME,
			       sizeof(pbuf),
			       pbuf,
			       NULL);
    assert (status == CL_SUCCESS);

    cout << pbuf <<endl;

    status = clGetPlatformInfo(Platforms[i],
			       CL_PLATFORM_VENDOR,
			       sizeof(pbuf),
			       pbuf,
			       NULL);
    assert (status == CL_SUCCESS);

    cout << pbuf <<endl;

  // affichage version opencl
    status = clGetPlatformInfo(Platforms[i],
			       CL_PLATFORM_VERSION,
			       sizeof(pbuf),
			       pbuf,
			       NULL);
    assert (status == CL_SUCCESS);

    cout << pbuf <<endl;

  }

  NumPlatform=NbPlatforms-1;
  NumPlatform=0;
  assert(NumPlatform < NbPlatforms);


  // comptage du nombre de devices
  status = clGetDeviceIDs(Platforms[NumPlatform],
			  CL_DEVICE_TYPE_ALL,
			  0,
			  NULL,
			  &NbDevices);
  assert (status == CL_SUCCESS);
  assert(NbDevices > 0);
  //cout << NbDevices << endl;

  // allocation du tableau des devices
  Devices = new cl_device_id[NbDevices];

  // choix du numéro de device
  // en général, le premier est le gpu
  // (mais pas toujours)
  NumDevice=0;
  assert(NumDevice < NbDevices);


  cout << "We choose Device "<<NumDevice+1<<"/"<<NbDevices<<
    " of Platform "<<NumPlatform+1<<"/"<<NbPlatforms<<":"<<endl;
  

  // remplissage du tableau des devices
  status = clGetDeviceIDs(Platforms[NumPlatform],
			  CL_DEVICE_TYPE_ALL,
			  NbDevices,
			  Devices,
			  NULL);
  assert (status == CL_SUCCESS);

  assert(NbDevices > NumDevice);
 
  // informations diverses

  // type du device
  status = clGetDeviceInfo(
			   Devices[NumDevice],
			   CL_DEVICE_TYPE,
			   sizeof(cl_device_type),
			   (void*)&DeviceType,
			   NULL);
  assert (status == CL_SUCCESS);

  // device name
  status = clGetDeviceInfo(
			   Devices[NumDevice],
			   CL_DEVICE_NAME,
			   sizeof(exten),
			   pbuf,
			   NULL);
  assert (status == CL_SUCCESS);
  
  cout << pbuf<<endl;

  cout << "OpenCL Extensions for that device: ";
  status = clGetDeviceInfo(
			   Devices[NumDevice],
			   CL_DEVICE_EXTENSIONS,
			   sizeof(exten),
			   exten,
			   NULL);
  assert (status == CL_SUCCESS);
  cout << exten<<endl<<endl;



  // type du device
  status = clGetDeviceInfo(
			   Devices[NumDevice],
			   CL_DEVICE_TYPE,
			   sizeof(cl_device_type),
			   (void*)&DeviceType,
			   NULL);
  assert (status == CL_SUCCESS);

  if (DeviceType == CL_DEVICE_TYPE_CPU){
    cout << "Sort on CPU"<<endl;
  }
  else{
    cout << "Sort on GPU"<<endl;
  }

  // compute units number
  cl_int cores;
  status = clGetDeviceInfo(
			   Devices[NumDevice],
			   CL_DEVICE_MAX_COMPUTE_UNITS,
			   sizeof(cl_int),
			   (void*)&cores,
			   NULL);
  assert (status == CL_SUCCESS);

  cout << "Compute units="<<cores<<endl;

  // context opencl
  cout <<"Create the context"<<endl;
  Context = clCreateContext(0,
			    1,
			    &Devices[NumDevice],
			    NULL, 
			    NULL,
			    &status);

  assert (status == CL_SUCCESS);
  


  // create the commandqueue
  cout <<"Create the command queue"<<endl;
  CommandQueue = clCreateCommandQueue(
				      Context,
				      Devices[NumDevice],
				      CL_QUEUE_PROFILING_ENABLE,
				      &status);
  assert (status == CL_SUCCESS);

  cout <<"OpenCL initializations OK !"<<endl<<endl;

  // end of the minimal opencl declarations

  // declaration of a CLRadixSort object
  static CLRadixSort rs(Context,Devices[NumDevice],CommandQueue);

  cout << "Radix="<<_RADIX<<endl;
  cout << "Max Int="<<(uint) _MAXINT <<endl;
  cout << "Local Memory="<<rs.LocalMemSize<<" Bytes"<<endl;
  cout << "Max work-group size="<<rs.MaxWorkGroupSize<<endl;
    
  // sort

  // test a non power of two size list
  //rs.Resize((1 << 20) -1);
  //rs.Resize(64);


  // cout << rs;

  // cout <<"transpose"<<endl;
  // rs.Transpose();

  // cout << rs;

  // assert(1==2);

  cout << "sorting "<< rs.nkeys <<" keys"<<endl<<endl;

  //rs.Sort();

  // rs.SortBlocks(0);
  // rs.RecupGPU();
  // //cout << rs;
  // cout << rs.histo_time;
  // assert(1==2);
  // rs.ScanSatish();
  // rs.ReorderSatish(0);

#ifdef SATISH
  rs.SortSatish();
#else
  rs.Sort();
#endif

  rs.RecupGPU();


  cout << rs.histo_time<<" s in the histograms"<<endl;
  cout << rs.scan_time<<" s in the scanning"<<endl;
  cout << rs.reorder_time<<" s in the reordering"<<endl;
  cout << rs.transpose_time<<" s in the transposition"<<endl;

  cout << rs.sort_time <<" s total GPU time (without memory transfers)"<<endl;

  // display the data (for debugging)
  if (VERBOSE) {
    //cout << rs;
  }

  // check the results (debugging)
  rs.Check();



  // sort with the standard c++ sort algorithm
  cout << "Cpu sorting..."<<endl;
  clock_t init, final;
  double tcpu;
  init=clock();
  sort(rs.h_checkKeys,rs.h_checkKeys+rs.nkeys);
  final=clock()-init;
  tcpu=(double)final / ((double)CLOCKS_PER_SEC);
  cout << tcpu <<" s"<<endl;

  cout <<"speedup="<<tcpu/rs.sort_time<<endl;


  // pic sorting test
  // cout << "PIC sorting test"<<endl;
  // rs.PICSorting();
  

  return 0;


}
