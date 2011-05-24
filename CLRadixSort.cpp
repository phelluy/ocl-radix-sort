// C++ class for sorting integer list in OpenCL
// copyright Philippe Helluy, Université de Strasbourg, France, 2011, helluy@math.unistra.fr
// licensed under the GNU Lesser General Public License see http://www.gnu.org/copyleft/lesser.html
// if you find this software usefull you can cite the following work in your reports or articles:
// Philippe HELLUY, A portable implementation of the radix sort algorithm in OpenCL, HAL 2011.

// members of the class CLRadixSort
// see a description in the hpp...

#include "CLRadixSort.hpp"

using namespace std; 


CLRadixSort::CLRadixSort(cl_context GPUContext,
			 cl_device_id dev,
			 cl_command_queue CommandQue) :
  Context(GPUContext),
  NumDevice(dev),
  CommandQueue(CommandQue),
  nkeys(_N)
{

  nkeys_rounded=nkeys;
  // check some conditions
  assert(_TOTALBITS % _BITS == 0);
  assert(_N % (_GROUPS * _ITEMS) == 0);
  assert( (_GROUPS * _ITEMS * _RADIX) % _HISTOSPLIT == 0);
  assert(pow(2,(int) log2(_GROUPS)) == _GROUPS);
  assert(pow(2,(int) log2(_ITEMS)) == _ITEMS);

  // init the timers
  histo_time=0;
  scan_time=0;
  reorder_time=0;
  transpose_time=0;
  
  //read the program
  string prog;   // program
  string ligne;   // source file line reading
  // kernel sources are in CLRadixsort.cl and we add at the beginning the
  // file CLRadixSortParam.hpp
  ifstream fichierprog("CLRadixSortParam.hpp",ios::in);
  assert(fichierprog && "Le fichier n'existe pas");  
  while(!fichierprog.eof()){
    getline(fichierprog,ligne);
    prog=prog+ligne+"\n";
  }
  fichierprog.close();

  fichierprog.open("CLRadixSort.cl",ios::in);
  assert(fichierprog && "Le fichier n'existe pas"); 
  while(!fichierprog.eof()){
    getline(fichierprog,ligne);
    prog=prog+ligne+"\n";
  }
  fichierprog.close();


  cl_int err;

  Program = clCreateProgramWithSource(Context, 1, (const char **)&prog, NULL, &err);
  if (!Program) {
    printf("Error: Failed to create compute program!\n");
  }

  assert(err == CL_SUCCESS);

  // compilation du code source des kernels

  // avec drapeau
// #ifdef MAC
//     const char *flags = "-DMAC -cl-fast-relaxed-math";
// #else
//     const char *flags = "-cl-fast-relaxed-math";
// #endif
//   err = clBuildProgram(Program, 0, NULL, flags, NULL, NULL);

  // sans drapeau
  err = clBuildProgram(Program, 0, NULL, NULL, NULL, NULL);
  // si la compilation échoue, affichage des erreurs et arrêt
  if (err != CL_SUCCESS) { 
    size_t len;
    char buffer[2048];
    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(Program, NumDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    assert( err == CL_SUCCESS);
  }


  ckHistogram = clCreateKernel(Program, "histogram", &err);
  assert(err == CL_SUCCESS);
  ckScanHistogram = clCreateKernel(Program, "scanhistograms", &err);
  assert(err == CL_SUCCESS);
  ckPasteHistogram = clCreateKernel(Program, "pastehistograms", &err);
  assert(err == CL_SUCCESS);
  ckReorder = clCreateKernel(Program, "reorder", &err);
  assert(err == CL_SUCCESS);
  ckTranspose = clCreateKernel(Program, "transpose", &err);
  assert(err == CL_SUCCESS);
   

  cout << "Construct the random list"<<endl;
  // construction of a random list
  uint maxint=_MAXINT;
  assert(_MAXINT != 0);
  for(uint i = 0; i < _N; i++){
    h_Keys[i] = ((rand())% maxint);
    h_checkKeys[i]=h_Keys[i];
  }

  // construction of the initial permutation
  for(uint i = 0; i < _N; i++){
    h_Permut[i] = i;
  }

  cout << "Send to the GPU"<<endl;
  // copy on the GPU
  d_inKeys  = clCreateBuffer(Context,
			     CL_MEM_READ_WRITE,
			     sizeof(uint)* _N ,
			     NULL,
			     &err);
  assert(err == CL_SUCCESS);

  d_outKeys  = clCreateBuffer(Context,
			      CL_MEM_READ_WRITE,
			      sizeof(uint)* _N ,
			      NULL,
			      &err);
  assert(err == CL_SUCCESS);

  d_inPermut  = clCreateBuffer(Context,
			       CL_MEM_READ_WRITE,
			       sizeof(uint)* _N ,
			       NULL,
			       &err);
  assert(err == CL_SUCCESS);

  d_outPermut  = clCreateBuffer(Context,
				CL_MEM_READ_WRITE,
				sizeof(uint)* _N ,
				NULL,
				&err);
  assert(err == CL_SUCCESS);

  err = clEnqueueWriteBuffer(CommandQueue,
			     d_inKeys,
			     CL_TRUE, 0,
			     sizeof(uint) * _N,
			     h_Keys,
			     0, NULL, NULL);
  assert(err == CL_SUCCESS);

  err = clEnqueueWriteBuffer(CommandQueue,
			     d_inPermut,
			     CL_TRUE, 0,
			     sizeof(uint) * _N,
			     h_Permut,
			     0, NULL, NULL);
  assert(err == CL_SUCCESS);


  // copy on the GPU
  d_Histograms  = clCreateBuffer(Context,
				 CL_MEM_READ_WRITE,
				 sizeof(uint)* _RADIX * _GROUPS * _ITEMS,
				 NULL,
				 &err);
  assert(err == CL_SUCCESS);


  // copy on the GPU
  d_globsum  = clCreateBuffer(Context,
			      CL_MEM_READ_WRITE,
			      sizeof(uint)* _HISTOSPLIT,
			      NULL,
			      &err);
  assert(err == CL_SUCCESS);

  // temporary vector when the sum is not needed
  d_temp  = clCreateBuffer(Context,
			   CL_MEM_READ_WRITE,
			   sizeof(uint)* _HISTOSPLIT,
			   NULL,
			   &err);
  assert(err == CL_SUCCESS);

  Resize(nkeys);


  // we set here the fixed arguments of the OpenCL kernels
  // the changing arguments are modified elsewhere in the class

  err = clSetKernelArg(ckHistogram, 1, sizeof(cl_mem), &d_Histograms);
  assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckHistogram, 3, sizeof(uint)*_RADIX*_ITEMS, NULL);
  assert(err == CL_SUCCESS);

  // err = clSetKernelArg(ckHistogram, 3, sizeof(uint)*_ITEMS, NULL);
  // assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckPasteHistogram, 0, sizeof(cl_mem), &d_Histograms);
  assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckPasteHistogram, 1, sizeof(cl_mem), &d_globsum);
  assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckReorder, 2, sizeof(cl_mem), &d_Histograms);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckReorder, 6,
			sizeof(uint)* _RADIX * _ITEMS ,
			NULL); // mem cache
  assert(err == CL_SUCCESS);


}

// resize the sorted vector
void CLRadixSort::Resize(int nn){

  assert(nn <= _N);

  if (VERBOSE){
    cout << "Resize to  "<<nn<<endl;
  }
  nkeys=nn;

  // length of the vector has to be divisible by (_GROUPS * _ITEMS)
  int reste=nkeys % (_GROUPS * _ITEMS);
  nkeys_rounded=nkeys;
  cl_int err;
  unsigned int pad[_GROUPS * _ITEMS];
  for(int ii=0;ii<_GROUPS * _ITEMS;ii++){
    pad[ii]=_MAXINT-(unsigned int)1;
  }
  if (reste !=0) {
    nkeys_rounded=nkeys-reste+(_GROUPS * _ITEMS);
    // pad the vector with big values
    assert(nkeys_rounded <= _N);
    err = clEnqueueWriteBuffer(CommandQueue,
			       d_inKeys,
			       CL_TRUE, sizeof(uint)*nkeys,
			       sizeof(uint) *(_GROUPS * _ITEMS - reste) ,
			       pad,
			       0, NULL, NULL);
    //cout << nkeys<<" "<<nkeys_rounded<<endl;
    assert(err == CL_SUCCESS);   
  }

}

// transpose the list for faster memory access
void CLRadixSort::Transpose(int nbrow,int nbcol){

// __kernel void transpose(const __global int* invect,
// 			__global int* outvect,
// 			const int nbcol,
// 			const int nbrow,
// 			const __global int* inperm,
// 			__global int* outperm){


  cl_int err;

  err  = clSetKernelArg(ckTranspose, 0, sizeof(cl_mem), &d_inKeys);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckTranspose, 1, sizeof(cl_mem), &d_outKeys);
  assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckTranspose, 2, sizeof(uint), &nbcol);
  assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckTranspose, 3, sizeof(uint), &nbrow);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckTranspose, 4, sizeof(cl_mem), &d_inPermut);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckTranspose, 5, sizeof(cl_mem), &d_outPermut);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckTranspose, 6, sizeof(uint)*_GROUPS*_GROUPS, NULL);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckTranspose, 7, sizeof(uint)*_GROUPS*_GROUPS, NULL);
  assert(err == CL_SUCCESS);

  // err  = clSetKernelArg(ckScanHistogram, 1,
  // 			sizeof(uint)* maxmemcache ,
  // 			NULL); // mem cache


  cl_event eve;

  size_t global_work_size[2];
  size_t local_work_size[2];

  assert(nbrow%_GROUPS == 0);
  assert(nbcol%_GROUPS == 0);

  global_work_size[0]=nbrow/_GROUPS;
  global_work_size[1]=nbcol;

  local_work_size[0]=1;
  local_work_size[1]=_GROUPS;


  err = clEnqueueNDRangeKernel(CommandQueue,
			       ckTranspose,
			       2,   // two dimensions: rows and columns
 			       NULL,
			       global_work_size,
			       local_work_size,
			       0, NULL, &eve);

  //exchange the pointers

  // swap the old and new vectors of keys
  cl_mem d_temp;
  d_temp=d_inKeys;
  d_inKeys=d_outKeys;
  d_outKeys=d_temp;

  // swap the old and new permutations
  d_temp=d_inPermut;
  d_inPermut=d_outPermut;
  d_outPermut=d_temp;


  // timing
  clFinish(CommandQueue);

  cl_ulong debut,fin;

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_QUEUED,
			   sizeof(cl_ulong),
			       (void*) &debut,
			   NULL);
  //cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<endl;
  assert(err== CL_SUCCESS);

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_END,
			   sizeof(cl_ulong),
			       (void*) &fin,
			   NULL);
  assert(err== CL_SUCCESS);

  transpose_time += (float) (fin-debut)/1e9;





}

// global sorting algorithm

void CLRadixSort::Sort(){

  assert(nkeys_rounded <= _N);
  assert(nkeys <= nkeys_rounded);
  int nbcol=nkeys_rounded/(_GROUPS * _ITEMS);
  int nbrow= _GROUPS * _ITEMS;

  if (VERBOSE){
    cout << "Start storting "<<nkeys<< " keys"<<endl;
  }

  if (TRANSPOSE){
    if (VERBOSE) {
      cout << "Transpose"<<endl;
    }
    Transpose(nbrow,nbcol);
  }

  for(uint pass=0;pass<_PASS;pass++){
    //for(uint pass=0;pass<1;pass++){
    if (VERBOSE) {
      cout << "pass "<<pass<<endl;
    }
    //for(uint pass=0;pass<1;pass++){
    if (VERBOSE) {
      cout << "Build histograms "<<endl;
    }
    Histogram(pass);
    if (VERBOSE) {
      cout << "Scan histograms "<<endl;
    }
    ScanHistogram();
    if (VERBOSE) {
      cout << "Reorder "<<endl;
    }
    Reorder(pass);
  }

  if (TRANSPOSE){
    if (VERBOSE) {
      cout << "Transpose back"<<endl;
    }
    Transpose(nbcol,nbrow);
  }

  sort_time=histo_time+scan_time+reorder_time+transpose_time;
  if (VERBOSE){
    cout << "End sorting"<<endl;
  }
}


// check the computation at the end
void CLRadixSort::Check(){
  
  cout << "Get the data from the GPU"<<endl;

  RecupGPU();

  cout << "Test order"<<endl;

  // first see if the final list is ordered
  for(uint i=0;i<nkeys-1;i++){
    if (!(h_Keys[i] <= h_Keys[i+1])) {
      cout <<"erreur tri "<< i<<" "<<h_Keys[i]<<" ,"<<i+1<<" "<<h_Keys[i+1]<<endl;
    }
    assert(h_Keys[i] <= h_Keys[i+1]);
  }

  if (PERMUT) {
    cout << "Check the permutation"<<endl;
    // check if the permutation corresponds to the original list
    for(uint i=0;i<nkeys;i++){
      if (!(h_Keys[i] == h_checkKeys[h_Permut[i]])) {
	cout <<"erreur permut "<< i<<" "<<h_Keys[i]<<" ,"<<i+1<<" "<<h_Keys[i+1]<<endl;
      }
      //assert(h_Keys[i] == h_checkKeys[h_Permut[i]]);
    }
  }

  cout << "test OK !"<<endl;

}

CLRadixSort::~CLRadixSort()
{
  clReleaseKernel(ckHistogram);
  clReleaseKernel(ckScanHistogram);
  clReleaseKernel(ckPasteHistogram);
  clReleaseKernel(ckReorder);
  clReleaseProgram(Program);
  clReleaseMemObject(d_inKeys);
  clReleaseMemObject(d_outKeys);
  clReleaseMemObject(d_Histograms);
  clReleaseMemObject(d_globsum);
  clReleaseMemObject(d_inPermut);
  clReleaseMemObject(d_outPermut);
};

void CLRadixSort::RecupGPU(void){

  cl_int status;

  clFinish(CommandQueue);  // wait end of read

  status = clEnqueueReadBuffer( CommandQueue,
				d_inKeys,
				CL_TRUE, 0, 
				sizeof(uint)  * _N,
				h_Keys,
				0, NULL, NULL ); 
 
  assert (status == CL_SUCCESS);
  clFinish(CommandQueue);  // wait end of read

  status = clEnqueueReadBuffer( CommandQueue,
				d_inPermut,
				CL_TRUE, 0, 
				sizeof(uint)  * _N,
				h_Permut,
				0, NULL, NULL ); 
 
  assert (status == CL_SUCCESS);
  clFinish(CommandQueue);  // wait end of read

  status = clEnqueueReadBuffer( CommandQueue,
				d_Histograms,
				CL_TRUE, 0, 
				sizeof(uint)  * _RADIX * _GROUPS * _ITEMS,
				h_Histograms,
				0, NULL, NULL );  
  assert (status == CL_SUCCESS);

  status = clEnqueueReadBuffer( CommandQueue,
				d_globsum,
				CL_TRUE, 0, 
				sizeof(uint)  * _HISTOSPLIT,
				h_globsum,
				0, NULL, NULL );  
  assert (status == CL_SUCCESS);

  clFinish(CommandQueue);  // wait end of read
}

// display
ostream& operator<<(ostream& os,  CLRadixSort &radi){

  radi.RecupGPU();

  for(uint rad=0;rad<_RADIX;rad++){
    for(uint gr=0;gr<_GROUPS;gr++){
      for(uint it=0;it<_ITEMS;it++){
	os <<"Radix="<<rad<<" Group="<<gr<<" Item="<<it<<" Histo="<<radi.h_Histograms[_GROUPS * _ITEMS * rad +_ITEMS * gr+it]<<endl;
      }
    }
  }
  os<<endl;

  for(uint i=0;i<_HISTOSPLIT;i++){
    os <<"histo "<<i<<" sum="<<radi.h_globsum[i]<<endl;
  }
  os<<endl;

  for(uint i=0;i<radi.nkeys;i++){
    os <<i<<" key="<<radi.h_Keys[i]<<endl;
  }
  os<<endl;

  for(uint i=0;i<radi.nkeys;i++){
    os <<i<<" permut="<<radi.h_Permut[i]<<endl;
  }
  os << endl;

  return os;

}

// compute the histograms
void CLRadixSort::Histogram(uint pass){

  cl_int err;

  size_t nblocitems=_ITEMS;
  size_t nbitems=_GROUPS*_ITEMS;

  assert(_RADIX == pow(2,_BITS));

  err  = clSetKernelArg(ckHistogram, 0, sizeof(cl_mem), &d_inKeys);
  assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckHistogram, 2, sizeof(uint), &pass);
  assert(err == CL_SUCCESS);

  assert( nkeys_rounded%(_GROUPS * _ITEMS) == 0);
  assert( nkeys_rounded <= _N);

  err = clSetKernelArg(ckHistogram, 4, sizeof(uint), &nkeys_rounded);
  assert(err == CL_SUCCESS);

  cl_event eve;

  err = clEnqueueNDRangeKernel(CommandQueue,
			       ckHistogram,
			       1, NULL,
			       &nbitems,
			       &nblocitems,
			       0, NULL, &eve);

  //cout << err<<" , "<<CL_OUT_OF_RESOURCES<<endl;
  assert(err== CL_SUCCESS);

  clFinish(CommandQueue);

  cl_ulong debut,fin;

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_QUEUED,
			   sizeof(cl_ulong),
			       (void*) &debut,
			   NULL);
  //cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<endl;
  assert(err== CL_SUCCESS);

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_END,
			   sizeof(cl_ulong),
			       (void*) &fin,
			   NULL);
  assert(err== CL_SUCCESS);

  histo_time += (float) (fin-debut)/1e9;


}

// scan the histograms
void CLRadixSort::ScanHistogram(void){

  cl_int err;

  // numbers of processors for the local scan
  // half the size of the local histograms
  size_t nbitems=_RADIX* _GROUPS*_ITEMS / 2;


  size_t nblocitems= nbitems/_HISTOSPLIT ;


  int maxmemcache=max(_HISTOSPLIT,_ITEMS * _GROUPS * _RADIX / _HISTOSPLIT);

  // scan locally the histogram (the histogram is split into several
  // parts that fit into the local memory)

  err = clSetKernelArg(ckScanHistogram, 0, sizeof(cl_mem), &d_Histograms);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckScanHistogram, 1,
			sizeof(uint)* maxmemcache ,
			NULL); // mem cache

  err = clSetKernelArg(ckScanHistogram, 2, sizeof(cl_mem), &d_globsum);
  assert(err == CL_SUCCESS);

  cl_event eve;

  err = clEnqueueNDRangeKernel(CommandQueue,
			       ckScanHistogram,
			       1, NULL,
			       &nbitems,
			       &nblocitems,
			       0, NULL, &eve);

  // cout << err<<","<< CL_INVALID_WORK_ITEM_SIZE<< " "<<nbitems<<" "<<nblocitems<<endl;
  // cout <<CL_DEVICE_MAX_WORK_ITEM_SIZES<<endl;
  assert(err== CL_SUCCESS);
  clFinish(CommandQueue); 

  cl_ulong debut,fin;

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_QUEUED,
			   sizeof(cl_ulong),
			       (void*) &debut,
			   NULL);
  //cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<endl;
  assert(err== CL_SUCCESS);

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_END,
			   sizeof(cl_ulong),
			       (void*) &fin,
			   NULL);
  assert(err== CL_SUCCESS);

  scan_time += (float) (fin-debut)/1e9;

  // second scan for the globsum
  err = clSetKernelArg(ckScanHistogram, 0, sizeof(cl_mem), &d_globsum);
  assert(err == CL_SUCCESS);

  // err  = clSetKernelArg(ckScanHistogram, 1,
  // 			sizeof(uint)* _HISTOSPLIT,
  // 			NULL); // mem cache

  err = clSetKernelArg(ckScanHistogram, 2, sizeof(cl_mem), &d_temp);
  assert(err == CL_SUCCESS);

  nbitems= _HISTOSPLIT / 2;
  nblocitems=nbitems;

  err = clEnqueueNDRangeKernel(CommandQueue,
  			       ckScanHistogram,
  			       1, NULL,
  			       &nbitems,
  			       &nblocitems,
  			       0, NULL, &eve);

  assert(err== CL_SUCCESS);
  clFinish(CommandQueue); 

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_QUEUED,
			   sizeof(cl_ulong),
			       (void*) &debut,
			   NULL);
  //cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<endl;
  assert(err== CL_SUCCESS);

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_END,
			   sizeof(cl_ulong),
			       (void*) &fin,
			   NULL);
  assert(err== CL_SUCCESS);

  //  cout <<"durée global scan ="<<(float) (fin-debut)/1e9<<" s"<<endl;
  scan_time += (float) (fin-debut)/1e9;


  // loops again in order to paste together the local histograms
  nbitems = _RADIX* _GROUPS*_ITEMS/2;
  nblocitems=nbitems/_HISTOSPLIT;

  err = clEnqueueNDRangeKernel(CommandQueue,
  			       ckPasteHistogram,
  			       1, NULL,
  			       &nbitems,
  			       &nblocitems,
  			       0, NULL, &eve);

  assert(err== CL_SUCCESS);
  clFinish(CommandQueue);  

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_QUEUED,
			   sizeof(cl_ulong),
			       (void*) &debut,
			   NULL);
  //cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<endl;
  assert(err== CL_SUCCESS);

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_END,
			   sizeof(cl_ulong),
			       (void*) &fin,
			   NULL);
  assert(err== CL_SUCCESS);

  //  cout <<"durée paste ="<<(float) (fin-debut)/1e9<<" s"<<endl;

  scan_time += (float) (fin-debut)/1e9;


}

// reorder the data from the scanned histogram
void CLRadixSort::Reorder(uint pass){


  cl_int err;

  size_t nblocitems=_ITEMS;
  size_t nbitems=_GROUPS*_ITEMS;


  clFinish(CommandQueue);

  err  = clSetKernelArg(ckReorder, 0, sizeof(cl_mem), &d_inKeys);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckReorder, 1, sizeof(cl_mem), &d_outKeys);
  assert(err == CL_SUCCESS);

  err = clSetKernelArg(ckReorder, 3, sizeof(uint), &pass);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckReorder, 4, sizeof(cl_mem), &d_inPermut);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckReorder, 5, sizeof(cl_mem), &d_outPermut);
  assert(err == CL_SUCCESS);

  err  = clSetKernelArg(ckReorder, 6,
			sizeof(uint)* _RADIX * _ITEMS ,
			NULL); // mem cache
  assert(err == CL_SUCCESS);

  assert( nkeys_rounded%(_GROUPS * _ITEMS) == 0);

  err = clSetKernelArg(ckReorder, 7, sizeof(uint), &nkeys_rounded);
  assert(err == CL_SUCCESS);


  assert(_RADIX == pow(2,_BITS));

  cl_event eve;

  err = clEnqueueNDRangeKernel(CommandQueue,
			       ckReorder,
			       1, NULL,
			       &nbitems,
			       &nblocitems,
			       0, NULL, &eve);
  
  //cout << err<<" , "<<CL_MEM_OBJECT_ALLOCATION_FAILURE<<endl;

  assert(err== CL_SUCCESS);
  clFinish(CommandQueue);  

  cl_ulong debut,fin;

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_QUEUED,
			   sizeof(cl_ulong),
			       (void*) &debut,
			   NULL);
  //cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<endl;
  assert(err== CL_SUCCESS);

  err=clGetEventProfilingInfo (eve,
			   CL_PROFILING_COMMAND_END,
			   sizeof(cl_ulong),
			       (void*) &fin,
			   NULL);
  assert(err== CL_SUCCESS);

  //cout <<"durée="<<(float) (fin-debut)/1e9<<" s"<<endl;
  reorder_time += (float) (fin-debut)/1e9;



  // swap the old and new vectors of keys
  cl_mem d_temp;
  d_temp=d_inKeys;
  d_inKeys=d_outKeys;
  d_outKeys=d_temp;

  // swap the old and new permutations
  d_temp=d_inPermut;
  d_inPermut=d_outPermut;
  d_outPermut=d_temp;

}

