// OpenCL kernel sources for the CLRadixSort class
// the precompilateur does not exist in OpenCL
// thus we simulate the #include "CLRadixSortParam.hpp" by
// string manipulations


// change of index for the transposition
int index(int i,int n){
  int ip;
  if (TRANSPOSE) {
    int k,l;
    k= i/(n/_GROUPS/_ITEMS);
    l = i%(n/_GROUPS/_ITEMS);
    ip = l * (_GROUPS*_ITEMS) + k;
  }
  else {
    ip=i;
  }
  return ip;
}



// compute the histogram for each radix and each virtual processor for the pass
__kernel void histogram(const __global int* d_Keys,
			__global int* d_Histograms,
			const int pass,
			__local int* loc_histo,
			const int n){


  int it = get_local_id(0);
  int ig = get_global_id(0);

  int gr = get_group_id(0);

  int groups=get_num_groups(0);
  int items=get_local_size(0);

  // set the local histograms to zero
  for(int ir=0;ir<_RADIX;ir++){
    //d_Histograms[ir * groups * items + items * gr + it]=0;
    loc_histo[ir * items + it] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);  


  // range of keys that are analyzed by the work item
  //int start= gr * n/groups + it * n/groups/items;
  int start= ig *(n/groups/items);
  int size= n/groups/items;

  int key,shortkey;

  for(int i= start; i< start + size;i++){
    key=d_Keys[index(i,n)];   

    // extract the group of _BITS bits of the pass
    // the result is in the range 0.._RADIX-1
    shortkey=(( key >> (pass * _BITS)) & (_RADIX-1));  

    //d_Histograms[shortkey * groups * items + items * gr + it]++;
    loc_histo[shortkey *  items + it ]++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);  

  // copy the local histogram to the global one
  for(int ir=0;ir<_RADIX;ir++){
    d_Histograms[ir * groups * items + items * gr + it]=loc_histo[ir * items + it];
  }
  
  barrier(CLK_GLOBAL_MEM_FENCE);  


}

__kernel void transpose(const __global int* invect,
			__global int* outvect,
			const int nbcol,
			const int nbrow,
			const __global int* inperm,
			__global int* outperm,
			__local int* blockmat,
			__local int* blockperm,
			const int tilesize){
  
  int i0 = get_global_id(0)*tilesize;  // first row index
  int j = get_global_id(1);  // column index

  int iloc = 0;  // first local row index
  int jloc = get_local_id(1);  // local column index

  // fill the cache
  for(iloc=0;iloc<tilesize;iloc++){
    int k=(i0+iloc)*nbcol+j;  // position in the matrix
    blockmat[iloc*tilesize+jloc]=invect[k];
    if (PERMUT) {
      blockperm[iloc*tilesize+jloc]=inperm[k];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);  

  // first row index in the transpose
  int j0=get_group_id(1)*tilesize;

  // put the cache at the good place
  // loop on the rows
  for(iloc=0;iloc<tilesize;iloc++){
    int kt=(j0+iloc)*nbrow+i0+jloc;  // position in the transpose
    outvect[kt]=blockmat[jloc*tilesize+iloc];
    if (PERMUT) {
      outperm[kt]=blockperm[jloc*tilesize+iloc];
    }
  }
 
}

// each virtual processor reorders its data using the scanned histogram
__kernel void reorder(const __global int* d_inKeys,
		      __global int* d_outKeys,
		      __global int* d_Histograms,
		      const int pass,
		      __global int* d_inPermut,
		      __global int* d_outPermut,
		      __local int* loc_histo,
		      const int n){

  int it = get_local_id(0);
  int ig = get_global_id(0);

  int gr = get_group_id(0);
  int groups=get_num_groups(0);
  int items=get_local_size(0);

  //int start= gr * n/groups + it * n/groups/items;
  int start= ig *(n/groups/items);
  int size= n/groups/items;

  // take the histograms in the cache
  for(int ir=0;ir<_RADIX;ir++){
    loc_histo[ir * items + it]=
      d_Histograms[ir * groups * items + items * gr + it];
  }
  barrier(CLK_LOCAL_MEM_FENCE);  


  int newpos,ik,key,shortkey;

  for(int i= start; i< start + size;i++){
    key = d_inKeys[index(i,n)];   
    shortkey=((key >> (pass * _BITS)) & (_RADIX-1)); 
    //ik= shortkey * groups * items + items * gr + it;
    //newpos=d_Histograms[ik];
    newpos=loc_histo[shortkey * items + it];
    d_outKeys[index(newpos,n)]= key;  // killing line !!!
    //d_outKeys[index(i)]= key;  
    if(PERMUT) {
      d_outPermut[index(newpos,n)]=d_inPermut[index(i,n)]; 
    }
    newpos++;
    loc_histo[shortkey * items + it]=newpos;
    //d_Histograms[ik]=newpos;
    //barrier(CLK_GLOBAL_MEM_FENCE);  

  }
  

}


// perform a parallel prefix sum (a scan) on the local histograms
// (see Blelloch 1990) each workitem worries about two memories
// see also http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__kernel void scanhistograms( __global int* histo,__local int* temp,__global int* globsum){


  int it = get_local_id(0);
  int ig = get_global_id(0);
  int decale = 1; 
  int n=get_local_size(0) * 2 ;
  int gr=get_group_id(0);

  // load input into local memory
  // up sweep phase
  temp[2*it] = histo[2*ig];  
  temp[2*it+1] = histo[2*ig+1];  
 	
  // parallel prefix sum (algorithm of Blelloch 1990) 
  for (int d = n>>1; d > 0; d >>= 1){   
    barrier(CLK_LOCAL_MEM_FENCE);  
    if (it < d){  
      int ai = decale*(2*it+1)-1;  
      int bi = decale*(2*it+2)-1;  	
      temp[bi] += temp[ai];  
    }  
    decale *= 2; 
  }
  
  // store the last element in the global sum vector
  // (maybe used in the next step for constructing the global scan)
  // clear the last element
  if (it == 0) {
    globsum[gr]=temp[n-1];
    temp[n - 1] = 0;
  }
                 
  // down sweep phase
  for (int d = 1; d < n; d *= 2){  
    decale >>= 1;  
    barrier(CLK_LOCAL_MEM_FENCE);

    if (it < d){  
      int ai = decale*(2*it+1)-1;  
      int bi = decale*(2*it+2)-1;  
         
      int t = temp[ai];  
      temp[ai] = temp[bi];  
      temp[bi] += t;   
    }  

  }  
  barrier(CLK_LOCAL_MEM_FENCE);

  // write results to device memory

  histo[2*ig] = temp[2*it];  
  histo[2*ig+1] = temp[2*it+1];  

  barrier(CLK_GLOBAL_MEM_FENCE);

}  

// use the global sum for updating the local histograms
// each work item updates two values
__kernel void pastehistograms( __global int* histo,__global int* globsum){


  int ig = get_global_id(0);
  int gr=get_group_id(0);

  int s;

  s=globsum[gr];
  
  // write results to device memory
  histo[2*ig] += s;  
  histo[2*ig+1] += s;  

  barrier(CLK_GLOBAL_MEM_FENCE);

}  



