#include"header.h"
#define NCOLS 746

__device__ long int coord2idx1(int i, int j, int nRows, int nCols) {
  long int idx = ERROR_COORD;
  if(i >= 0 && i < nRows && j >= 0 && j < nCols) {
    idx = i * nCols + j;
  }
  return idx;
}


__device__ float computeGlbProb(int i, int j,
                     int nRows, int nCols,
                     short *pLandUse,
                     float *pElev,
                     float *pSlope,
                     float *pDist2CityCtr,
                     float *pDist2Trnsprt,float *_aLandUseCoeffs,int nRows_global) {
  float glbProb = ERROR_VALUE;
  short landuse;
  float z, elev, slope, dist2city, dist2trnsprt;
 
  long int cellIdx1 = coord2idx1(i, j, nRows_global, nCols);
  long int cellIdx = coord2idx1(i, j, nRows, nCols);
  if(cellIdx1 != ERROR_COORD) {
    landuse = pLandUse[cellIdx];
    if(landuse <= 2) {
      glbProb = 0.0;
    }
    else {
      z = _coeffConst;

      elev = pElev[cellIdx];
      z += _coeffElev * elev;

      slope = pSlope[cellIdx];
      z += _coeffSlope * slope;

      dist2city = pDist2CityCtr[cellIdx];
      z += _coeffDist2CityCtr * dist2city;

      dist2trnsprt = pDist2Trnsprt[cellIdx];
      z += _coeffDist2Trnsprt * dist2trnsprt;

      z += _aLandUseCoeffs[landuse-1];

      glbProb = 1.0/(1.0 + exp(z));
    }
  }
  
  return glbProb; 
}
__global__ void Kernel_computerglbprob( short *GPUplanduse,float *GPUpelev,float *GPUpslope,
                                        float *GPUpdist2cityctr,float *GPUpdist2trnsprt,
                                        float *GPUpglbprob,int nRows, int nCols,
                                        float *GPU_aLandUseCoeffs,int nRows_global){

  int i= blockIdx.x* BLOCK_SIZE + threadIdx.x;
  int j= blockIdx.y* BLOCK_SIZE + threadIdx.y;
  float glbProb=0;

  if(( i < nRows-1)&&(j < nCols-1)){
    if((i>0)&&(j>0)){
      glbProb = computeGlbProb(i, j, nRows, nCols,GPUplanduse, GPUpelev, GPUpslope, GPUpdist2cityctr, GPUpdist2trnsprt,GPU_aLandUseCoeffs,nRows_global);
         
      if(fabs(glbProb - ERROR_VALUE) > EPSINON) {
        GPUpglbprob[coord2idx1(i, j, nRows, nCols)] = glbProb;
      }
    }    
  }
}

void COMPUTEglbprob(int myrank,int npes,int nRows,int nCols,short *pLandUse,float *pElev, float *pSlope,float *pDist2CityCtr,float *pDist2Trnsprt,float *pGlbProb,int ROW_GRID,int rowExtra,int rowStep,int nRows_global){
  long long int short_size=sizeof(short)*nRows*nCols;
  long long int float_size=sizeof(float)*nRows*nCols;
  float _aLandUseCoeffs[9] = {0.0, 0.0,-9.8655,-8.7469,-9.2688, -8.0321, -9.1693, -8.9420,-9.4500 };
  short *GPUplanduse=NULL;
  float *GPUpelev=NULL;
  float *GPUpslope=NULL;
  float *GPUpdist2cityctr=NULL;
  float *GPUpdist2Trnsprt=NULL;
  float *GPUpglbprob=NULL;
  float *GPU_aLandUseCoeffs=NULL;   

  cudaMalloc((void**) &GPUplanduse, short_size);
  cudaMalloc((void**) &GPUpelev, float_size);
  cudaMalloc((void**) &GPUpslope, float_size);
  cudaMalloc((void**) &GPUpdist2cityctr, float_size);
  cudaMalloc((void**) &GPUpdist2Trnsprt, float_size);
  cudaMalloc((void**) &GPUpglbprob, float_size);
  cudaMalloc((void**) &GPU_aLandUseCoeffs,sizeof(float)*9);

  cudaMemcpy(GPUplanduse,pLandUse,short_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpelev,pElev,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpslope, pSlope,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpdist2cityctr,pDist2CityCtr,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpdist2Trnsprt,pDist2Trnsprt,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpglbprob,pGlbProb,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_aLandUseCoeffs,_aLandUseCoeffs,sizeof(float)*9,cudaMemcpyHostToDevice);

  dim3 dimBlock,dimGrid;
  dimBlock.x=BLOCK_SIZE;
  dimBlock.y=BLOCK_SIZE;
  dimBlock.z=1;
        
  int GRID_SIZE= (nRows*nCols)/(BLOCK_SIZE*BLOCK_SIZE)+((nRows*nCols)%(BLOCK_SIZE*BLOCK_SIZE))?1:0;
  dimGrid.x=ROW_GRID;
  dimGrid.y=NCOLS;   
  dimGrid.z=1;
  
  Kernel_computerglbprob<<<dimGrid,dimBlock>>>(GPUplanduse,GPUpelev,GPUpslope,GPUpdist2cityctr,GPUpdist2Trnsprt,GPUpglbprob,nRows, nCols,GPU_aLandUseCoeffs,nRows_global);
  cudaMemcpy(pGlbProb,GPUpglbprob,float_size,cudaMemcpyDeviceToHost);

  cudaFree(GPUplanduse);
  cudaFree(GPUpelev);
  cudaFree(GPUpslope);
  cudaFree(GPUpdist2cityctr);
  cudaFree(GPUpdist2Trnsprt);
  cudaFree(GPUpglbprob);
  cudaFree(GPU_aLandUseCoeffs);
}


__device__ float computeLclDensity( int myrank,int npes,int i, int j,
                                    int nPows, int nCols,
                                    short *pUrban,float *nHTRows, int nRows_global) {
  float density = 0.0;
  long int iNbr, jNbr, idxNbr,idxNbr1;

  for(iNbr = i-1; iNbr <= i+1; iNbr++) {
    for(jNbr = j-1; jNbr <= j+1; jNbr++) {
      if((iNbr != i) || (jNbr != j)) {
        idxNbr1 = coord2idx1(iNbr, jNbr, nRows_global, nCols);
        idxNbr = coord2idx1(iNbr, jNbr, nPows, nCols);
        if(idxNbr1!=ERROR_COORD){ 
          if(i+1>=nPows||i-1<0){//invaild
            if((j+1>=0)&&(j+1<nCols)){
 		          if(myrank==0){//// 0-th thread, only consider tail
			          if(i+1>=nPows){
				          density+=nHTRows[1*nCols+jNbr];
			          }
		          } else if(myrank==npes-1){// // last thread, only consider head
			            if(i-1<0){
                    density+=nHTRows[jNbr];
			            }
		          } else { // intermediate threads, consider both head and tail
                  if(i-1<0) {
                    density+=nHTRows[jNbr];
			            } else if(i+1>=nPows){
				              density+=nHTRows[1*nCols+jNbr];
			            }
		          }
            }
          } else {//vaild
              if(idxNbr1 != ERROR_COORD) {
                density+=*(pUrban+idxNbr);
              }
	        }
        }
      }
    }
  }
  density = density / 8.0;
  return density;
}

__device__ short computeCellCnstrnt(int i, int j,
                                    int nRows, int nCols,
                                    short *pExcluded,int nRows_global) {
  short cellCnstrnt = (short)ERROR_VALUE;
  short ifExcluded;
  long int cellIdx1 = coord2idx1(i, j, nRows_global, nCols);
  long int cellIdx = coord2idx1(i, j, nRows, nCols);
  if(cellIdx1 != ERROR_COORD) {
    ifExcluded = *(pExcluded + cellIdx);
    cellCnstrnt = ifExcluded ? 0:1;
  }
  return cellCnstrnt;
}

__device__ float computeJointProb(int myrank,int npes,int i, int j,
                                  int nRows, int nCols,
                                  float *pGlbProb,
                                  short  *pUrban,
                                  short  *pExcluded,float *nHTRows_d,int nRows_global) {
  float jointProb = ERROR_VALUE;
  float glbProb, lclDensity;
  short cellCnstrnt;
  
  long int cellIdx1 = coord2idx1(i, j, nRows_global, nCols);
  long int cellIdx = coord2idx1(i, j, nRows, nCols);
  if(cellIdx1 != ERROR_COORD) {
    glbProb = *(pGlbProb + cellIdx);
    if(glbProb > 0.0) {
      lclDensity = computeLclDensity(myrank,npes,i, j, nRows, nCols, pUrban,nHTRows_d,nRows_global);
      cellCnstrnt = computeCellCnstrnt(i, j, nRows, nCols, pExcluded,nRows_global);
      jointProb = glbProb * cellCnstrnt * lclDensity;
    }
    else {
      jointProb = 0.0;
    }
  }

  return jointProb;
}

__device__ float computeDistDecayProb(int i, int j,
                                      int nRows, int nCols,
                                      float *pJointProb,
                                      float maxJointProb,int nRows_global) {
  float distDecayProb = ERROR_VALUE;
  float jointProb;
  long int cellIdx1 = coord2idx1(i, j, nRows_global, nCols);
  long int cellIdx = coord2idx1(i, j, nRows, nCols);

  if(cellIdx1 != ERROR_COORD) {
    jointProb = *(pJointProb + cellIdx);
    distDecayProb = jointProb * exp(-DISPERSION * (1 - jointProb / maxJointProb));
  }

  return distDecayProb;
}

__global__ void Kernel_computerjoint(int myrank,int npes,float *GPUpglbprob,
                                     short *GPUpurban,short *GPUpexcluded,int nRows, 
                                     int nCols,float *GPUpjointprob,
                                     float *nHTRows_d,int nRows_global){

  __shared__ float maxJointProb[1];
  maxJointProb[0]=0.0;
  __shared__ float  sumDistDecayProb[1];
  sumDistDecayProb[0]=0.0;
  float constrainedProb=0.0;
  float  jointProb=0.0;
  float  distDecayProb=0.0;
  short  convert=0;
  int nCellsConvd = 0;
  int i= blockIdx.x* BLOCK_SIZE + threadIdx.x;
  int j= blockIdx.y* BLOCK_SIZE + threadIdx.y;

  if(myrank==0){
    if((i>0)&&(j>0)){
      if(( i < nRows-1)&&(j < nCols-1)) {
        jointProb = computeJointProb(myrank,npes,i, j, nRows, nCols,
                                     GPUpglbprob,
                                     GPUpurban,
                                     GPUpexcluded,nHTRows_d,nRows_global);
       
        if(fabs(jointProb - ERROR_VALUE) > EPSINON) {
          GPUpjointprob[coord2idx1(i, j, nRows, nCols)] = jointProb;	

        } else {
          GPUpjointprob[coord2idx1(i, j, nRows, nCols)] = 0.0;
        }
      
      } // end of row loop (i) and column loop (j)
    }
  } else {
      if((j>0)){
        if(( i < nRows-1)&&(j < nCols-1)) {
          jointProb = computeJointProb(myrank,npes,i, j, nRows, nCols,
                                     GPUpglbprob,
                                     GPUpurban,
                                     GPUpexcluded,nHTRows_d,nRows_global);

        if(fabs(jointProb - ERROR_VALUE) > EPSINON) {
          GPUpjointprob[coord2idx1(i, j, nRows, nCols)] = jointProb;
        }
        else {
          GPUpjointprob[coord2idx1(i, j, nRows, nCols)] = 0.0;
        }
      } // end of row loop (i) and column loop (j)
    }
  }
}

__global__ void Kernel_computerdistancedecay(int myrank,int npes,float maxJointProb,
                                             int nRows, int nCols,float *GPUpjointprob,
                                             float *GPUpdistdecayprob,float *GPUsum, 
                                             int nRows_global){
   float  sumDistDecayProb = 0.0;
   float  distDecayProb=0;
   int i= blockIdx.x* BLOCK_SIZE + threadIdx.x;
   int j= blockIdx.y* BLOCK_SIZE + threadIdx.y;
    
   if(myrank==0){
    if(( i < nRows-1)&&(j < nCols-1)&&(i>0)&&(j>0)) {
      distDecayProb = computeDistDecayProb(i, j, nRows, nCols,
                              GPUpjointprob, maxJointProb,nRows_global);
      if(fabs(distDecayProb - ERROR_VALUE) > EPSINON) {
        GPUpdistdecayprob[coord2idx1(i, j, nRows, nCols)] = distDecayProb;
      }
      else {
        GPUpdistdecayprob[coord2idx1(i, j, nRows, nCols)] = 0.0;
      }
       
    } // end of row loop (i) and  end of column loop (j)
 }
 else {
    if(( i < nRows-1)&&(j < nCols-1)&&(j>0)) {
        distDecayProb = computeDistDecayProb(i, j, nRows, nCols,
                                             GPUpjointprob, maxJointProb,nRows_global);
        if(fabs(distDecayProb - ERROR_VALUE) > EPSINON) {
          GPUpdistdecayprob[coord2idx1(i, j, nRows, nCols)] = distDecayPro
        }
        else {
         GPUpdistdecayprob[coord2idx1(i, j, nRows, nCols)] = 0.0;
        }
    } // end of row loop (i) and  end of column loop (j)
  }
}

float COMPUTEjoint(int myrank,int npes,int nRows,int nCols,float *pGlbProb,
                   short *pUrban,short *pExcluded,float *pJointProb,float *pDistDecayProb, 
                   int nCells2UrbanPerYear,int ROW_GRID,float *nHTRows,int rowExtra,
                   int rowStep,int nRows_global){ 

  long long int short_size=sizeof(short)*nRows*nCols;
  long long int float_size=sizeof(float)*nRows*nCols;
  /* compute joint probability */
  float *GPUpglbprob=NULL;
  short *GPUpurban=NULL;
  short *GPUpexcluded=NULL;
  float *GPUpjointprob=NULL;
  float *nHTRows_d=NULL;
  /* compute joint probability */
  cudaMalloc((void**) &GPUpglbprob,  float_size);
  cudaMalloc((void**) &GPUpurban,    short_size);
  cudaMalloc((void**) &GPUpexcluded, short_size);
  cudaMalloc((void**) &GPUpjointprob,float_size);
  cudaMalloc((void**) &nHTRows_d,sizeof(float)*2*nCols);
  /* compute joint probability */
  cudaMemcpy(GPUpglbprob,pGlbProb,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpurban,pUrban,short_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpexcluded,pExcluded,short_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpjointprob,pJointProb,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(nHTRows_d,nHTRows,sizeof(float)*2*nCols,cudaMemcpyHostToDevice);
 
  dim3 dimBlock,dimGrid;
  dimBlock.x=32;
  dimBlock.y=32;
  dimBlock.z=1;
        
  dimGrid.x=ROW_GRID;
  dimGrid.y=NCOLS;
  dimGrid.z=1;
        
  Kernel_computerjoint<<<dimGrid,dimBlock>>>(myrank,npes,GPUpglbprob,GPUpurban,GPUpexcluded,nRows, nCols,GPUpjointprob,nHTRows_d,nRows_global); 

  cudaMemcpy(pJointProb,GPUpjointprob,float_size,cudaMemcpyDeviceToHost);
       
  float maxJointProb1=0.0;
  for(int i=0; i<nRows*nCols;i++){
      if(pJointProb[i]>maxJointProb1)
  }     maxJointProb1=pJointProb[i];
               
  cudaFree(GPUpglbprob);
  cudaFree(GPUpurban);
  cudaFree(GPUpexcluded);
  cudaFree(GPUpjointprob);
  
  return maxJointProb1;

}


float COMPUTEdistancedecay(int myrank,int npes,int nRows,int nCols,float *pJointProb,
                           float *pDistDecayProb, float  maxJointProb,int ROW_GRID,
                          int rowExtra,int rowStep,int nRows_global){
 
  long long int short_size=sizeof(short)*nRows*nCols;
  long long int float_size=sizeof(float)*nRows*nCols;
  long long int sum_size  =sizeof(float)*NCOLS*ROW_GRID;
  
  /* compute distance decay probability */     
  float *GPUpdistdecayprob=NULL;
  float *GPUpjointprob=NULL;
  float *GPUsum=NULL;      
  float *sum;
  sum=(float *)malloc(sum_size);
  memset( sum,0,sum_size);
  
  /* compute distance decay probability */
  cudaMalloc((void**) &GPUpdistdecayprob,float_size);
  cudaMalloc((void**) &GPUpjointprob,float_size);
  cudaMalloc((void**) &GPUsum,sum_size);           

  /* compute distance decay probability */
  cudaMemcpy(GPUpdistdecayprob,pDistDecayProb,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpjointprob,pJointProb,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUsum,sum,sum_size,cudaMemcpyHostToDevice);

  dim3 dimBlock,dimGrid;
  dimBlock.x=32;
  dimBlock.y=32;
  dimBlock.z=1;
        
  dimGrid.x=ROW_GRID;
  dimGrid.y=NCOLS;
  dimGrid.z=1;

  Kernel_computerdistancedecay<<<dimGrid,dimBlock>>>( myrank,npes,maxJointProb,nRows, nCols,GPUpjointprob,GPUpdistdecayprob,GPUsum, nRows_global); 

  cudaMemcpy(pDistDecayProb,GPUpdistdecayprob,float_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(sum,GPUsum,sum_size,cudaMemcpyDeviceToHost);
  
  float sum_2=0;
  for(unsigned long long int i=0;i<nRows*nCols;i++)
      sum_2+=pDistDecayProb[i];

  cudaFree(GPUpdistdecayprob);
  cudaFree(GPUpjointprob);
  cudaFree(GPUsum);
  free(sum);
  return sum_2;
 
}

__device__ float computeConstrainedProb(int i, int j,
                             int nRows, int nCols,
                             float *pDistDecayProb,
                             float sumDistDecayProb,
                             int nCells2Urban,long *GPUseed,int nRows_global) {
  float constrainedProb = ERROR_VALUE;
  float distDecayProb;
  long int cellIdx1 = coord2idx1(i, j,  nRows_global, nCols);
  long int cellIdx = coord2idx1(i, j,  nRows, nCols);

  if(cellIdx1 != ERROR_COORD) {
    distDecayProb = *(pDistDecayProb + cellIdx);
    if(nCells2Urban > 0) {
      constrainedProb = nCells2Urban * distDecayProb / sumDistDecayProb;
    }
  }
  return constrainedProb;
}


__device__ short computeConvert(float constrainedProb,long seed1[STREAMS], 
                                int stream1,double *GPUrandom_input,int i,int j,int nRows, 
                                int nCols,int index,int rowExtra,int rowStep,int nRows_global,int myrank) {

  const long Q = MODULUS / MULTIPLIER;
  const long R = MODULUS % MULTIPLIER;
  long t;
  long seed[1];
  int stream =0;
  double m;
  seed[stream]=123456789; 
  t =MULTIPLIER * (seed[stream] % Q) - R * (seed[stream] / Q);
  if (t > 0)
    seed[stream] = t;
  else
    seed[stream] = t + MODULUS;
  m=seed[stream] / MODULUS;
  return constrainedProb >m?1:0;
}

__global__ void Kernel_computerother(int myrank,int npes,short *GPUpurban,int nRows, 
                                     int nCols,float *GPUpdistdecayprob,float sumDistDecayProb, 
                                     int nCells2UrbanPerYear,int *GPUnCellsConvd,long *GPUseed,
                                     int *GPUtest,int stream,double *GPUrandom_input,int index,  
                                     int nRows_global,int rowExtra,int rowStep){
  float constrainedProb =0.0;
  short convert=0;
  int i= blockIdx.x* BLOCK_SIZE + threadIdx.x;
  int j= blockIdx.y* BLOCK_SIZE + threadIdx.y; 
    
  if(myrank==0){
    if(( i < nRows-1)&&(j < nCols-1)&&(i>0)&&(j>0)) {
        constrainedProb = computeConstrainedProb(i, j, nRows, nCols,   GPUpdistdecayprob, sumDistDecayProb, nCells2UrbanPerYear,GPUseed,nRows_global);        

        if(fabs(constrainedProb - ERROR_VALUE) > EPSINON) {
          convert = computeConvert(constrainedProb,GPUseed,stream,GPUrandom_input, i, j, nRows,  nCols,index, rowExtra, rowStep, nRows_global,myrank);
          if(GPUpurban[coord2idx1(i, j, nRows, nCols)] == 0 && convert == 1) {
		        GPUpurban[coord2idx1(i, j, nRows, nCols)] = 1;
            GPUtest[coord2idx1(i, j, nRows, nCols)]+=1;
		      }
        }
    }
  } else {
      if(( i < nRows-1)&&(j < nCols-1)&&(j>0)) {
        constrainedProb = computeConstrainedProb(i, j, nRows, nCols,   GPUpdistdecayprob, sumDistDecayProb, nCells2UrbanPerYear,GPUseed,nRows_global);

        if(fabs(constrainedProb - ERROR_VALUE) > EPSINON) {
          convert = computeConvert(constrainedProb,GPUseed,stream,GPUrandom_input, i, j, nRows,  nCols,index, rowExtra, rowStep, nRows_global,myrank);
          if(GPUpurban[coord2idx1(i, j, nRows, nCols)] == 0 && convert == 1) {
            GPUpurban[coord2idx1(i, j, nRows, nCols)] = 1;
            GPUtest[coord2idx1(i, j, nRows, nCols)]+=1;
          }
        }
      }
  }
}

int COMPUTEothers(int myrank,int npes,int nRows,int nCols,float *pDistDecayProb,
                  float sumDistDecayProb,short  *pUrban,  int nCells2UrbanPerYear,
                  double *random_input,int yearNum,int index,int ROW_GRID,
                  int rowExtra,int rowStep,int nRows_global){
 
  long long int short_size=sizeof(short)*nRows*nCols;
  long long int float_size=sizeof(float)*nRows*nCols;
  short *GPUpurban=NULL;
  float *GPUpdistdecayprob=NULL;
  long *GPUseed=NULL;
        
  // compute stochastic probability and convert 
  int *CPUnCellsConvd;
  CPUnCellsConvd=(int *)malloc(sizeof(int)*1);
  int *test;
  test=(int *)malloc(sizeof(int)*nRows*nCols);
  for(int i=0;i<nRows*nCols;i++)
    test[i]=0;

  CPUnCellsConvd[0]=0;
  int *GPUnCellsConvd=NULL;
  int *GPUtest=NULL;
  double *GPUrandom_input;
  cudaMalloc((void**) &GPUrandom_input,sizeof(double)*1);
  cudaMalloc((void**) &GPUpurban,    short_size);
  cudaMalloc((void**) &GPUtest,    sizeof(float)*nRows*nCols);
       
  // compute distance decay probability 
  cudaMalloc((void**) &GPUpdistdecayprob,float_size);
  cudaMalloc((void**) &GPUseed,sizeof(long)*STREAMS);
  // compute stochastic probability and convert 
  cudaMalloc((void**) &GPUnCellsConvd, sizeof(int)); 

  cudaMemcpy(GPUpurban,pUrban,short_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUtest,test, sizeof(float)*nRows*nCols,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUpdistdecayprob,pDistDecayProb,float_size,cudaMemcpyHostToDevice);
  cudaMemcpy(GPUnCellsConvd,CPUnCellsConvd,sizeof(int),cudaMemcpyHostToDevice);

  dim3 dimBlock,dimGrid;
  dimBlock.x=32;
  dimBlock.y=32;
  dimBlock.z=1;
        
  dimGrid.x=ROW_GRID;
  dimGrid.y=NCOLS;
       
  dimGrid.z=1;

  Kernel_computerother<<<dimGrid,dimBlock>>>(myrank,npes,GPUpurban,nRows, nCols,GPUpdistdecayprob,sumDistDecayProb, nCells2UrbanPerYear,GPUnCellsConvd,GPUseed,GPUtest,stream,GPUrandom_input,index,nRows_global, rowExtra, rowStep); 

  cudaMemcpy(pUrban,GPUpurban,short_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(CPUnCellsConvd,GPUnCellsConvd,sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(test,GPUtest, sizeof(float)*nRows*nCols,cudaMemcpyDeviceToHost);
        
  int cnt=0;
  for(int i=0;i<nRows*nCols;i++)
    cnt+=test[i];

  cudaFree(GPUpurban);
  cudaFree(GPUpdistdecayprob);
  cudaFree(GPUnCellsConvd);
  cudaFree(GPUseed);
  cudaFree(GPUtest);
  cudaFree(GPUrandom_input);
  free(CPUnCellsConvd);
  free(test);
  return cnt; 
}
