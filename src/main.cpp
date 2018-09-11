
#include"header.h"
#include "gdal.h"
#include "cpl_conv.h"
#include<mpi.h>

/* ----------------------------------------------------------------
 * Main Function
 * ----------------------------------------------------------------*/
void  COMPUTEglbprob(int myrank,int npes,int  pRows,int  nCols,short *pLandUse,float *pElev,float *pSlope,float *pDist2CityCtr,float *pDist2Trnsprt,float *pGlbProb,int ROW_GRID,int rowExtra,int rowStep,int nRows);

float COMPUTEjoint(int myrank,int npes,int pRows,int  nCols,float *pGlbProb,short *pUrban, short *pExcluded,float *pJointProb,float *pDistDecayProb,int nCells2UrbanPerYear,int ROW_GRID,float *nHTRows,int rowExtra,int rowStep,int nRows);


float COMPUTEdistancedecay(int myrank,int npes,int pRows,int  nCols,float *pJointProb,float *pDistDecayProb,float maxJointProb,int ROW_GRID,int rowExtra,int rowStep,int nRows);

int COMPUTEothers(int myrank,int npes,int pRows,int  nCols,float *pDistDecayProb,float sumDistDecayProb, short *pUrban, int nCells2UrbanPerYear,double *random_input,int yearNum,int index_year,int ROW_GRID,int rowExtra,int rowStep,int nRows);

long int coord2idx(int i, int j, int nRows, int nCols) {
        long int idx = ERROR_COORD;
        if(i >= 0 && i < nRows && j >= 0 && j < nCols) {
                idx = i * nCols + j;
         }
         return idx;
 }


int main(int argc, char *argv[]) {

  int myrank,npes;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
  MPI_Comm_size (MPI_COMM_WORLD, &npes);


  MPI_Status status;


  float _aLandUseCoeffs[9] = {0.0, 0.0,-9.8655,-8.7469,-9.2688, -8.0321, -9.1693, -8.9420,-9.4500 };
  int t,t1,t2,t3;
  t=clock();
  /* ----------------------- VARIABLES ----------------------- */
  int nRows, nCols;
  int i, j;

  int yearStart=1992, yearEnd=2012;
  int year;
  int nCells2UrbanPerYear = 20000; // San Diego data
  int nCellsConvd = 0, nTotCellsConvd = 0;

  float jointProb, maxJointProb;
  float distDecayProb, sumDistDecayProb;
  float constrainedProb;
  short convert;

  /* ----------------------- LAYERS ----------------------- */
  short *pUrban;
  short *pLandUse;
  float *pElev;
  float *pSlope;
  float *pDist2CityCtr;
  float *pDist2Trnsprt;
  short *pExcluded;
  float *pGlbProb;
  float *pJointProb;
  float *pDistDecayProb;

  /* ----------------------- GDAL VARIABLES ----------------------- */
  GDALDatasetH hOutUrbanDS, hUrbanDS, hLandUseDS, hElevDS, hSlopeDS, hDist2CityCtrDS, hDist2TrnsprtDS, hExcludedDS;
  GDALRasterBandH hOutUrbanBd, hUrbanBd, hLandUseBd, hElevBd, hSlopeBd, hDist2CityCtrBd, hDist2TrnsprtBd, hExcludedBd;
  char *aOutUrbanFile = "outUrban.tif";
  char *aUrbanFile = "../lcUrban.tif";
  char *aLandUseFile = "../landuse92.tif";
  //char *aElevFile = "dem_nrm.tif";
  char *aElevFile = "../dem30m_nrm.tif";
  //char *aSlopeFile = "slope_nrm.tif";
  char *aSlopeFile = "../slope30m_nrm.tif";
  char *aDist2CityCtrFile = "../dist2cityCtrs_nrm.tif";
  char *aDist2TrnsprtFile = "../dist2transp_nrm.tif";
  char *aExcludedFile = "../excluded.tif";

  const char *aFormatName = "GTiff";
  GDALDriverH hGTiffDriver;

  /* ----------------------- LOAD DATA ----------------------- */
  GDALAllRegister();
  hGTiffDriver = GDALGetDriverByName(aFormatName);
  if(hGTiffDriver == NULL) {
        printf("Error: GTiff driver can not be found!\n");
    exit(1);
  }

  hUrbanDS = GDALOpen(aUrbanFile, GA_ReadOnly);
  nRows = GDALGetRasterYSize(hUrbanDS);
  nCols = GDALGetRasterXSize(hUrbanDS);

  if(myrank==0){
         printf("nRows,nCols=%d,%d\n",nRows,nCols);
         printf("npes is %d\n",npes);
  }
 
  int rowExtra=nRows%npes;
  int rowStep =nRows/npes;
  int pRows;

        if(myrank==0){
             pRows=rowExtra+rowStep;

                 hUrbanBd = GDALGetRasterBand(hUrbanDS, 1);
                 pUrban = (short *)CPLMalloc(sizeof(short)*pRows*nCols);
                 GDALRasterIO(hUrbanBd, GF_Read, 0, 0, nCols, pRows, pUrban, nCols, pRows, GDT_Int16, 0, 0);

                 hLandUseDS = GDALOpen(aLandUseFile, GA_ReadOnly);
                 hLandUseBd = GDALGetRasterBand(hLandUseDS, 1);
                 pLandUse = (short *)CPLMalloc(sizeof(short)*pRows*nCols);
                 GDALRasterIO(hLandUseBd, GF_Read, 0, 0, nCols, pRows, pLandUse, nCols, pRows, GDT_Int16, 0, 0);
                 GDALClose(hLandUseDS);

                 hElevDS = GDALOpen(aElevFile, GA_ReadOnly);
                 hElevBd = GDALGetRasterBand(hElevDS, 1);
                 pElev = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
                 GDALRasterIO(hElevBd, GF_Read, 0, 0, nCols, pRows, pElev, nCols, pRows, GDT_Float32, 0, 0);
                 GDALClose(hElevDS);

                 hSlopeDS = GDALOpen(aSlopeFile, GA_ReadOnly);
                 hSlopeBd = GDALGetRasterBand(hSlopeDS, 1);
                 pSlope = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
                 GDALRasterIO(hSlopeBd, GF_Read, 0, 0, nCols, pRows, pSlope, nCols, pRows, GDT_Float32, 0, 0);
                 GDALClose(hSlopeDS);

                 hDist2CityCtrDS = GDALOpen(aDist2CityCtrFile, GA_ReadOnly);
                 hDist2CityCtrBd = GDALGetRasterBand(hDist2CityCtrDS, 1);
                 pDist2CityCtr = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
                 GDALRasterIO(hDist2CityCtrBd, GF_Read, 0, 0, nCols, pRows, pDist2CityCtr, nCols, pRows, GDT_Float32, 0, 0);
                 GDALClose(hDist2CityCtrDS);

                 hDist2TrnsprtDS = GDALOpen(aDist2TrnsprtFile, GA_ReadOnly);
                 hDist2TrnsprtBd = GDALGetRasterBand(hDist2TrnsprtDS, 1);
                 pDist2Trnsprt = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
                 GDALRasterIO(hDist2TrnsprtBd, GF_Read, 0, 0, nCols, pRows, pDist2Trnsprt, nCols, pRows, GDT_Float32, 0, 0);
                 GDALClose(hDist2TrnsprtDS);

                 hExcludedDS = GDALOpen(aExcludedFile, GA_ReadOnly);
                 hExcludedBd = GDALGetRasterBand(hExcludedDS, 1);
                 pExcluded = (short *)CPLMalloc(sizeof(short)*pRows*nCols);
                 GDALRasterIO(hExcludedBd, GF_Read, 0, 0, nCols, pRows, pExcluded, nCols, pRows, GDT_Int16, 0, 0);
                 GDALClose(hExcludedDS);

           } else {

                pRows=rowStep;

                hUrbanBd = GDALGetRasterBand(hUrbanDS, 1);
                pUrban = (short *)CPLMalloc(sizeof(short)*pRows*nCols);
                GDALRasterIO(hUrbanBd, GF_Read,0, rowExtra + myrank * rowStep, nCols, pRows, pUrban, nCols, pRows, GDT_Int16, 0, 0);
                 
                hLandUseDS = GDALOpen(aLandUseFile, GA_ReadOnly);
                hLandUseBd = GDALGetRasterBand(hLandUseDS, 1);
                pLandUse = (short *)CPLMalloc(sizeof(short)*pRows*nCols);
                GDALRasterIO(hLandUseBd, GF_Read,0, rowExtra + myrank * rowStep, nCols, pRows, pLandUse, nCols, pRows, GDT_Int16, 0, 0);
                GDALClose(hLandUseDS);

                hElevDS = GDALOpen(aElevFile, GA_ReadOnly);
                hElevBd = GDALGetRasterBand(hElevDS, 1);
                pElev = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
                GDALRasterIO(hElevBd, GF_Read,0, rowExtra + myrank * rowStep, nCols, pRows, pElev, nCols, pRows, GDT_Float32, 0, 0);
                GDALClose(hElevDS);

                hSlopeDS = GDALOpen(aSlopeFile, GA_ReadOnly);
                hSlopeBd = GDALGetRasterBand(hSlopeDS, 1);
                pSlope = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
                GDALRasterIO(hSlopeBd, GF_Read,0, rowExtra + myrank * rowStep, nCols, pRows, pSlope, nCols, pRows, GDT_Float32, 0, 0);
                GDALClose(hSlopeDS);

                hDist2CityCtrDS = GDALOpen(aDist2CityCtrFile, GA_ReadOnly);
                hDist2CityCtrBd = GDALGetRasterBand(hDist2CityCtrDS, 1);
                pDist2CityCtr = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
                GDALRasterIO(hDist2CityCtrBd, GF_Read,0, rowExtra + myrank * rowStep, nCols, pRows, pDist2CityCtr, nCols, pRows, GDT_Float32, 0, 0);
                GDALClose(hDist2CityCtrDS);

                hDist2TrnsprtDS = GDALOpen(aDist2TrnsprtFile, GA_ReadOnly);
                hDist2TrnsprtBd = GDALGetRasterBand(hDist2TrnsprtDS, 1);
                pDist2Trnsprt = (float *)CPLMalloc(sizeof(float)*pRows*nCols);
   		          GDALRasterIO(hDist2TrnsprtBd, GF_Read, 0,rowExtra + myrank * rowStep, nCols, pRows, pDist2Trnsprt, nCols, pRows, GDT_Float32, 0, 0);
  		          GDALClose(hDist2TrnsprtDS);

  		          hExcludedDS = GDALOpen(aExcludedFile, GA_ReadOnly);
 	              hExcludedBd = GDALGetRasterBand(hExcludedDS, 1);
  		          pExcluded = (short *)CPLMalloc(sizeof(short)*pRows*nCols);
  		          GDALRasterIO(hExcludedBd, GF_Read, 0, rowExtra + myrank * rowStep, nCols, pRows, pExcluded, nCols,pRows, GDT_Int16, 0, 0);
  		          GDALClose(hExcludedDS);
          }
       
  MPI_Barrier(MPI_COMM_WORLD);  

  pGlbProb = (float *)calloc(pRows*nCols, sizeof(float));
  pJointProb = (float *)calloc(pRows*nCols, sizeof(float));
 	pDistDecayProb = (float *)calloc(pRows*nCols, sizeof(float));

  for(int i=0;i<pRows*nCols;i++){
    pGlbProb[i]=0;
    pJointProb[i]=0;
    pDistDecayProb[i]=0;
  }

  int yearNum=yearEnd-yearStart; 
  double *random_input=(double *)malloc(sizeof(double)*1);
  random_input[0]=1.0;   
  float maxJointProb_rank[npes];
  int nHoTSize = nCols;
  float *nHTRows = (float *)malloc(sizeof(float) * 2 * nHoTSize);
  int ROW_GRID=0;
  ROW_GRID=pRows/32+(pRows%32?1:0);
  
  COMPUTEglbprob( myrank,npes,pRows, nCols,pLandUse, pElev, pSlope,pDist2CityCtr, pDist2Trnsprt,pGlbProb,ROW_GRID,rowExtra,rowStep,nRows);

  CPLFree(pLandUse);
  CPLFree(pElev);
  CPLFree(pSlope);
  CPLFree(pDist2CityCtr);
  CPLFree(pDist2Trnsprt);


  float sumDistDecayProb_1;
  float sumDistDecayProb_local;
  float sumDistDecayProb_global[npes];
  int sum_local=0;
  int sum_global=0;
  int index_year=0;
  for(year = yearStart+1; year <= yearEnd; year++) {
    maxJointProb = 0.0;
    sumDistDecayProb = 0.0;
    nCellsConvd = 0;

  	if (myrank % 2 == 1) {
      // send tail and recv head 
      MPI_Send(pUrban+(pRows-1)*nCols, nHoTSize, MPI_FLOAT, (myrank + 1) % npes, 1, MPI_COMM_WORLD);
      MPI_Recv(nHTRows, nHoTSize, MPI_FLOAT, (myrank - 1 + npes) % npes, 1, MPI_COMM_WORLD, &status);
      // send head and recv tail
      MPI_Send(pUrban , nHoTSize, MPI_FLOAT, (myrank - 1 + npes) % npes, 2, MPI_COMM_WORLD);
      MPI_Recv(nHTRows + nCols, nHoTSize, MPI_FLOAT, (myrank + 1) % npes, 2, MPI_COMM_WORLD, &status);
  	} else {
      // recv head and send tail
      MPI_Recv(nHTRows, nHoTSize, MPI_FLOAT, (myrank - 1 + npes) % npes, 1, MPI_COMM_WORLD, &status);
      MPI_Send(pUrban + ( pRows - 1) * nCols, nHoTSize, MPI_FLOAT, (myrank + 1) % npes, 1, MPI_COMM_WORLD);
      // recv tail and send head
      MPI_Recv(nHTRows + nCols, nHoTSize, MPI_INT, (myrank + 1) % npes, 2, MPI_COMM_WORLD, &status);
      MPI_Send(pUrban, nHoTSize, MPI_FLOAT, (myrank - 1 + npes) % npes, 2, MPI_COMM_WORLD);
    }

    maxJointProb =COMPUTEjoint(myrank,npes,pRows, nCols,pGlbProb,pUrban, pExcluded,pJointProb,pDistDecayProb,nCells2UrbanPerYear,ROW_GRID,nHTRows,rowExtra,rowStep,nRows);

    for(int i=0;i<npes;i++){
      if(i!=myrank){
        MPI_Send(&maxJointProb,1,MPI_FLOAT,i,3,MPI_COMM_WORLD);
        MPI_Recv(&maxJointProb_rank[i],1,MPI_FLOAT,i,3,MPI_COMM_WORLD,&status);
  		}
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
  
    for(int i=0; i<npes; i++){
	    if(i!=myrank){
	 		  if( maxJointProb_rank[i]>maxJointProb)
          maxJointProb=maxJointProb_rank[i];
		  }
    }		

    sumDistDecayProb_1=COMPUTEdistancedecay(myrank,npes,pRows, nCols,pJointProb,pDistDecayProb,maxJointProb,ROW_GRID,rowExtra,rowStep,nRows);
 
    sumDistDecayProb_local=sumDistDecayProb_1;

    for(int i=0;i<npes;i++){
      if(i!=myrank){
        MPI_Send(&sumDistDecayProb_local,1,MPI_FLOAT,i,4,MPI_COMM_WORLD);
        MPI_Recv(&sumDistDecayProb_global[i],1,MPI_FLOAT,i,4,MPI_COMM_WORLD,&status);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=0; i<npes; i++){
      if(i!=myrank){
        sumDistDecayProb_local+=sumDistDecayProb_global[i];
      }
    }

    sum_local=COMPUTEothers(myrank,npes,pRows, nCols,pDistDecayProb,sumDistDecayProb_local, pUrban, nCells2UrbanPerYear,random_input,yearNum,index_year,ROW_GRID,rowExtra,rowStep,nRows);

    MPI_Reduce(&sum_local,&sum_global,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    
    if(myrank==0){
      printf("livecell converts to Urban is %d\n",sum_global);
      printf("maxjoint is %f\n ",maxJointProb);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD); 

  CPLFree(pExcluded);
  free(pGlbProb);
  free(pJointProb);
  free(pDistDecayProb);
  free(random_input);

  return 0;
}

































