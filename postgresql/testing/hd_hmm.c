#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "catalog/pg_type.h"
#include "libpq/pqformat.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/typcache.h"
#include <string.h>
#include <hmm/gmm.hpp>
#include <hmm/kld.hpp>

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

//char * models = "models";
//char * modelids = "modelids";
//char * weights = "weights";
//char * transitions = "transitions";
//char * inits = "inits";


PG_FUNCTION_INFO_V1( print_hmm );
Datum
compare_hmm( PG_FUNCTION_ARGS ){
  HeapTupleHeader record1 = PG_GETARG_HEAPTUPLEHEADER(0);
  HeapTupleHeader record2 = PG_GETARG_HEAPTUPLEHEADER(1);
  bool isnull;
  int         i = 0;
  /*
  char        *desired_col_name;
  Oid tupType1, tupType2;
  //int32 tupTypmod;
  //TupleDesc   tupdesc;
  //HeapTupleData   tuple;
  //Datum *values;
  bool *nulls;
  int nitems1;
  int nitems2;
  
  
  int         ncolumns;
  //tupType = HeapTupleHeaderGetTypeId(record);
  //tupTypmod = HeapTupleHeaderGetTypMod(record);
  //tupdesc = lookup_rowtype_tupdesc(tupType, tupTypmod);
  //ncolumns = tupdesc->natts;
  */
  ArrayType * gmm1 = DatumGetArrayTypeP(GetAttributeByName(record1, "models", &isnull));
  ArrayType * modelids1 = DatumGetArrayTypeP(GetAttributeByName(record1, "modelids", &isnull));
  ArrayType * weights1 = DatumGetArrayTypeP(GetAttributeByName(record1, "weights", &isnull));
  ArrayType * transitions1 = DatumGetArrayTypeP(GetAttributeByName(record1, "transitions", &isnull));
  ArrayType * inits1 = DatumGetArrayTypeP(GetAttributeByName(record1, "inits", &isnull));
  
  ArrayType * gmm2 = DatumGetArrayTypeP(GetAttributeByName(record2, "models", &isnull));
  ArrayType * modelids2 = DatumGetArrayTypeP(GetAttributeByName(record2, "modelids", &isnull));
  ArrayType * weights2 = DatumGetArrayTypeP(GetAttributeByName(record2, "weights", &isnull));
  ArrayType * transitions2 = DatumGetArrayTypeP(GetAttributeByName(record2, "transitions", &isnull));
  ArrayType * inits2 = DatumGetArrayTypeP(GetAttributeByName(record2, "inits", &isnull));
  
  GM_c *gmmdata1, *gmmdata2;
  int *idsdata1, *idsdata2;
  double *weightsdata1, *weightsdata2;
  double *transitionsdata1, *transitionsdata2;
  double *initsdata1, *initsdata2;
  
  gmmdata1 = (GM_c *) ARR_DATA_PTR(gmm1);
  int ngmm1 = ArrayGetNItems(ARR_NDIM(gmm1), ARR_DIMS(gmm1));
  idsdata1 = (int *) ARR_DATA_PTR(modelids1);
  int nids1 = ArrayGetNItems(ARR_NDIM(modelids1), ARR_DIMS(modelids1));
  weightsdata1 = (double *) ARR_DATA_PTR(weights1);
  int nweights1 = ArrayGetNItems(ARR_NDIM(weights1), ARR_DIMS(weights1));
  transitionsdata1 = (double *) ARR_DATA_PTR(transitions1);
  int ntransitions1 = ArrayGetNItems(ARR_NDIM(transitions1), ARR_DIMS(transitions1));
  initsdata1 = (double *) ARR_DATA_PTR(inits1);
  int ninits1 = ArrayGetNItems(ARR_NDIM(inits1), ARR_DIMS(inits1));


  gmmdata2 = (GM_c *) ARR_DATA_PTR(gmm2);
  int ngmm2 = ArrayGetNItems(ARR_NDIM(gmm2), ARR_DIMS(gmm2));
  idsdata2 = (int *) ARR_DATA_PTR(modelids2);
  int nids2 = ArrayGetNItems(ARR_NDIM(modelids2), ARR_DIMS(modelids2));
  weightsdata2 = (double *) ARR_DATA_PTR(weights2);
  int nweights2 = ArrayGetNItems(ARR_NDIM(weights2), ARR_DIMS(weights2));
  transitionsdata2 = (double *) ARR_DATA_PTR(transitions2);
  int ntransitions2 = ArrayGetNItems(ARR_NDIM(transitions2), ARR_DIMS(transitions2));
  initsdata2 = (double *) ARR_DATA_PTR(inits2);
  int ninits2 = ArrayGetNItems(ARR_NDIM(inits2), ARR_DIMS(inits2));
  
//DEBUGGING
  if (ngmm1 != nids1 || ngmm1 != nweights1) 
    elog(INFO, "GMs weren't tagged correctly: %d, %d, %d", ngmm1, nids1, nweights1);
    
  int max1 = 0;
  for (i = 0; i < nids1; ++i) 
    max1 = (max1 > idsdata1[i])?max1:idsdata1[i];
  if (max1 + 1 != ninits1) 
    elog(INFO, "Wrong number of states: %d, %d", max1 + 1, ninits1);
  if (ninits1 * ninits1 != ntransitions1) 
    elog(INFO, "Wrong number of transitions: %d, %d", ninits1 * ninits1, ntransitions1);
  if (ngmm2 != nids2 || ngmm2 != nweights2) 
    elog(INFO, "GMs weren't tagged correctly: %d, %d, %d", ngmm2, nids2, nweights2);
    
  int max2 = 0;
  for (i = 0; i < nids2; ++i) 
    max2 = (max2 > idsdata2[i])?max2:idsdata2[i];
  if (max2 + 2 != ninits2) 
    elog(INFO, "Wrong number of states: %d, %d", max2 + 2, ninits2);
  if (ninits2 * ninits2 != ntransitions2) 
    elog(INFO, "Wrong number of transitions: %d, %d", ninits2 * ninits2, ntransitions2);

return compareHMMs(
gmmdata1, idsdata1, weightsdata1, ngmm1, transitionsdata1, initsdata1, ninits1,
gmmdata2, idsdata2, weightsdata2, ngmm2, transitionsdata2, initsdata2, ninits2);
  //values = (Datum *) palloc(ncolumns * sizeof(Datum));
/*
  for (i = 0; i < nitems1; ++i) {
    Gm * test = &gm[i];
    elog(INFO, "%g \n ", test->mean[0]);
  }

  for (i = 0; i < nitems2; ++i) {
    elog(INFO, "%d \n ", s[i]);
  }
  elog(INFO, "%d %d", nitems1, nitems2);
 */
 //ReleaseTupleDesc(tupdesc);
  
  

  //desired_col_name = PG_TEXT_GET_CSTR(PG_GETARG_TEXT_P(1));

}
