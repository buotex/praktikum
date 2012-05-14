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



int checkString(char * cstring) {
  if (strcmp("gid", cstring) == 0)
    return 0;
  if (strcmp("models", cstring) == 0)
    return 1;
  if (strcmp("modelids", cstring) == 0)
    return 2;

  if (strcmp("weights", cstring) == 0)
    return 3;

  if (strcmp("transitions", cstring) == 0)
    return 4;

  if (strcmp("inits", cstring) == 0)
    return 5;
  return -1;
}

void
extractHMM(double * matrix, int ndata, Datum * values, bool * nulls, TupleDesc tupdesc, int nclusters, int * edges, int nedges ) {
  void * hmm = constructHMM(matrix, ndata, nclusters, edges, nedges);
  int i;

  for (i = 0; i < tupdesc->natts; ++i) {
    int id = checkString(tupdesc->attrs[i]->attname.data);
    if (id > 5 || id < 1)
      continue;
    Oid elmtype = get_base_element_type(tupdesc->attrs[i]->atttypid);
    int nelems;
    int16 elmlen;
    bool elmbyval;
    char elmalign;
    get_typlenbyvalalign(elmtype, &elmlen, &elmbyval, &elmalign);

    Datum * element;
    switch(id) {
      case 1:
        element = (Datum*) getModels(hmm, &nelems);
        nulls[i] = false;
        break;
      case 2:
        element = (Datum*) getModelIds(hmm, &nelems);
        nulls[i] = false;
        break;
      case 3:
        element = (Datum*) getWeights(hmm, &nelems);
        nulls[i] = false;
        break;
      case 4:
        element = (Datum*) getTransitions(hmm, &nelems);
        nulls[i] = false;
        break;
      case 5:
        element = (Datum*) getInits(hmm, &nelems);
        nulls[i] = false;
        break;
    }

    values[i] = PointerGetDatum(construct_array(element, nelems, elmtype, elmlen, elmbyval, elmalign));
  }



  deleteHMM(hmm);

}


PG_FUNCTION_INFO_V1 ( hd_hmm_create );
Datum
hd_hmm_create ( PG_FUNCTION_ARGS ){
  TupleDesc            tupdesc;
  HeapTuple             tuple;
  //AttInMetadata       *attinmeta;
  int32 gid = PG_GETARG_INT32(0);
  int32 nclusters = PG_GETARG_INT32(1);
  Datum * values;
  bool * nulls;
  int i, ndata, nedges;

  double x, y, z;
  int source, target;
  char sql[100];
  int  ret, proc;
  bool isnull;


  sprintf(sql, "SELECT st_x(node), st_y(node), st_z(node) FROM graph WHERE gid=%d;", gid);

  if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
    elog(ERROR, "hd_hmm_create: return type must be a row type");

  if ((ret = SPI_connect()) < 0)
    elog(INFO, "hd_hmm_create: SPI_connect returned %d", ret);
  ret = SPI_execute(sql, true, 0);
  if (ret < 0)
    elog(ERROR, "hd_hmm_create: SPI_execute returned %d", ret);

  proc = SPI_processed;
  ndata = proc;
  if (proc < 1)
    elog(ERROR, "hd_hmm_create: No rows where found, missing gid: %d", gid);
  //elog (INFO, "hd_hmm_create: there are %d rows", proc);
  double * matrix = malloc(3 * proc *  sizeof(double));
  if (matrix == 0) 
    elog(ERROR, "hd_hmm_create: Couldn't alloc memory for matrix");
  if (SPI_tuptable != NULL) {

    TupleDesc inputtupdesc = SPI_tuptable->tupdesc;
    SPITupleTable *inputtuptable = SPI_tuptable;
    bool x_isnull, y_isnull, z_isnull;
    int i;
    for (i = 0; i < proc; ++i) {
      HeapTuple inputtuple = inputtuptable->vals[i];
      x = DatumGetFloat8(SPI_getbinval(inputtuple, inputtupdesc, 1, &x_isnull));
      y = DatumGetFloat8(SPI_getbinval(inputtuple, inputtupdesc, 2, &y_isnull));
      z = DatumGetFloat8(SPI_getbinval(inputtuple, inputtupdesc, 3, &z_isnull));
      if (x_isnull || y_isnull || z_isnull) {
        elog(ERROR, "hd_hmm_create: NULL pointer error!");
        PG_RETURN_NULL();
      }
      matrix[3 * i] = x;
      matrix[3 * i + 1] = y;
      matrix[3 * i + 2] = z;
    }
  }
  int * edges = 0;
  if (nclusters <= 0){ 
    sprintf(sql, "SELECT source, target FROM edges WHERE gid=%d;", gid);
    proc = SPI_processed;

    ret = SPI_execute(sql, true, 0);
    if (ret < 0)
      elog(ERROR, "hd_hmm_create: SPI_execute returned %d", ret);

    proc = SPI_processed;
    nedges = proc;
    if (proc < 1)
      elog(ERROR, "hd_hmm_create: No rows where found, missing gid: %d", gid);
    edges = malloc(2 * proc *  sizeof(int));
    if (edges == 0) 
      elog(ERROR, "hd_hmm_create: Couldn't alloc memory for edges");
    if (SPI_tuptable != NULL) {

      TupleDesc inputtupdesc = SPI_tuptable->tupdesc;
      SPITupleTable *inputtuptable = SPI_tuptable;
      bool source_isnull, target_isnull;
      int i;
      for (i = 0; i < proc; ++i) {
        HeapTuple inputtuple = inputtuptable->vals[i];
        source = DatumGetInt32(SPI_getbinval(inputtuple, inputtupdesc, 1, &source_isnull));
        target = DatumGetInt32(SPI_getbinval(inputtuple, inputtupdesc, 2, &target_isnull));
        if (source_isnull || target_isnull ) {
          elog(ERROR, "hd_hmm_create: NULL pointer error!");
          PG_RETURN_NULL();
        }
        edges[2 * i] = source;
        edges[2 * i + 1] = target;
      }
    }
  }

  values = palloc(sizeof(Datum) * tupdesc->natts);
  nulls = palloc(sizeof(bool) * tupdesc->natts);
  for (i = 0; i < tupdesc->natts; ++i) {
    nulls[i] = true;
  }

  for (i = 0; i < tupdesc->natts; ++i) {
    int id = checkString(tupdesc->attrs[i]->attname.data);
    if (id == 0) {
      nulls[i] = false;
      values[i] = Int32GetDatum(gid);
    }
  }
  extractHMM(matrix, ndata, values, nulls, tupdesc, nclusters, edges, nedges );



  tuple = heap_form_tuple(tupdesc, values, nulls);
  SPI_finish();
  if (edges != 0) free(edges);
  free(matrix);
  PG_RETURN_DATUM(HeapTupleGetDatum(tuple));

}

PG_FUNCTION_INFO_V1( hd_hmm_compare );
Datum
hd_hmm_compare( PG_FUNCTION_ARGS ){
  HeapTupleHeader record1 = PG_GETARG_HEAPTUPLEHEADER(0);
  HeapTupleHeader record2 = PG_GETARG_HEAPTUPLEHEADER(1);
  int32 choice = PG_GETARG_INT32(2);
  bool isnull;
  int         i = 0;
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
  if (max2 + 1 != ninits2) 
    elog(INFO, "Wrong number of states: %d, %d", max2 + 1, ninits2);
  if (ninits2 * ninits2 != ntransitions2) 
    elog(INFO, "Wrong number of transitions: %d, %d", ninits2 * ninits2, ntransitions2);

  float8 similarity = compareHMMs(
      choice,
      gmmdata1, idsdata1, weightsdata1, ngmm1, transitionsdata1, initsdata1, ninits1,
      gmmdata2, idsdata2, weightsdata2, ngmm2, transitionsdata2, initsdata2, ninits2);
  PG_RETURN_FLOAT8(similarity);

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
