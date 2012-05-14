#include "postgres.h"
#include <string.h>
#include "fmgr.h"
#include "libpq/pqformat.h"		
#include "hmm/gmm.hpp"
#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

PG_FUNCTION_INFO_V1(gm_in);

Datum
gm_in(PG_FUNCTION_ARGS) {
  char *str = PG_GETARG_CSTRING(0);
  double mean[3];
  double sigma[6];
  GM_c *result;
  if (sscanf(str, " ( %lf %lf %lf ) - ( %lf %lf %lf %lf %lf %lf )", &mean[0], &mean[1], &mean[2], &sigma[0], &sigma[1], &sigma[2], &sigma[3], &sigma[4], &sigma[5]) != 9) {
    ereport(ERROR,
        (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
         errmsg("invalid input syntax for gm: \"%s\"",
           str)));
  }
  result = (GM_c *) palloc(sizeof(GM_c));
  memcpy(result->mean, mean, sizeof(mean));
  memcpy(result->sigma, sigma, sizeof(sigma));
  
  PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(gm_out);

Datum
gm_out(PG_FUNCTION_ARGS) {
  GM_c *gm = (GM_c *) PG_GETARG_POINTER(0);
  char *result;
  result = (char *) palloc(200);
  snprintf(result, 200, "(%g %g %g) - (%g %g %g %g %g %g)", gm->mean[0], gm->mean[1],gm->mean[2], gm->sigma[0], gm->sigma[1], gm->sigma[2], gm->sigma[3], gm->sigma[4], gm->sigma[5]);
  PG_RETURN_CSTRING(result);
}

PG_FUNCTION_INFO_V1(gm_recv);

Datum
gm_recv(PG_FUNCTION_ARGS) {
  StringInfo buf = (StringInfo) PG_GETARG_POINTER(0);
  GM_c *result;
  result = (GM_c *) palloc(sizeof(GM_c));
  result->mean[0] = pq_getmsgfloat8(buf);
  result->mean[1] = pq_getmsgfloat8(buf);
  result->mean[2] = pq_getmsgfloat8(buf);
  result->sigma[0] = pq_getmsgfloat8(buf);
  result->sigma[1] = pq_getmsgfloat8(buf);
  result->sigma[2] = pq_getmsgfloat8(buf);
  result->sigma[3] = pq_getmsgfloat8(buf);
  result->sigma[4] = pq_getmsgfloat8(buf);
  result->sigma[5] = pq_getmsgfloat8(buf);
  PG_RETURN_POINTER(result);
}


PG_FUNCTION_INFO_V1(gm_send);

Datum
gm_send(PG_FUNCTION_ARGS) {
  GM_c *gm = (GM_c*) PG_GETARG_POINTER(0);
  StringInfoData buf;
  pq_begintypsend(&buf);
  pq_sendfloat8(&buf, gm->mean[0]);
  pq_sendfloat8(&buf, gm->mean[1]);
  pq_sendfloat8(&buf, gm->mean[2]);
  pq_sendfloat8(&buf, gm->sigma[0]);
  pq_sendfloat8(&buf, gm->sigma[1]);
  pq_sendfloat8(&buf, gm->sigma[2]);
  pq_sendfloat8(&buf, gm->sigma[3]);
  pq_sendfloat8(&buf, gm->sigma[4]);
  pq_sendfloat8(&buf, gm->sigma[5]);
  PG_RETURN_BYTEA_P(pq_endtypsend(&buf));

}


