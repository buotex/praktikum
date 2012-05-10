DROP TYPE hmm CASCADE;
DROP TABLE test_hmm;
CREATE TYPE hmm AS (
  gid           int,
  models        gm[],
  modelids      int[],
  weights       float8[],
  transitions   float8[],
  inits         float8[]

);


CREATE TABLE test_hmm (
  hmm hmm
);



CREATE OR REPLACE FUNCTION hd_hmm_compare(hmm, hmm, int default 1)
RETURNS double precision
AS '$libdir/hd_hmm', 'hd_hmm_compare'
LANGUAGE 'C' IMMUTABLE STRICT;


CREATE OR REPLACE FUNCTION hd_hmm_create(int,int)
RETURNS hmm
AS '$libdir/hd_hmm', 'hd_hmm_create'
LANGUAGE 'C' IMMUTABLE STRICT;

INSERT INTO test_hmm SELECT hd_hmm_create(1,3) AS value;
INSERT INTO test_hmm SELECT hd_hmm_create(2,3) AS value;
SELECT hd_hmm_compare(T1.hmm, T2.hmm, 1) from test_hmm as T1, test_hmm as T2;

select * from test_hmm;

--DROP TYPE gm CASCADE;
