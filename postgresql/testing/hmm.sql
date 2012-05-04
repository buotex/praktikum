DROP TYPE hmm CASCADE;
DROP TABLE test_hmm;
CREATE TYPE hmm AS (
  gid           int,
  models        _gm,
  modelids      int[],
  weights       float8[],
  transitions   float8[],
  inits         float8[]

);


CREATE TABLE test_hmm (
  hmm hmm
);



CREATE OR REPLACE FUNCTION hd_hmm_compare(hmm, hmm)
RETURNS double precision
AS '$libdir/hd_hmm', 'hd_hmm_compare'
LANGUAGE 'C' IMMUTABLE STRICT;


--INSERT INTO test_hmm VALUES('("{(1.0 2.5 3.5) - (1.0 4.5 3.4 0.5 3.5), (4.2 3.55 1.6) - (6.3 3.5 2.3 5.3 2.4)}", {1,2}, {2.9, 4.0}, {1.0, 2.0}, {2.0, 4.0})' );
--INSERT INTO test_hmm VALUES('{(1.0 2.5 3.5) - (1.0 4.5 3.4 0.5 3.5), (4.2 3.55 1.6) - (6.3 3.5 2.3 5.3 2.4)}' );
--INSERT INTO test_hmm VALUES (
--ROW(0,
--ARRAY[('(2.0 2.5 3.5) - (1.0 2.0 1.7 9.5 1.5 5.0)')::gm, ('(4.2 3.55 1.6) - (6.3 3.5 2.3 5.3 2.4 3.0)')::gm],
--ARRAY[0,1],
--ARRAY[1.0,1.0],
--ARRAY[0.5,0.2,0.5,0.8],
--ARRAY[0.4,0.6]
--));
--INSERT INTO test_hmm VALUES (
--ROW(1,
--ARRAY[('(2.0 2.5 3.5) - (5.0 2.0 1.7 9.5 1.5 10.0)')::gm, ('(4.2 3.55 1.6) - (6.3 3.5 2.3 5.3 2.4 4.0)')::gm],
--ARRAY[0,1],
--ARRAY[1.0,1.0],
--ARRAY[0.5,0.2,0.5,0.8],
--ARRAY[0.3,0.7]
--));
--
--SELECT * from test_hmm;
SELECT hd_hmm_compare(T1.hmm, T2.hmm) from test_hmm as T1, test_hmm as T2;

CREATE OR REPLACE FUNCTION hd_hmm_create(int)
RETURNS hmm
AS '$libdir/hd_hmm', 'hd_hmm_create'
LANGUAGE 'C' IMMUTABLE STRICT;

INSERT INTO test_hmm SELECT hd_hmm_create(1) AS value;
select * from test_hmm;

--DROP TYPE gm CASCADE;
