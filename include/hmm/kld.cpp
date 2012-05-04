extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}
#include <kld.hpp>
extern "C" {
  double
    compareHMMs(
        GM_c* gms1, int * ids1, double * weights1, int gm_n1, double * transitions1, double * inits1, int state_n1,
        GM_c* gms2, int * ids2, double * weights2, int gm_n2, double * transitions2, double * inits2, int state_n2) {
      double similarity = -1;
      try{
        HMM hmm1 = HMM(gms1, ids1, weights1, gm_n1, transitions1, inits1, state_n1);
        HMM hmm2 = HMM(gms2, ids2, weights2, gm_n2, transitions2, inits2, state_n2);
        std::stringstream temp;
        //hmm1.print(temp);
        //hmm2.print(temp);
        //elog(INFO, "%s", temp.str().c_str());
        
  
        similarity = HMMComp::sMrandomWalk(hmm1, hmm2);
        //elog(INFO, "BLUB %f", similarity);
      }
      catch(const std::exception& e) { elog(INFO, "%s", e.what());}
      catch(...) {
      
      }

      return similarity;
    }

  
  

}
