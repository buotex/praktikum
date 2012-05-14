extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}
#include <chinesewhisper.hpp>
#include <kmeans.hpp>
#include <kld.hpp>
arma::urowvec
normalizeLabels(arma::urowvec labels) {
  std::vector<unsigned int> sorted_labels(labels.n_elem);
  std::copy(labels.begin(), labels.end(), sorted_labels.begin());
  std::sort(sorted_labels.begin(), sorted_labels.end());
  std::vector<unsigned int>::iterator it = std::unique(sorted_labels.begin(), sorted_labels.end());
  sorted_labels.resize(it - sorted_labels.begin());

  std::map<unsigned int, unsigned int> labelmapping;
  for (size_t i = 0; i < sorted_labels.size(); ++i) {
    labelmapping[sorted_labels[i]] = i;  
  }

  for (unsigned int i = 0; i < labels.n_elem; ++i) {
    labels(i) = labelmapping[labels(i)];
  }
  return labels;
}

extern "C" {
  void * 
    constructHMM(double * matrix, int ndata, int nclusters, int * edges, int nedges) {
      HMM * hmm = 0;
      try {

        //elog(INFO, "wtf");
        KMeansParams params(nclusters);
        //temp << nclusters;
        //elog(INFO, temp.str().c_str());
        const arma::mat data(matrix, 3, ndata);
        arma::urowvec labels;
        if (nclusters > 0) {  
          labels = kmeans(data, params);
        }
        else {
          int nsteps = -nclusters;
          const arma::umat adjlist((unsigned int *) edges, 2, nedges);
          labels = chinesewhisper(data, adjlist, nsteps); 
        }

        hmm = new HMM;
        labels = normalizeLabels(labels);
        GMMCreator creator(1,5);
        std::vector<GMM> models = creator(data, labels);
        hmm->baumWelchCached(data, models);
        //hmm->print(temp);
        //elog(INFO, temp.str().c_str());
      }
      catch(const std::exception& e) { 
        elog(INFO, "%s", e.what());
      }
      catch (...)
      {
        elog(INFO, "hd_hmm_create: constructHMM has thrown an error");
      }
      return hmm;
    }


  void
    deleteHMM( void * hmm ) {
      delete (HMM *) hmm;
    }

  void *
    getModels( void * hmm_, int * nelemsp) {
      HMM * hmm = (HMM *) hmm_;
      *nelemsp = hmm->countGMs();

      Datum * element = (Datum*)palloc((*nelemsp) * sizeof(Datum));

      const std::vector<GMM> & models = hmm->getModels();
      int counter = 0; 
      for (size_t i = 0; i < models.size(); ++i) {
        for (size_t j = 0; j < models[i].n_gm(); ++j) {
          GM_c * gm_c = (GM_c*) palloc(sizeof(GM_c));
          const GM & gm = models[i].getGM(j);
          const GM_c & gmc = gm.getGM_c();
          memcpy(gm_c, &gmc, sizeof(GM_c));
          element[counter] = PointerGetDatum(gm_c);
          ++counter;
        }
      }

      return element;

    }
  void *
    getModelIds( void * hmm_, int * nelemsp) {
      HMM * hmm = (HMM *) hmm_;
      *nelemsp = hmm->countGMs();

      Datum * element = (Datum*)palloc((*nelemsp) * sizeof(Datum));
      const std::vector<GMM> & models = hmm->getModels();
      int counter = 0;
      for (size_t i = 0; i < models.size(); ++i) {
        for (size_t j = 0; j < models[i].n_gm(); ++j) {
          element[counter] = Int32GetDatum((int)i);
          ++counter;
        }
      } 
      return element;
      //fill up models

    }


  void *
    getWeights( void * hmm_, int * nelemsp) {
      HMM * hmm = (HMM *) hmm_;
      *nelemsp = hmm->countGMs();

      Datum * element = (Datum*)palloc((*nelemsp) * sizeof(Datum));

      const std::vector<GMM> & models = hmm->getModels();
      int counter = 0; 
      for (size_t i = 0; i < models.size(); ++i) {
        arma::vec weights = models[i].getWeights();
        for (size_t j = 0; j < models[i].n_gm(); ++j) {
          element[counter] = Float8GetDatum(weights(j));
          ++counter;
        }
      }
      return element;
    }

  void *
    getTransitions( void * hmm_, int * nelemsp) {
      HMM * hmm = (HMM *) hmm_;
      *nelemsp = hmm->getN() * hmm->getN();
      Datum * element = (Datum*)palloc((*nelemsp) * sizeof(Datum));
      arma::mat A = hmm->getTransitions();
      for (int i = 0; i < *nelemsp; ++i) {
        element[i] = Float8GetDatum(A(i));
      }
      return element;
    }

  void *
    getInits( void * hmm_, int * nelemsp) {
      HMM * hmm = (HMM *) hmm_;
      *nelemsp = hmm->getN();
      Datum * element = (Datum*)palloc((*nelemsp) * sizeof(Datum));
      arma::rowvec pi = hmm->getInits();
      for (int i = 0; i < *nelemsp; ++i) {
        element[i] = Float8GetDatum(pi(i));
      }
      return element;
    }
  double
    compareHMMs(

        int tryLinearTransform,
        GM_c* gms1, int * ids1, double * weights1, int gm_n1, double * transitions1, double * inits1, int state_n1,
        GM_c* gms2, int * ids2, double * weights2, int gm_n2, double * transitions2, double * inits2, int state_n2) {
      double similarity = -1;
      try{
        HMM hmm1 = HMM(gms1, ids1, weights1, gm_n1, transitions1, inits1, state_n1);
        HMM hmm2 = HMM(gms2, ids2, weights2, gm_n2, transitions2, inits2, state_n2);
        std::stringstream temp;

        switch(tryLinearTransform) {
          case 0:
            {
              similarity = HMMComp::sMrandomWalk(hmm1, hmm2);
            }
            break;
          case 1:
            {
              arma::cube transformation = HMMComp::findTransformationCube(hmm1, hmm2);
              double maxsim = 0.0;
              for (int i = 0; i < transformation.n_slices; ++i) {

                MatrixTransformationFunctor mtf(transformation.slice(i));
                maxsim = std::max(maxsim, HMMComp::sMrandomWalk(hmm1, hmm2, mtf));
              }
              similarity = maxsim; 
            }
        }

      }
      catch(const std::exception& e) { elog(INFO, "%s", e.what());}
      catch(...) {

      }

      return similarity;
    }




}
