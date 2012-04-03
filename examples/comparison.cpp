#include <pdb_utilities.hpp>
#include <hmm/hmm.hpp>
#include <hmm/kld.hpp>
#include <hmm/gmm.hpp>
#include <hmm/kmeans.hpp>
#include <iostream>
int main() {
  HMM hmm, hmm2;

  unsigned int kmin = 3;
  unsigned int kmax = 15;
  arma::mat data = createMatrix("data/2HDZ.pdb");
  arma::mat data2 = createMatrix("data/3F27.pdb");
  auto labels = kmeansWithSubset(data, 5, 100, 0);
  auto labels2 = kmeansWithSubset(data2, 5, 100, 0);


  hmm.createGMM(data, labels, kmin, kmax);
  hmm2.createGMM(data2, labels2, kmin, kmax);

  hmm.baumWelchCached(data);
  hmm2.baumWelchCached(data2);
  std::cout << HMMComp::symmetric_kld(hmm, hmm2) << std::endl;


}



