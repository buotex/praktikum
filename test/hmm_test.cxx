#include "unittest.hxx"
#include <hmm/gmm.hpp>
#include <hmm/hmm.hpp>
#include <hmm/chinesewhisper.hpp>
#include <hmm/kmeans.hpp>
#include <hmm/kld.hpp>
//#include <ctime>


using namespace vigra;
struct HMMTestSuite : vigra::test_suite {
  HMMTestSuite() : vigra::test_suite("HMM") {
    add(testCase(&HMMTestSuite::testKmeans));
    add(testCase(&HMMTestSuite::testGm));
    add(testCase(&HMMTestSuite::testGmm));
    add(testCase(&HMMTestSuite::testHmm));
    add(testCase(&HMMTestSuite::testKld));
  }

  void testKmeans() {
    unsigned int n = 5000;
    arma::mat testData = 100 *  arma::randn(2, n);
    kmeansWithSubset(testData, 5,100, 1); 
  
  }
  void testGm() {
  
    unsigned int n = 5000;
    arma::mat testData = 100 *  arma::randn(1, n);
    GM gm(testData);
    //gm.print("gm");
    //testData.print("testData");
    arma::vec u(n);
    arma::mat cov = arma::cov(testData.t());
    //cov.print("cov");

    for (unsigned int i = 0; i < n; ++i) {
      u(i) = gm.getProb(testData.col(i));  
    }
    //u.print("u");
  }
  void testGmm() {

    GMM gmm; 
    using arma::join_cols;
    arma::mat A = arma::randn(1, 500);
    arma::mat B = 2 * arma::randn(1, 500);
    arma::mat C = arma::randn(1, 300) + 50;
    arma::mat D = arma::randn(1, 100) + 70;
    arma::mat E = 3 * arma::randn(1, 400) + 500;
    arma::mat testData = join_rows(E, join_rows(D, join_rows(C, join_rows(A,B))));
    //unsigned int blub = time(NULL);
    //std::cout << "seed" << blub << std::endl;
    //srand(blub);
    gmm.findMaxLikelihood(testData, 3, 15);
    GMM gmm2 = gmm;
    //gmm2.print("gmm_results");
  }

  void testHmm() {
    HMM hmm;

    unsigned int kmin = 1;
    unsigned int kmax = 2;
    unsigned int n = 500;

    arma::mat A = 10 *  arma::randn(1, n);
    arma::mat B = 20 *  arma::randn(1, n) + 100;
    arma::mat testData = join_rows(A,B);
    auto labels = kmeansWithSubset(testData, 2, 100, 1);
    //create gmm for all labels
    unsigned int numLabels = (unsigned int) arma::as_scalar(arma::max(labels)) + 1;
    std::vector<GMM> mixtureModels;

    for (unsigned int i = 0; i < numLabels; ++i){
      arma::uvec indices = arma::find(labels == i);
      mixtureModels.push_back(GMM(testData.cols(indices), kmin, kmax));
    }
    //DEBUGGING
    arma::vec newMu = arma::randn(1, 1);
    arma::mat newSigma = 100 * arma::randn(1,1) + 500;
    newMu.print("newMu");
    newSigma.print("newSigma");
    std::cout << arma::det(newSigma);
    mixtureModels[0].updateGM(0, 0.2, newMu, newSigma);
    mixtureModels[0].normalizeWeights();
    mixtureModels[0].print("blub");
    hmm.baumWelch(testData, mixtureModels);
    hmm.print();
  }

  void
    testKld() {
      HMM hmm,hmm2, hmm3;
      unsigned int kmin = 1;
      unsigned int kmax = 1;
      unsigned int n = 5000;

      arma::mat A = 10 *  arma::randn(2, n);
      arma::mat B = 20 *  arma::randn(2, n) + 100;
      arma::mat C = 20 *  arma::randn(2, n) + 500;
      arma::mat D = 10 *  arma::randn(2, n);

      arma::mat testData = join_rows(A,B);
      arma::mat testData2 = join_rows(C,D);
      auto labels = kmeansWithSubset(testData, 2, 100, 0);
      auto labels2 = kmeansWithSubset(testData2, 2, 100, 0);

      hmm.createGMM(testData, labels, kmin, kmax);
      hmm2.createGMM(testData2, labels2, kmin, kmax);
      hmm3.createGMM(testData, labels, kmin, kmax);

      hmm.baumWelch(testData);
      hmm2.baumWelch(testData2);
      hmm3.baumWelchCached(testData);

      hmm.sort(0);
      hmm2.sort(0);
      hmm3.sort(0);
      std::cout << HMMComp::symmetric_kld(hmm, hmm2) << std::endl;
      std::cout << HMMComp::symmetric_kld(hmm, hmm) << std::endl;
      std::cout << HMMComp::symmetric_kld(hmm, hmm3) << std::endl;

    }
};

int main() {
  HMMTestSuite test;
  int success = test.run();
  std::cout << test.report() << std::endl;
  return success;


}
