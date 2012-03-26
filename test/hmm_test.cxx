#include "unittest.hxx"
#include <hmm/gmm.hpp>
#include <hmm/hmm.hpp>
#include <hmm/chinesewhisper.hpp>
#include <hmm/kmeans.hpp>
//#include <ctime>


using namespace vigra;
struct HMMTestSuite : vigra::test_suite {
  HMMTestSuite() : vigra::test_suite("HMM") {
    add(testCase(&HMMTestSuite::testKmeans));
    add(testCase(&HMMTestSuite::testGm));
    add(testCase(&HMMTestSuite::testGmm));
    add(testCase(&HMMTestSuite::testHmm));
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
    unsigned int kmax = 5;
    unsigned int n = 100;

    arma::mat A = 10 *  arma::randn(2, n);
    arma::mat B = 20 *  arma::randn(2, n) + 100;
    arma::mat testData = join_rows(A,B);
    auto labels = kmeansWithSubset(testData, 1, 100, 1);
    //create gmm for all labels
    unsigned int numLabels = (unsigned int) arma::as_scalar(arma::max(labels)) + 1;
    std::vector<GMM> mixtureModels;
    std::vector<arma::mat> dataSubsets(numLabels);;
    assert(testData.n_cols == labels.n_elem);
    arma::uvec allDims = arma::uvec(testData.n_rows);
    for (unsigned int i = 0; i < testData.n_rows; ++i){
      allDims(i) = i;
    }

    for (unsigned int i = 0; i < numLabels; ++i){
      arma::uvec indices = arma::find(labels == i);
      mixtureModels.push_back(GMM(testData, kmin, kmax));
    }
//DEBUGGING
arma::vec newMu = arma::randn(2, 1);
arma::mat newSigma = 100 * arma::randn(2,2) + 500;
newMu.print("newMu");
newSigma.print("newSigma");
std::cout << arma::det(newSigma);
    mixtureModels[0].updateGM(0, 0.2, newMu, newSigma);
    mixtureModels[0].normalizeWeights();
    mixtureModels[0].print("blub");
    hmm.baumWelch(testData, mixtureModels);
    hmm.print();
  }

};

int main() {
  HMMTestSuite test;
  int success = test.run();
  std::cout << test.report() << std::endl;
  return success;


}
