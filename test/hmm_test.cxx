#include "unittest.hxx"
#include <hmm/gmm.hpp>
#include <hmm/hmm.hpp>
#include <hmm/chinesewhisper.hpp>
#include <hmm/kmeans.hpp>
void
handler()
{
    void *trace_elems[20];
    int trace_elem_count(backtrace( trace_elems, 20 ));
    char **stack_syms(backtrace_symbols( trace_elems, trace_elem_count ));
    for ( int i = 0 ; i < trace_elem_count ; ++i )
    {
        std::cout << stack_syms[i] << "\n";
    }
    free( stack_syms );

    exit(1);
}   



using namespace vigra;
struct HMMTestSuite : vigra::test_suite {
  HMMTestSuite() : vigra::test_suite("HMM") {
    add(testCase(&HMMTestSuite::testGm));
    add(testCase(&HMMTestSuite::testGmm));
    add(testCase(&HMMTestSuite::testHmm));
  }

  void testGm() {
  
    unsigned int n = 5000;
    arma::mat testData = 100 *  arma::randn(1, n);
    GM gm(testData);
    gm.print("gm");
    //testData.print("testData");
    arma::vec u(n);
    arma::mat cov = arma::cov(testData.t());
    cov.print("cov");

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

    gmm.findMaxLikelihood(testData, 3, 15);
    GMM gmm2 = gmm;
    gmm2.print("gmm_results");
  }

void testHmm() {
  HMM hmm;
}

};

int main() {
  HMMTestSuite test;
  int success = test.run();
  std::cout << test.report() << std::endl;
  return success;


}
