#include "unittest.hxx"

#include <pdb_utilities.hpp>

using namespace vigra;
struct PDBTestSuite : vigra::test_suite {
  PDBTestSuite() : vigra::test_suite("PDB") {
    add(testCase(&PDBTestSuite::testPdbInput));
  }

  void testPdbInput() {
    arma::mat data = createMatrix("data/2HDZ.pdb");
    arma::mat data2 = createMatrix("data/3F27.pdb");
  }
};
int main() {
  PDBTestSuite test;
  int success = test.run();
  std::cout << test.report() << std::endl;
  return success;

}
