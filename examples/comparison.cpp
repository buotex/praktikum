#include <pdb_utilities.hpp>
#include <hmm/hmm.hpp>
#include <hmm/kld.hpp>
#include <hmm/gmm.hpp>
#include <hmm/kmeans.hpp>
#include <iostream>
#include <cmath>
arma::mat
rotX(const arma::mat & data, double phi) {
  arma::mat rotX = arma::zeros(3,3);
  rotX(0,0) = 1;
  rotX(1,1) = rotX(2,2) = std::cos(phi);
  rotX(2,1) = std::sin(phi);
  rotX(1,2) = - std::sin(phi);

  return rotX * data;
}

arma::mat
rotY(const arma::mat & data, double phi) {
  arma::mat rotY = arma::zeros(3,3);
  rotY(1,1) = 1;
  rotY(0,0) = rotY(2,2) = std::cos(phi);
  rotY(2,0) = std::sin(phi);
  rotY(0,2) = - std::sin(phi);

  return rotY * data;
}

arma::mat
rotZ(const arma::mat & data, double phi) {
  arma::mat rotZ = arma::zeros(3,3);
  rotZ(2,2) = 1;
  rotZ(1,1) = rotZ(0,0) = std::cos(phi);
  rotZ(1,0) = std::sin(phi);
  rotZ(0,1) = - std::sin(phi);

  return rotZ * data;
}
int main() {
  HMM hmm, hmm2, hmm3, hmm4, hmm5;

  unsigned int kmin = 1;
  unsigned int kmax = 1;
  unsigned int clusternumber = 3;
  double eps = 1E-4;
  arma::mat data = createMatrix("data/2HDZ.pdb");
  arma::mat data2 = createMatrix("data/3F27.pdb");
  //arma::mat data3 = data2;
  //data3.swap_cols(0,1);
  arma::mat data3 = arma::shuffle(data2, 1);
  arma::mat data4 = rotY(rotX(data2, 0.7 * M_PI), 0.2 * M_PI);
  auto labels = kmeans(data, clusternumber, 100, 0);
  auto labels2 = kmeans(data2, clusternumber, 100, 0);
  auto labels3 = kmeans(data3, clusternumber, 100, 0);
  auto labels4 = kmeans(data4, clusternumber, 100, 0);
  hmm.setEps(eps);
  hmm2.setEps(eps);
  hmm3.setEps(eps);
  hmm4.setEps(eps);

  hmm.createGMM(data, labels, kmin, kmax);
  hmm2.createGMM(data2, labels2, kmin, kmax);
  hmm3.createGMM(data3, labels3, kmin, kmax);
  hmm4.createGMM(data4, labels4, kmin, kmax);

  hmm.baumWelchCached(data);
  hmm2.baumWelchCached(data2);
  hmm3.baumWelchCached(data3);
  hmm4.baumWelchCached(data4);
  //hmm.sort(0);
  //hmm2.sort(0);
  //hmm3.sort(0);
  //hmm4.sort(0);

  //hmm2.print("hmm2");
  //std::cout << "SPLIT";
  //hmm3.print("hmm3");
  std::cout << "RESULTS" << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm, hmm) << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm, hmm2) << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm3, hmm2) << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm4, hmm2) << std::endl;

  //std::cout << HMMComp::sMrandomWalk(hmm, hmm) << std::endl;
  //std::cin.get();
  //std::cout << HMMComp::sMrandomWalk(hmm, hmm2) << std::endl;
  //std::cin.get();
  //std::cout << HMMComp::sMrandomWalk(hmm3, hmm2) << std::endl;
  //std::cin.get();
  //std::cout << HMMComp::sMrandomWalk(hmm4, hmm2) << std::endl;
  arma::mat trans = arma::inv(rotY(rotX(arma::eye(3,3), 0.4 * M_PI), 0.1 * M_PI));
  arma::mat transformation1 = HMMComp::findTransformationMatrix(hmm2, hmm4);
  arma::cube transformation2 = HMMComp::findTransformationMatrix2(hmm2, hmm4);
  arma::cube transformation3 = HMMComp::findTransformationMatrix2(hmm, hmm2);
  trans.print("origTrans");
  transformation1.print("trans1");
  transformation2.print("trans2");


  MatrixTransformationFunctor mtf1(transformation1);
  std::cout << HMMComp::sMrandomWalk(hmm2, hmm4, mtf1) << std::endl;
  for (unsigned int i = 0; i < transformation2.n_slices; ++i) {
    MatrixTransformationFunctor mtf2(transformation2.slice(i));
    std::cout << HMMComp::sMrandomWalk(hmm2, hmm4, mtf2) << std::endl;
  }
  for (unsigned int i = 0; i < transformation3.n_slices; ++i) {
    MatrixTransformationFunctor mtf3(transformation3.slice(i));
    std::cout << HMMComp::sMrandomWalk(hmm, hmm2, mtf3) << std::endl;
  }





}



