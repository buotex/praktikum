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
  //HMM hmm, hmm2, hmm3, hmm4, hmm5;

  //define the number of clusters to be defined in kmeans
  unsigned int numClusters = 3;
  KMeansFunctor kmeansFunctor(numClusters);

  //define min and max values for the number of Gaussian Models that should be used when creating a GMM
  unsigned int kmin = 1;
  unsigned int kmax = 1;
  GMMCreator creator(kmin, kmax);
  
  

  //load these protein files as a base for our experiments
  HMM hmm1 = buildHMM(parsePdb("data/2HDZ.pdb"), creator, kmeansFunctor);
  HMM hmm2 = buildHMM(parsePdb("data/3F27.pdb"), creator, kmeansFunctor);

  //How robust is our algorithm regarding to shuffling? Keep in mind, that HMMs are time-dependent and will change when
  //you change the order of the input data.
  std::function<arma::mat (const arma::mat &)> shuffling = 
  [] (const arma::mat &data) { return arma::shuffle(data, 1);};
  HMM hmm3 = buildHMM(parsePdb("data/3F27.pdb"), creator, kmeansFunctor.bind(shuffling));
  
  
  //Let's try a transformation to our data: Extract it, then rotate it around the Z and X axis according to the values
  
  std::function<arma::mat (const arma::mat &)> rot = 
  std::bind(rotX, std::bind(rotZ, std::placeholders::_1, 0.7 * M_PI), 0.3 * M_PI);
  //Our kmeansFunctor spawns another version of itself, just now with the added functor to transform the data
  //The default functor is just a identity transformation
  HMM hmm4 = buildHMM(parsePdb("data/3F27.pdb"), creator, kmeansFunctor.bind(rot));


 std::cout << "Old algorithm" << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm1, hmm1) << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm1, hmm2) << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm3, hmm2) << std::endl;
  std::cout << HMMComp::symmetric_kld(hmm4, hmm2) << std::endl;
 
  std::cout << "New algorithm" << std::endl;
  std::cout << HMMComp::sMrandomWalk(hmm1, hmm1) << std::endl;
  //std::cin.get();
  std::cout << HMMComp::sMrandomWalk(hmm1, hmm2) << std::endl;
  
  std::cout << HMMComp::sMrandomWalk(hmm3, hmm2) << std::endl;
  //std::cin.get();
  std::cout << HMMComp::sMrandomWalk(hmm2, hmm4) << std::endl;
 

  std::cout << "Transformations were made to the original data - transform back!";
  //We will now try to reconstruct the linear transformation made before, to enable a better matching
  arma::mat transformation1 = HMMComp::findTransformationMatrix(hmm2, hmm4);
  arma::cube transformation2 = HMMComp::findTransformationCube(hmm2, hmm4);

  MatrixTransformationFunctor mtf1(transformation1);
  std::cout << HMMComp::sMrandomWalk(hmm2, hmm4, mtf1) << std::endl;
  //Just try the most likely coordinate transformation, though there are others.
  MatrixTransformationFunctor mtf2(transformation2.slice(0));
  std::cout << HMMComp::sMrandomWalk(hmm2, hmm4, mtf2) << std::endl;


// =================================================================================//

  ProteinChainFunctor pFunctor;
  //Let's now create 2 hmms and label them via their chains
  HMM hmm5 = buildHMM(parsePdb("data/2HDZ.pdb"), creator, pFunctor);
  HMM hmm6 = buildHMM(parsePdb("data/3F27.pdb"), creator, pFunctor);


  std::cout << HMMComp::sMrandomWalk(hmm5, hmm6) << std::endl;
}



