#include <array>
#include <armadillo>
#ifndef __INCLUDE_HMM_KMEANS_HPP__
#define __INCLUDE_HMM_KMEANS_HPP__

/** 
 *  Pack of parameters for Kmeans
 *
 * */
struct KMeansParams {
  size_t numClusters_;
  size_t maxIterations_;
  size_t purgingThreshold_;

  KMeansParams(size_t nC, size_t mI = 100, size_t pT = 1): numClusters_(nC), maxIterations_(mI), purgingThreshold_(pT){}

};



arma::urowvec
kmeansLoop(const arma::mat & data, arma::mat & means, KMeansParams params) {
  size_t maxIterations = params.maxIterations_;
  size_t purgingThreshold = params.purgingThreshold_;
  double checksum = 0;
  double oldChecksum = 0;
  size_t counter = 0;
  unsigned int numData = data.n_cols;
  arma::urowvec labels = arma::urowvec(numData);

  arma::mat meanCounter(1,means.n_cols);
  
  while(1) {
    for (unsigned int i = 0; i < data.n_cols; ++i) {
      arma::mat differences = data.col(i) * arma::ones(1, means.n_cols) - means;
      arma::rowvec norms = arma::zeros(1, means.n_cols);
      for (unsigned int j = 0; j < means.n_cols; ++j) {
        norms(j) = arma::norm(differences.col(j), 2);
      }
      double * smallestElem = std::min_element(norms.begin(), norms.end());
      unsigned int label = (unsigned int)(smallestElem - norms.begin());
      labels(i) = label;
    }

    checksum = sum(labels);
    if (checksum == oldChecksum || counter == maxIterations) break;
    ++counter;
    oldChecksum = checksum;

    means.fill(0);
    meanCounter.fill(0);

    for (unsigned int i = 0; i < data.n_cols; ++i) {
      unsigned int label = labels(i);
      means.col(label) += data.col(i);
      meanCounter(label) += 1;
    }
    arma::uvec tinyClusters = arma::find(meanCounter > (double)purgingThreshold);
    if (tinyClusters.n_elem != means.n_cols) {
      try{
        meanCounter.print("meanCounter");
        means.print("means");
        means = means.cols(tinyClusters);
        meanCounter = meanCounter.cols(tinyClusters);
        meanCounter.print("meanCounter");
        means.print("means");
      }
      catch(const std::logic_error & e) {
        std::cout << "Purging :" << (double) purgingThreshold << std::endl;
        meanCounter.print("meanCounter");
        means.print("means");
        tinyClusters.print("tiny");
        std::cout << "size :" << means.n_cols << std::endl;
        std::cout << "vecsize :" << meanCounter.n_elem << std::endl;
        throw e;
      }
    }
    means /= (arma::ones(data.n_rows, 1) * meanCounter);
  }

  return labels;
}

/** Kmeans algorithm, starting with randomized labels for every datapoint */
arma::urowvec
kmeansRandom(const arma::mat & data, arma::mat & means, KMeansParams params) {

  arma::urowvec labels(data.n_cols);
  unsigned int numMaxClusters = means.n_cols;
  arma::mat meanCounter(means.n_rows, numMaxClusters);


  std::mt19937 rSeedEngine;
  typedef std::uniform_int_distribution<unsigned int> Distribution;

  std::for_each(labels.begin(), labels.end(), [&] (unsigned int & label) {label = Distribution(0, numMaxClusters-1)(rSeedEngine);});

  means.fill(0);
  meanCounter.fill(0);

  for (unsigned int i = 0; i < data.n_cols; ++i) {
    unsigned int label = labels(i);
    means.col(label) += data.col(i);
    (meanCounter.col(label)) += arma::ones(data.n_rows, 1);
  }
  means /= meanCounter;

  return kmeansLoop(data, means, params);

}
/** KMeans algorithm, doing a run with a small subset of the data and using the resulting means as a starting point*/
arma::urowvec
kmeansWithSubset(const arma::mat & data, KMeansParams params) {

  unsigned int numClusters = (unsigned int) params.numClusters_;

  unsigned int vecSize = data.n_cols;
  arma::mat subset(data.n_rows, (unsigned int)sqrt(vecSize));
  std::mt19937 rSeedEngine;
  typedef std::uniform_int_distribution<unsigned int> Distribution;
  for (unsigned int i = 0; i < subset.n_cols; ++i) {
    unsigned int randIndex = Distribution(0, vecSize-1)(rSeedEngine);
    subset.col(i) = data.col(randIndex);
  }
  arma::mat means = arma::mat(data.n_rows, numClusters);
  kmeansRandom(subset,means, params);
  return kmeansLoop(data, means, params);

}

/** Default KMeans algorithm, picking random datapoints as starting means*/
arma::urowvec
kmeans(const arma::mat & data, KMeansParams params) {
  unsigned int numClusters = (unsigned int) params.numClusters_;
  unsigned int vecSize = data.n_cols;
  std::mt19937 rSeedEngine;
  typedef std::uniform_int_distribution<unsigned int> Distribution;
  arma::mat means = arma::mat(data.n_rows, numClusters);
  for (unsigned int i = 0; i < means.n_cols; ++i) {
    unsigned int randIndex = Distribution(0, vecSize-1)(rSeedEngine);
    means.col(i) = data.col(randIndex);
  }
  return kmeansLoop(data, means, params);
}


#endif
