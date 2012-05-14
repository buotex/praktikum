#ifndef __INCLUDE_HMM_CHINESEWHISPER_HPP
#define __INCLUDE_HMM_CHINESEWHISPER_HPP

#include <vector>
#include <map>
#include <armadillo>
#include "prettyprint.hpp"
arma::urowvec 
chinesewhisper(const arma::mat & data, const arma::umat & edges, int numSteps) {
  unsigned int N = data.n_cols;
  arma::urowvec labels(N);
  std::vector<std::size_t>descriptors(N);

  std::vector<std::vector<unsigned int> >adjlist(N);
  for (unsigned int i = 0; i < edges.n_cols; ++i) {
    adjlist[(std::size_t) edges(0,i)-1].push_back(edges(1,i)-1);
    adjlist[(std::size_t) edges(1,i)-1].push_back(edges(0,i)-1);
  }

  for (std::size_t i = 0; i < N; ++i) {
    labels(i) = i;
    descriptors[i] = i;
  }
  for (int step = 0; step < numSteps; ++step) {
    std::random_shuffle(descriptors.begin(), descriptors.end());
    for (std::size_t i = 0; i < N; ++i) {
      std::map<unsigned int, unsigned int> labelcounts;  
      std::size_t currNode = descriptors[i];
      const std::vector<unsigned int> & curredgelist = adjlist[currNode];
      if (curredgelist.size() == 0) continue;
      for (std::size_t j = 0; j < curredgelist.size(); ++j) {
        ++labelcounts[labels(curredgelist[j])];
      }

      typedef std::pair<unsigned int, unsigned int> uintpair;
      std::vector<uintpair> labelcountsvector(labelcounts.begin(), labelcounts.end());

std::sort(labelcountsvector.begin(), labelcountsvector.end(), [](uintpair a, uintpair b){return a.second < b.second;});
unsigned int maxVal = labelcountsvector.back().second;
std::vector<uintpair>::iterator it = std::find_if(labelcountsvector.begin(), labelcountsvector.end(), [maxVal](uintpair a) {return a.second == maxVal;});
      std::random_shuffle(it, labelcountsvector.end());

      unsigned int newLabel = labelcountsvector.back().first;
      labels(i) = newLabel;
    }
  }
  return labels;
}



#endif
