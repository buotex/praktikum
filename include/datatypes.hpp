#include <iostream>
#ifndef _DATATYPES_HPP__
#define _DATATYPES_HPP__
template<typename HistogramIndexType>
struct Match {
  HistogramIndexType histIndex1_;
  HistogramIndexType histIndex2_;
  double fij_;
  Match(HistogramIndexType iQ, HistogramIndexType iP, double fij):histIndex1_(iQ), histIndex2_(iP), fij_(fij) {}
};

std::ostream & operator<<(std::ostream & os, const std::pair<typename std::vector<Match<size_t> >::const_iterator, typename std::vector<Match<size_t> >::const_iterator> & matchiter) {
  os << "Top matches: " << "\n";
  std::for_each(matchiter.first, matchiter.second, [&](const Match<size_t> & match) {
    os << match.histIndex1_ << " -- " << match.histIndex2_ << " : " <<  match.fij_ << "\t";
    });
  os << "\n";
  return os;
}

template <typename Graph, typename HistogramIndexType>
struct EmdResult {
  Graph * g1_;
  Graph * g2_;

  std::vector<Match<HistogramIndexType> > fijVector_;
  double emd_;
  
  typedef typename std::vector<Match<HistogramIndexType> >::const_iterator IterType;
  std::pair<IterType, IterType> getMaxElements(size_t i) {
    std::size_t num = std::min(i, fijVector_.size()); 
    std::nth_element(fijVector_.begin(), fijVector_.begin() + num, fijVector_.end(), [] (const Match<HistogramIndexType> & match1, const Match<HistogramIndexType> & match2){ return match1.fij_ > match2.fij_;});
    return std::make_pair(fijVector_.cbegin(), fijVector_.cbegin() + num);
  }
};

template <typename Graph, typename HistogramIndexType>
  std::ostream & operator<< (std::ostream & os, const EmdResult<Graph, HistogramIndexType> & result) {
    os << "Earth Mover's Distance between " << result.g1_ << " and " << result.g2_ <<  " is equal to " << result.emd_ << std::endl;
    return os;
  }


template <typename T, int N>
struct NodeWithLocation {

  std::array<T, N> location_;

  NodeWithLocation(const std::array<T,N> &values) : location_(values ){}
  NodeWithLocation(const NodeWithLocation & node) : location_(node.location_){}
  NodeWithLocation() {}
};
template <typename HistogramType>
struct EmdGraphPropertyBundle {
  std::string name_;
  HistogramType histogram_;
};

template <typename value_type>
struct BinLimits {
  value_type lowLim_;
  value_type binSize_;
  value_type highLim_;
  BinLimits(const value_type lowLim, const value_type binSize, const value_type highLim): lowLim_(lowLim), binSize_(binSize), highLim_(highLim) {}
  BinLimits() {}
};

#endif
