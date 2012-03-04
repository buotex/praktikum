#include <iostream>
struct Match {
  size_t iQ_;
  size_t iP_;
  double fij_;
  Match(size_t iQ, size_t iP, double fij):iQ_(iQ), iP_(iP), fij_(fij) {}
};
struct EmdResult {
  size_t index1_;
  size_t index2_;
  std::vector<Match> fij_;
  double emd_;
};

  std::ostream & operator<< (std::ostream & os, const EmdResult & result) {
    os << "Earth Mover's Distance between " << result.index1_ << " and " << result.index2_ <<  " is equal to " << result.emd_ << std::endl;

    return os;
  }


template <typename T, int N>
struct NodeWithLocation {

  std::array<T, N> location_;

  NodeWithLocation(const std::array<T,N> &values) : location_(values ){}
  NodeWithLocation(const NodeWithLocation & node) : location_(node.location_){}
  NodeWithLocation() {}
};
struct EmdGraphPropertyBundle {
  std::string name_;
  std::vector<double> histogram_;
};



template <typename value_type>
struct BinLimits {
  value_type lowLim_;
  value_type binSize_;
  value_type highLim_;
};


