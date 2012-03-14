#include <array>
#include <armadillo>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES
#endif

template <size_t N>
class GM {
  
  arma::vec mu_;
  arma::mat sigma_;
  arma::mat invSigma_;

  double coeff_; 
  
 public:

  double getProb(const arma::vec & loc) {
    arma::vec diff = loc - mu_;
    double exponent = -0.5 * diff.t() * invSigma_ * diff; //TODO: check how transpose works
    return coeff * std::exp(exponent);
  }

  void
  setSigma(const arma::mat & sigma) {
    sigma_ = sigma;
    invSigma_ = arma::inv(sigma_);
    coeff_ = 1./std::sqrt(std::pow(2. * M_PI, N) * arma::det(sigma_));
  }
  void 
  setMu_(const arma::vec & mu) {
    mu_ = mu;
  }
};


class GMM {
  //a vector of gaussian models
  std::vector<GM> mixture_;
  public:
  //implementation of the EM-algorithm
  void
      //TODO:perhaps use a matrix for the data?
  findMaxLikelihood(const arma::mat& data) {

  }

};



