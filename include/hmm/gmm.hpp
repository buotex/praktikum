#include <algorithm>
#include <armadillo>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES
#endif

#ifndef __INCLUDE_HMM_GMM_HPP__
#define __INCLUDE_HMM_GMM_HPP__
bool hasZeros(arma::mat & data) {
  arma::uvec temp = arma::find(data);
  return temp.n_elem != data.n_elem; 
}
class GM {
  
  arma::vec mu_;
  arma::mat sigma_;
  arma::mat invSigma_;

  double coeff_; 
  
 public:

  double getProb(const arma::vec & loc) {
    arma::vec diff = loc - mu_;
    double exponent = -0.5 * arma::as_scalar(diff.t() * invSigma_ * diff); //TODO: check how transpose works
    return coeff_ * std::exp(exponent);
  }

  void
  setSigma(const arma::mat & sigma) {
    unsigned int d = sigma.n_rows;
    sigma_ = sigma;
    invSigma_ = arma::inv(sigma_);
    coeff_ = 1./std::sqrt(std::pow(2. * M_PI, d) * arma::det(sigma_));
  }
  void 
  setMu(const arma::vec & mu) {
    mu_ = mu;
  }

  //create gaussian model with the highest loglikelyhood for the given data
  GM(const arma::mat& data) {
    setMu(arma::conv_to<arma::vec>::from(arma::mean(data, 1)));
    setSigma(arma::cov(data.t()));
  }
  GM() {}
  void print(std::string header = "") {
    std::cout << header << std::endl;
    mu_.print("Mean vector");
    sigma_.print("Sigma");
  }
};

class GMM {
  //a vector of gaussian models
  std::vector<GM> mixture_;
  arma::vec weights_;
  public:
  void
  updateGM(unsigned int index, double weight, const arma::vec & mu, const arma::mat & sigma) {
    weights_[index] = weight;
    mixture_[index].setMu(mu);
    mixture_[index].setSigma(sigma);
  }
  size_t
  getNumComponents() {
    return mixture_.size();
  }
  void print(std::string header = "") {
  std::cout << header << std::endl;
  arma::uvec nonZero = arma::find(weights_);
  std::for_each(nonZero.begin(), nonZero.end(), [this](unsigned int index) {
    std::cout << "weight: " << weights_(index) << std::endl;
    mixture_[index].print("mix");
  });
  }
 
 double getProb(const arma::vec & datum, unsigned int start, unsigned int end) {
    double prob = 0.;
    for(unsigned int i = start; i < end; ++i) {
      prob += weights_(i) * mixture_[i].getProb(datum);
    }
    return prob;
 }
double getProb(const arma::vec & datum, unsigned int index) {
  return getProb(datum, index, index + 1);
  }
double getProb(const arma::vec & datum) {
  return getProb(datum, 0, (unsigned int)mixture_.size());
  }


  //implementation of the EM-algorithm
  void
      //TODO:perhaps use a matrix for the data?
      //k == number of mixtures
  findMaxLikelihood(const arma::mat& data, unsigned int kmin, unsigned int kmax) {
    
    unsigned int n = data.n_cols; //number of data elements;
    unsigned int d = data.n_rows; //dimensionality of the data
    unsigned int N = d * (d+1)/2;
    double eps = 1E-5;

    unsigned int knz = kmax;
    double Lmin = std::numeric_limits<double>::max();
    double Lold;
    double Lnew;


    arma::mat u(n,kmax);
    arma::mat w(n,kmax);
    std::vector<GM> mixtureCandidate = initMixtureModel(data, kmax);
    arma::vec a(kmax);
    a.fill(1./kmax);

    unsigned int k = kmax;

    for (unsigned int m = 0; m < kmax; ++m) {
      for (unsigned int i = 0; i < n; ++i) {
        u(i,m) = mixtureCandidate[m].getProb(data.col(i));  
      }
    }
    Lnew = 0;
    for (unsigned int m = 0; m < kmax; ++m) {
      if (a(m) > 0) {
        Lnew += 0.5 * N * std::log(n * a(m) / 12);
      }
    }
    Lnew += knz/2. * log(n/12.) + (knz * N + knz) / 2.;
    Lnew -= arma::as_scalar(arma::sum(arma::log(u * a)));
    while(knz >= kmin) {
      //std::cout << "knz: " << knz << std::endl;
      do {
        Lold = Lnew;
        Lnew = 0;
        for(unsigned int m = 0 ; m < k; ++m){
          
          arma::vec scalingVec = u * a;
          arma::uvec nonNull = find(scalingVec);
          w.col(m) = a(m) * u.col(m);
          arma::uvec colId;
          colId << m;
          w(nonNull, colId) = w(nonNull, colId) / scalingVec.elem(nonNull);
          //w(nonNull, colId);

          arma::rowvec temp = arma::sum(w) - (arma::ones<arma::rowvec>(k) * 0.5 * N);
          arma::uvec zeroIndices = arma::find(temp <= 0);
          temp.elem(zeroIndices).fill(0);
          //a.print("blub");
          a(m) = std::max(0., arma::sum(w.col(m)) - 0.5 * N) / (arma::sum(temp));
        //a.print("a");
          a /= arma::sum(a);
          if (a(m) > 0) {
            double scalingFactor = arma::sum(w.col(m));
            arma::vec newMu = data * w.col(m) / scalingFactor;
            arma::mat tempMat1 = data - newMu * arma::ones<arma::rowvec>(n);
            arma::mat tempMat2 = tempMat1.t() % (w.col(m) * arma::ones<arma::rowvec>(d) );
            arma::mat newSigma = 1./scalingFactor * tempMat1 * tempMat2;
            mixtureCandidate[m].setMu(newMu);
            mixtureCandidate[m].setSigma(newSigma);
            for (unsigned int i = 0; i < n; ++i) {
              u(i,m) = mixtureCandidate[m].getProb(data.col(i));
            //std::cout << "boo1" << std::endl;
            }
          } else{
            --knz;
            --k;
            a.shed_row(m);
            mixtureCandidate.erase(mixtureCandidate.begin() + m);
            u.shed_col(m);
            w.shed_col(m);
          }
        }
        for (unsigned int m = 0; m < k; ++m) {
          if (a(m) <= 0) {
            //std::cout << "BLUB" << std::endl;
          }
            Lnew += 0.5 * N * std::log(n * a(m) / 12);
        }
        Lnew += 0.5 * (knz * log(n/12.) + (knz * N + knz));
        Lnew -= arma::as_scalar(arma::sum(arma::log(u * a)));
      }
      while((Lold - Lnew) >= eps * std::abs(Lold));
      if (Lnew <= Lmin) {
        Lmin = Lnew;
        mixture_ = mixtureCandidate;
        weights_ = a;
      }
      arma::uvec tempIndices = arma::find(a == arma::min(a));
      unsigned int erasedIndex = tempIndices(0);
      //std::cout << erasedIndex << "ERASOR" << std::endl;
      a.shed_row(erasedIndex);
      mixtureCandidate.erase(mixtureCandidate.begin() + erasedIndex);
      u.shed_col(erasedIndex);
      w.shed_col(erasedIndex);
      
      --knz;
      --k;
    //std::cout << "knz: " << knz << std::endl;
    }

  }
  std::vector<GM> initMixtureModel(const arma::mat & data, unsigned int kmax) {
    std::vector<GM> mixtureCandidate(kmax);
    unsigned int n = data.n_cols; //number of data elements;
    unsigned int d = data.n_rows; //number of dimensions;
    for (unsigned int m = 0; m < kmax; ++m) {
      arma::vec newMu = data.col(std::rand() % n);
      mixtureCandidate[m].setMu(newMu);
      arma::mat tempMat = data - newMu * arma::ones<arma::rowvec>(n);
      double thetasq = 1./(10 * d) * 1./n * arma::trace(tempMat * tempMat.t());
      arma::mat newSigma = thetasq * arma::eye<arma::mat>(d,d);
      mixtureCandidate[m].setSigma(newSigma);
    }
    return mixtureCandidate;
  }



};



#endif
