#ifndef __INCLUDE_HMM_GMM_HPP__
#define __INCLUDE_HMM_GMM_HPP__
typedef struct GM_c {
  double mean[3];
  double sigma[5];
} GM_c;

#ifdef __cplusplus
#include <algorithm>
#include <armadillo>
#include <vector>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES
#endif


/** 
 *  \class GM
 *  \brief A class to represent a gaussian model, using armadillo matrices
 *
 * */
class GM {
  unsigned int D_;

  arma::vec mu_;
  arma::mat sigma_;
  arma::mat invSigma_;

  /**The coeff is cached and recalculated on a sigma change */ 
  double coeff_; 
 

  public:
  unsigned int const & getD() const {
    return D_;
  }
  arma::vec const & getMu() const {
    return mu_;
  }
  arma::mat const & getSigma() const {
    return sigma_;
  }

  /** Probability for a single vector*/
  double getProb(const arma::vec & loc) const {
    const arma::vec diff = loc - mu_;
    double exponent = -0.5 * arma::as_scalar(diff.t() * invSigma_ * diff); //TODO: check how transpose works
    return coeff_ * std::exp(exponent);
  }

  /** For optimization reasons (to use vector multiplication), return the probability of a set of column-vectors*/
  arma::rowvec getDataProb(const arma::mat & data) const {
    const arma::mat diff = data - mu_ * arma::ones(1, data.n_cols);
    arma::rowvec exponent = -0.5 * arma::sum((invSigma_.t() * diff) % diff);
    return coeff_ * arma::exp(exponent);

  }


  
  void
    setSigma(const arma::mat & sigma) {
      D_ = sigma.n_rows;
      arma::uvec nonZeroCount = arma::find(sigma != 0);
      sigma_ = sigma;
      if (arma::det(sigma_) <= 0) {
        throw std::runtime_error("det(sigma) has to be positive");
      }
      invSigma_ = arma::inv(sigma_);
      coeff_ = 1./std::sqrt(std::pow(2. * M_PI, D_) * arma::det(sigma_));
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
  GM(const arma::vec & mu, const arma::mat & sigma) {
    setMu(mu);
    setSigma(sigma);
  }
  GM(const GM_c & gmc) {
     arma::mat sigma = arma::mat(3,3);
     arma::vec mu = arma::vec((double *)&gmc.mean, 3);
     sigma(0,0) = gmc.sigma[0];
     sigma(0,1) = gmc.sigma[1];
     sigma(0,2) = gmc.sigma[2];
     sigma(1,1) = gmc.sigma[3];
     sigma(1,2) = gmc.sigma[4];
     setMu(mu);
     setSigma(arma::symmatu(sigma));
  
  }
  /** For debugging reasons*/
  bool sanityCheck() const {
    bool sane = false;
    if (D_ == 0 || arma::det(sigma_) == 0 || !invSigma_.is_finite()) sane = false;
    if (!sane) {
      std::cout << "coeff: " << coeff_ << std::endl;
      std::cout << "D: " << D_ << std::endl;
      sigma_.print("sigma");
      invSigma_.print("invSigma");
      throw std::logic_error("Something is wrong with the GM");
    }
    return sane;
  }
  void print(std::string header = "") const {
    std::cout << header << std::endl;
    mu_.print("Mean vector");
    sigma_.print("Sigma");
  }
};

/** 
 * \brief Models a Gaussian mixture model
 *
 * */

class GMM {
  //a vector of gaussian models
  std::vector<GM> mixture_;
  arma::vec weights_;
  unsigned int D_;
  public:

  GMM() {} 
  GMM(const arma::mat& data, unsigned int kmin, unsigned int kmax) {
    findMaxLikelihood(data, kmin, kmax);
  }
  unsigned int
    getD() const {
      if (mixture_.size()) {
        return mixture_[0].getD();
      }
      else {
        return 0;
      }
    }

    unsigned int n_gm() const {
      return weights_.n_elem;
    }
  arma::vec const & getWeights() const {
    return weights_;
  }
  GM const & getGM(unsigned int index) const {
    return mixture_[index];
  }

  /** Removes any Gaussian models whose correspondent weight is zero or less*/
  arma::uvec
  cleanupGMs() {
    arma::uvec goodIndices = arma::find(weights_ > 0);
    while(1) {
      double * p = std::find(weights_.begin(), weights_.end(), 0.);
      if (p == weights_.end()) break;
      unsigned int index = (unsigned int) std::distance(weights_.begin(), p);
      mixture_.erase(mixture_.begin() + index);
      weights_.shed_row(index);
    }
    return goodIndices;
  }
  
  void insertGM(const GM & gm, double weight) {
    mixture_.push_back(gm);
    weights_.insert_rows(weights_.n_rows, weight);
  }
  
  bool  
    updateGM(unsigned int index, double weight, const arma::vec & mu, const arma::mat & sigma) {
      if (arma::det(sigma) <= 0) {
        //mixture_.erase(mixture_.begin() + index);
        weights_(index) = 0.;
        normalizeWeights();
        return false;
      }
      weights_[index] = weight;
      mixture_[index] = GM(mu, sigma);
      return true;
    }
  void setGM(unsigned int index, const GM & gm) {
    mixture_[index] = gm;
  }
  void
    normalizeWeights() {
      weights_ /= arma::accu(weights_);
    }
  size_t
    getNumComponents() {
      return mixture_.size();
    }
  void print(std::string header = "") const {
    std::cout << header << std::endl;
    arma::uvec nonZero = arma::find(weights_);
    std::for_each(nonZero.begin(), nonZero.end(), [this](unsigned int index) {
        std::cout << "\n weight: " << weights_(index) << std::endl;
        mixture_[index].print("\t Mixture");
        });
  }

  arma::vec getMu() const {
    arma::mat mu;

    for (unsigned int i = 0; i < weights_.n_elem; ++i) {
      mu = arma::join_rows(mu, mixture_[i].getMu());
    }
    return arma::conv_to<arma::vec>::from(mu * weights_);
  }

/** Return the probability of a vector, given the current mixture model. Only takes the models in [start, end) into
 * account.*/
  double getProb(const arma::vec & datum, unsigned int start, unsigned int end) const {
    double prob = 0.;
    for(unsigned int i = start; i < end; ++i) {
      prob += weights_(i) * mixture_[i].getProb(datum);
    }
    return prob;
  }
  double getProb(const arma::vec & datum, unsigned int index) const {
    return getProb(datum, index, index + 1);
  }
  double getProb(const arma::vec & datum) const {
    return getProb(datum, 0, (unsigned int)mixture_.size());
  }


  //implementation of the EM-algorithm
  /** 
   * \brief Find the best Gaussian mixture model for given data, via the EM-algorithm
   * \param[in] data The data that the model should be fitted to
   * \param[in] kmin the minimum number of Gaussian models that should be used
   * \param[in] kmax the maximum number (and therefore starting number) of Gaussian models that should be used
   * \see "Unsupervised learning of finite mixture models", M. Figueiredo and A. K. Jain,  http://www.lx.it.pt/~mtf/IEEE_TPAMI_2002.pdf  */
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
      while(1) {
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
            arma::uvec failure= arma::find(w != w);
            if (failure.n_elem > 0) {
              std::cout << "m: " << m <<std::endl;
              w.print("3");
              scalingVec.print("scaling");
              scalingVec.elem(nonNull).print("scalingmod");
              u.print("u");
              a.print("a");
              for (size_t i = 0; i < mixtureCandidate.size(); ++i) 
                mixtureCandidate[i].print("mixture");
            }

            //w.print("w");
            //w(nonNull, colId);

            arma::rowvec temp = arma::sum(w) - (0.5 * N);
            arma::uvec zeroIndices = arma::find(temp <= 0);
            temp.elem(zeroIndices).fill(0);
            a(m) = std::max(0., arma::sum(w.col(m)) - 0.5 * N) / (arma::sum(temp));
            a /= arma::sum(a);
            if (a(m) > 0) {
              double scalingFactor = arma::sum(w.col(m));
              arma::vec newMu = data * w.col(m) / scalingFactor;
              arma::mat tempMat1 = data - newMu * arma::ones<arma::rowvec>(n);
              arma::mat tempMat2 = tempMat1.t() % (w.col(m) * arma::ones<arma::rowvec>(d) );
              arma::mat newSigma = 1./scalingFactor * tempMat1 * tempMat2;
              arma::uvec notZeroIndices = arma::find(newSigma != 0);
              if (notZeroIndices.n_elem == 0) {
                --knz;
                --k;
                a.shed_row(m);
                mixtureCandidate.erase(mixtureCandidate.begin() + m);
                u.shed_col(m);
                w.shed_col(m);
                if (knz < kmin) break; 
                continue;
              }
              mixtureCandidate[m].setMu(newMu);
              mixtureCandidate[m].setSigma(newSigma);
              for (unsigned int i = 0; i < n; ++i) {
                u(i,m) = mixtureCandidate[m].getProb(data.col(i));
              }

              arma::uvec failure= arma::find(u != u);
              if (failure.n_elem > 0) {
                std::cout << m << " m" << std::endl;
                mixtureCandidate[m].print("Candidate");
                newSigma.print("sigma");
                tempMat1.print("tempMat");
                std::cout << "scalingFactor" <<scalingFactor << std::endl;
                w.col(m).print("wcol");
                tempMat2.print("tempMat2");
              }
            } else {
              --knz;
              --k;
              a.shed_row(m);
              mixtureCandidate.erase(mixtureCandidate.begin() + m);
              u.shed_col(m);
              w.shed_col(m);
              if (knz < kmin) break; 
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
        if (knz <= kmin) break;
        arma::uvec tempIndices = arma::find(a == arma::min(a));
        unsigned int erasedIndex = tempIndices(0);
        //std::cout << erasedIndex << "ERASOR" << std::endl;
        a.shed_row(erasedIndex);
        mixtureCandidate.erase(mixtureCandidate.begin() + erasedIndex);
        u.shed_col(erasedIndex);
        w.shed_col(erasedIndex);

        --knz;
        --k;
      }


    }

    /** 
     * \brief Initialize the Mixture model by picking a random subset of the data which will become the means
     * for every Gaussian model
     * */
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

/** 
 *  \class IdentityFunctor
 *  \brief A simple functor to pass a gmm untouched.
 *
 * */
struct IdentityFunctor {
  GMM operator()(const GMM & gmm) const {
    return gmm;
  }
};

/** 
 *  \class MatrixTransformationFunctor
 *  \brief A functor to apply a transformation Matrix to a Gaussian Mixture Model
 *
 * */
struct MatrixTransformationFunctor {
  /** Transformation matrix to be applied*/
  const arma::mat transMatrix_;
  MatrixTransformationFunctor(const arma::mat & transMatrix):transMatrix_(transMatrix) {}
  GM
    transformGM(const GM & gm) const {

      GM gm2 = gm;
      gm2.setSigma(transMatrix_ * gm2.getSigma() * transMatrix_.t());
      gm2.setMu(transMatrix_ * gm2.getMu());

      return gm2;
    }

  GMM
    transformGMM(const GMM & gmm) const {
      GMM gmm2 = gmm;
      for (unsigned int i = 0; i < gmm.n_gm(); ++i) {
        gmm2.setGM(i, transformGM(gmm.getGM(i)));
      }
      return gmm2;
    }
  GMM operator()(const GMM & gmm) const {
    return transformGMM(gmm);
  }
};
/** 
 * \class GMMCreator
 * \brief A class to construct a set of Gaussian Mixture Models where every Label is assigned a GMM
 *
 * */
struct GMMCreator {
  /** Minimum Number of GMs per Model*/
  unsigned int kmin_;
  /** Maximum Number of GMs per Model*/
  unsigned int kmax_;
  
  GMMCreator(unsigned int kmin, unsigned int kmax): kmin_(kmin), kmax_(kmax) {}

  std::vector<GMM>
    operator() (const arma::mat & data, const arma::urowvec & labels) {
      const unsigned int numLabels = (unsigned int) arma::as_scalar(arma::max(labels)) + 1;
      std::vector<GMM> BModels;

      for (unsigned int i = 0; i < numLabels; ++i){
        arma::uvec indices = arma::find(labels == i);
        if (indices.n_elem > 0) {
          BModels.push_back(GMM(data.cols(indices), kmin_, kmax_));
        }
      }
      return BModels;
    }
};
#endif //ifdef __cplusplus
#endif
