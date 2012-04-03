#include <hmm/gmm.hpp>
#include <armadillo>
#include <assert.h>

#ifndef __INCLUDE_HMM_HMM_HPP__
#define __INCLUDE_HMM_HMM_HPP__
class HMM {
  unsigned int N_; //number of states, Q= { 0, ... , N-1}
  unsigned int T_;
  arma::mat A_; //transition probabilities
  std::vector<GMM> B_; //Gaussian mixture models
  arma::rowvec pi_; //initial state distribution
  arma::mat alpha_;
  arma::mat beta_;
  arma::mat gamma_;
  arma::cube xi_;
  arma::rowvec c_;
  arma::rowvec C_;
  double pprob_;
  //for debugging purposes

 friend class HMMComp;
 
  void checkAllComponents() {
   
    arma::vec rowSumA = arma::sum(A_, 1);
    rowSumA.print("rowSumA");
    double sumPi = arma::sum(pi_);
    std::cout << "sumPi: " << sumPi << std::endl;
    arma::vec weights = arma::zeros((unsigned int)B_.size());
    for (unsigned int i = 0; i < (unsigned int) B_.size(); ++i) {
      weights(i) = arma::accu(B_[i].getWeights());
    }
    weights.print("bCumWeights");

    arma::rowvec checksum = arma::sum(gamma_);
    checksum.print("checksum");
    arma::uvec checksumIndices = arma::find(checksum < 1.0 - 1E-2);
    if (checksumIndices.n_elem >= 1) {
      arma::rowvec checksumAlpha = arma::sum(alpha_);
      checksumAlpha.print("checkAlpha");
      //alpha_.print("alpha");
      arma::rowvec checksumBeta = arma::sum(beta_);
      checksumBeta.print("checkBeta");
      //beta_.print("beta");
      c_.print("c");
      throw std::runtime_error("data going wonky");
    }


    if (!arma::is_finite(A_)) {
      A_.print("A Fail");
      throw std::runtime_error("A has invalid entries");
    }
    if (!arma::is_finite(pi_)) {
      pi_.print("pi Fail");
      throw std::runtime_error("pi has invalid entries");
    }
    if (!arma::is_finite(alpha_)) {
      alpha_.print("alpha Fail");
      throw std::runtime_error("alpha has invalid entries");
    }
    if (!arma::is_finite(beta_)) {
      beta_.print("beta Fail");
      throw std::runtime_error("beta has invalid entries");
    }
    if (!arma::is_finite(gamma_)) {
      gamma_.print("gamma Fail");
      throw std::runtime_error("gamma has invalid entries");
    }
    if (!arma::is_finite(xi_)) {
      xi_.print("xi Fail");
      throw std::runtime_error("xi has invalid entries");
    }
  }
  public:
  void
    print(std::string header = "") {
      A_.print("A");
      for (size_t i = 0; i < B_.size(); ++i) {
        B_[i].print("B");
      }
      pi_.print("pi");

    }

  double baumWelch(const arma::mat & data, const std::vector<GMM> & B, unsigned int seed);
  double baumWelch(const arma::mat & data, unsigned int seed);
void createGMM(const arma::mat & data, const arma::urowvec & labels, unsigned int kmin, unsigned int kmax);
//Warning: This will invalidate the internal data other than the relevant model data A, B and pi.
void
  sort(unsigned int dimension) {
    arma::mat means; 
    //1. step: sort all B_; according to their median
    try{
      for (size_t i = 0; i < B_.size(); ++i) {
        means = arma::join_rows(means, B_[i].sort(dimension));
      }
    }
    catch(const std::logic_error & e) {
      throw e;
    }
    means.print("means");
    arma::rowvec relevantDim = means.row(dimension);
    arma::urowvec indices = arma::sort_index(relevantDim);
    print();
    indices.print("indices");
    A_ = A_.submat(indices, indices);
    pi_ = pi_.elem(indices).t();
    std::vector<size_t> vindices = arma::conv_to<std::vector<size_t> >::from(indices);
    reorder(B_, vindices);  
print();
  }
  private:
void init(const std::vector<GMM> & B, unsigned int seed);


double forwardProcedure(const arma::mat & data);
void backwardProcedure(const arma::mat & data);

void computeGamma();
void computeXi(const arma::mat & data);



};
void
HMM::createGMM(const arma::mat & data, const arma::urowvec & labels, unsigned int kmin, unsigned int kmax) {
  const unsigned int numLabels = (unsigned int) arma::as_scalar(arma::max(labels)) + 1;
  B_.clear();

  for (unsigned int i = 0; i < numLabels; ++i){
    arma::uvec indices = arma::find(labels == i);
    B_.push_back(GMM(data.cols(indices), kmin, kmax));
  }

}

void 
HMM::init(const std::vector<GMM> & B, unsigned int seed = 0) {
  std::srand(seed);
  N_ = (unsigned int) B.size();
  A_ = arma::randu(N_,N_);
  //A_ has a probability distribution in every row, so that has to be c_d
  A_ /= arma::sum(A_,1 ) * arma::ones(1, N_);


  B_ = B;
  pi_ = arma::randu(1,N_);
  pi_ /= arma::accu(pi_);

}

double
HMM::forwardProcedure(const arma::mat & data) {

  assert(N_ == B_.size());
  if (N_ != B_.size()) throw std::logic_error("The number of mixture models doesn't match the number of states");
  T_ = data.n_cols;

  alpha_ = arma::zeros(N_, T_);
  c_ = arma::zeros(1, T_);
  //initialisation
  for(unsigned int i = 0; i < N_; ++i) {
    alpha_(i, 0) = pi_[i] * B_[i].getProb(data.col(0));
  }

  c_(0) = arma::accu(alpha_.col(0));
  alpha_.col(0) /= arma::as_scalar(c_(0));

  //alpha_.print("alpha");
  //c_.print("scale");
  //iteration
  for(unsigned int t = 1; t < T_; ++t) {
    for (unsigned int j = 0; j < N_; ++j) {
      alpha_(j, t) = arma::as_scalar(A_.col(j).t() * alpha_.col(t-1)) * B_[j].getProb(data.col(t)); 
    }
    c_(t) = arma::accu(alpha_.col(t));
    alpha_.col(t) /= arma::as_scalar(c_(t)); 
  }

  pprob_ = arma::accu(arma::log(c_));
  return pprob_;
}

void
HMM::backwardProcedure(const arma::mat & data) {


  beta_ = arma::mat(N_, T_);
  beta_.col(T_-1).fill(1./c_(T_-1));


  //temporary probabilities
  arma::vec b(N_);

  //iteration
  for(unsigned int t = T_-1; t > 0; --t) {
    for (unsigned int j = 0; j < N_; ++j) {
      b(j) = B_[j].getProb(data.col(t));
    }
    for (unsigned int i = 0; i < N_; ++i) {
      beta_(i,t-1) = arma::as_scalar(A_.row(i) * (b % beta_.col(t)));
    }
    beta_.col(t-1) /= arma::as_scalar(c_(t-1)); 

  }
}




void
HMM::computeXi(const arma::mat & data) {
  xi_ = arma::cube(T_-1, N_, N_);


//note the index switch, it's for performance reasons
  for(unsigned int i = 0; i < N_; ++i) {
    for(unsigned int j = 0; j < N_; ++j) {
      for(unsigned int t = 0; t < T_ - 1; ++t) {
        xi_(t,j,i) = (alpha_(i,t) * A_(i,j)) * (B_[j].getProb(data.col(t+1)) *  beta_(j,t+1));
      }
    }
  }
}

void
HMM::computeGamma() {
  gamma_ = alpha_ % beta_;
  gamma_ %= (arma::ones(N_, 1) * c_);

}

double
HMM::baumWelch(const arma::mat & data, unsigned int seed = 0) {
  return baumWelch(data, B_, seed);
}
double
HMM::baumWelch(const arma::mat & data, const std::vector<GMM> & B, unsigned int seed = 0) {

  init(B, seed);

  double logprobc = forwardProcedure(data);
  backwardProcedure(data);
  computeXi(data);
  computeGamma();
  double logprobprev = 0.0;
  double delta;
  double eps = 1E-6;
  do {
    logprobprev = logprobc;
    //update state probabilities pi
    pi_ = gamma_.col(0).t();
    for (unsigned int i = 0; i < N_; ++i) {
      arma::rowvec xi_i = arma::sum(xi_.slice(i));
      double shortscale = 1./arma::accu(gamma_.submat(arma::span(i),arma::span(0,T_-2)));
      for (unsigned int j = 0; j < N_; ++j) {
        //update transition matrix A
        //A_(i,j) = arma::accu(xi_i.slice(j)) * shortscale;
        A_(i,j) = xi_i(j) * shortscale;
      }
      // and update distributions
      unsigned int numComponents = (unsigned int) B_[i].getNumComponents();

      arma::mat gamma_lt = arma::mat(T_, numComponents);

      for(unsigned int t = 0; t < T_ ; ++t) {
        double sumProb = B_[i].getProb(data.col(t));
        for (unsigned int l = 0; l < numComponents; ++l) {
          gamma_lt(t, l) = gamma_(i,t) * B_[i].getProb(data.col(t), l);
        }
        if (sumProb != 0.) {
          gamma_lt.row(t) /= sumProb; 
        }
      }
      //gamma_lt.print("gamma_lt");

      double scale = 1./arma::accu(gamma_.row(i));
      for (unsigned int l = 0; l < numComponents; ++l) {
        double sumGammaLt = arma::accu(gamma_lt.col(l));
        double newWeight = scale * sumGammaLt;
        arma::vec newMu = data * gamma_lt.col(l) / sumGammaLt;
        arma::mat tempMat = data- newMu * arma::ones(1,T_);
        unsigned int d = data.n_rows;
        arma::mat newSigma = arma::zeros(d, d);
        for (unsigned int t = 0; t < T_ ; ++t) {
          arma::vec diff = data.col(t) - newMu;
          newSigma += diff * diff.t() * gamma_lt(t, l)/ sumGammaLt;
        }

        try{
          B_[i].updateGM(l, newWeight, newMu, newSigma);
        }
        catch( const std::runtime_error & e) {
          tempMat.print("tempMat");
          gamma_lt.col(l).print("gamma_lt");
          throw e;
        }
      }
    }


    //do next iteration step

    //print();
    logprobc = forwardProcedure(data);
    backwardProcedure(data);
    computeXi(data);
    computeGamma();

    std::cout << logprobc << " " << logprobprev << std::endl;
    delta = logprobc - logprobprev;

  } 
  while (delta >= eps * std::abs(logprobprev));

  return logprobc;

}

double
kld(const HMM & a, const HMM & b) {

  return 0;
}







#endif
