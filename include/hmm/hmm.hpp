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
  arma::vec scale_;
  double pprob_;
  //for debugging purposes

  bool initSuccess_;
  bool forwardSuccess_;
  bool backwardSuccess_;
  bool gammaSuccess_;
  bool xiSuccess_;
public:
  HMM() {
    initSuccess_ = false;
    forwardSuccess_ = false;
    backwardSuccess_ = false;
    gammaSuccess_ = false;
    xiSuccess_ = false;

  }


  double baumWelch(const arma::mat & data, const std::vector<GMM> & B, unsigned int seed);
private:
  void init(const std::vector<GMM> & B, unsigned int seed);


  double forwardProcedure(const arma::mat & data);
  void backwardProcedure(const arma::mat & data);

  void computeGamma();
  void computeXi(const arma::mat & data);



};

void 
HMM::init(const std::vector<GMM> & B, unsigned int seed = 0) {
  std::srand(seed);
  N_ = (unsigned int) B.size();
  A_ = arma::randu(N_,N_);
  //A_ has a probability distribution in every row, so that has to be scale_d
  A_ /= (arma::ones(N_, 1) * arma::sum(A_,1 ));

  B_ = B;
  pi_ = arma::randu(1,N_);
  pi_ /= arma::ones(1,N_) * arma::sum(pi_);


  initSuccess_ = true;
}

double
HMM::forwardProcedure(const arma::mat & data) {

  assert(N_ == B_.size());
  assert(initSuccess_ == true);
  T_ = data.n_cols;
  alpha_ = arma::mat(N_, T_);
  //initialisation
  for(unsigned int i = 0; i < N_; ++i) {
    alpha_(i, 0) = pi_[i] * B_[i].getProb(data.col(0));
  }
  //scaling
  //
  scale_(0) = arma::sum(alpha_.col(0));
  alpha_.col(0) /= arma::ones(N_) * scale_(0);

  //iteration
  for(unsigned int t = 1; t < T_; ++t) {
    for (unsigned int i = 0; i < N_; ++i) {
      alpha_(i, t) = arma::as_scalar(A_.row(i) * alpha_.col(t-1)) * B_[i].getProb(data.col(t)); 
    }
    scale_(t) = arma::sum(alpha_.col(t));
    alpha_.col(t) /= arma::ones(N_) * scale_(t);
  }

  pprob_ = arma::as_scalar(arma::sum(arma::log(scale_)));

  forwardSuccess_ = true;
  return pprob_;
}

void
HMM::backwardProcedure(const arma::mat & data) {

  assert(forwardSuccess_ == true);

  beta_ = arma::mat(N_, T_);
  beta_.col(T_-1).fill(1. / scale_(T_-1));

  //temporary probabilities
  arma::vec b(N_);

  //iteration
  for(unsigned int t = T_-1; t > 0; --t) {
    for (unsigned int i = 0; i < N_; ++i) {
      b(i) = B_[i].getProb(data.col(t));
    }
    for (unsigned int i = 0; i < N_; ++i) {
      beta_(i,t-1) = arma::as_scalar(A_.row(i) * (b % beta_.col(t)));
    }
    beta_.col(t-1) /= arma::ones(N_) * scale_(t-1);
  }

  for (unsigned int i = 0; i < N_; ++i) {
    b(i) = B_[i].getProb(data.col(0));
  }

  backwardSuccess_ = true;
}


void
HMM::computeGamma() {
  gamma_ = arma::mat(N_, T_);

  for(unsigned int t = 0; t < T_; ++t) {
    gamma_.col(t) = alpha_.col(t) % beta_.col(t);
    gamma_.col(t) /= arma::ones(N_) * arma::sum(gamma_.col(t));
  }
  gammaSuccess_ = true;
}

void
HMM::computeXi(const arma::mat & data) {
  xi_ = arma::cube(T_, N_, N_);


  for(unsigned int t = 0; t < T_ - 1; ++t) {
    for(unsigned int j = 0; j < N_; ++j) {
      for(unsigned int i = 0; i < N_; ++i) {
        xi_(t,i,j) = gamma_(i,t) * A_(i,j) * B_[j].getProb(data.col(t+1)) * beta_(j,t+1) / beta_(i,t);
      }
    }


  }
  xiSuccess_ = true;
}


double
HMM::baumWelch(const arma::mat & data, const std::vector<GMM> & B, unsigned int seed = 0) {

  init(B, seed);

  double logprobprev = forwardProcedure(data);
  backwardProcedure(data);
  computeGamma();
  computeXi(data);
  double logprobc = 0.0;
  double delta;
  double eps = 1E-5;
  do {
    //update state probabilities pi
    pi_ = gamma_.col(0).t();

    for (unsigned int i = 0; i < N_; ++i) {
      double scale = 1./arma::as_scalar(arma::sum(gamma_.row(i)));
      for (unsigned int j = 0; j < N_; ++j) {
        //update transition matrix A
        arma::vec xi_ij= xi_.subcube(arma::span::all, arma::span(i), arma::span(j));
        A_(i,j) = arma::as_scalar(arma::sum(xi_ij)) * scale;
      }
      // and update distributions
      unsigned int numComponents = (unsigned int) B_[i].getNumComponents();

      arma::mat gamma_lt = arma::mat(T_, numComponents);

      for(unsigned int t = 0; t < T_ ; ++t) {
        double invSumProb = 1./B_[i].getProb(data.col(t));
        for (unsigned int l = 0; l < numComponents; ++l) {
          gamma_lt(l, t) = gamma_(i,t) * B_[i].getProb(data.col(t), l) * invSumProb;
        }
      }


      for (unsigned int l = 0; l < numComponents; ++l) {
        double sumGammaLt = arma::as_scalar(arma::sum(gamma_lt.col(l)));
        double newWeight = scale * sumGammaLt;
        arma::vec newMu = data * gamma_lt.col(l) / sumGammaLt;
        arma::mat tempMat = data - arma::ones(1,T_) * newMu;
        unsigned int d = data.n_rows;
        arma::mat newSigma = arma::ones(d,1) * gamma_lt.col(l).t() % tempMat * tempMat.t() / sumGammaLt;
        B_[i].updateGM(l, newWeight, newMu, newSigma);
      }
    }


    //do next iteration step


    logprobc = forwardProcedure(data);
    backwardProcedure(data);
    computeGamma();
    computeXi(data);

    delta = logprobc - logprobprev;
    logprobprev = logprobc;

  } 
  while (delta > eps);


  return logprobc;

}







#endif
