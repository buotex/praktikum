#include <armadillo>
#include <hmm/hmm.hpp>
#include <cmath>
#ifndef __INCLUDE_HMM_KLD_HPP__
#define __INCLUDE_HMM_KLD_HPP__


struct HMMComp{

  /*
  double
    pmfDistance(const arma::rowvec & v1, const arma::rowvec & v2) {
      return pmfDistance (v1.t(), v2.t());
    }
    */
  static
    double
    pmfDistance(const arma::mat & v1, const arma::mat & v2) {
      if (v1.n_elem != v2.n_elem) throw std::logic_error("different amount of entries in compared pmfs");
      double eps = 1E-6;
      arma::vec v1_ = arma::conv_to<arma::vec>::from(v1);
      arma::vec v2_ = arma::conv_to<arma::vec>::from(v2);
      arma::uvec v1Zero = arma::find(v1_ == 0.);
      arma::uvec v2Zero = arma::find(v2_ == 0.);
      v1_.elem(v1Zero) += eps;
      v2_.elem(v2Zero) += eps;

      arma::vec quotient = arma::log(v1_ / v2_);

      return arma::as_scalar(v1_.t() * quotient);

    }

  static
    double gmDistance(const GM & gm1, const GM & gm2){
      const arma::mat & C = gm1.getSigma();
      const arma::mat & C_ = gm2.getSigma();

      const arma::vec & mu = gm1.getMu();
      const arma::vec & mu_ = gm2.getMu();
      const unsigned int d = mu.n_elem;
      arma::vec diff = mu - mu_;
      double distance = 
        0.5 * (
            std::log(arma::det(C_)/ arma::det(C)) - 
            d + 
            arma::trace(arma::inv(C_) * C) +
            arma::as_scalar(diff.t() * arma::inv(C_) * diff)
            );

      return distance;

    }
  static
    double gmmDistance(const GMM & gmm1, const GMM & gmm2) {
      const arma::vec & w = gmm1.getWeights();
      const arma::vec & w_ = gmm2.getWeights();
      const unsigned int n1 = w.n_elem;
      const unsigned int n2 = w_.n_elem;


      double distance = 0.0;
      for (unsigned int j = 0; j < n1; ++j) {
        arma::vec D_ = arma::zeros(n2);
        const GM & gmj = gmm1.getGM(j);
        for (unsigned int k = 0; k < n2; ++k) {
          D_(k) = gmDistance(gmj, gmm2.getGM(k));
        }
        distance += w(j) * arma::min(D_);
      }

      for (unsigned int k = 0; k < n2; ++k) {
        arma::vec D_ = arma::zeros(n1);
        const GM & gmk = gmm2.getGM(k);
        for (unsigned int j = 0; j < n1; ++j) {
          D_(j) = gmDistance(gmm1.getGM(j), gmk);
        }
        distance += w_(k) * arma::min(D_);
      }
      return 0.5 * distance;

    }


  double 
    static kld(const HMM & hmm1, const HMM & hmm2) {

      double piDistance = pmfDistance(hmm1.pi_, hmm2.pi_);
      if (hmm1.N_ != hmm2.N_) throw std::logic_error("different number of states in the hmms");

      unsigned int N = hmm1.N_;
      arma::vec d = arma::zeros(N);
      arma::vec e = arma::zeros(N);


      const arma::mat & A = hmm1.A_;
      const arma::mat & A_ = hmm2.A_;
      const std::vector<GMM> & B = hmm1.B_;
      const std::vector<GMM> & B_ = hmm2.B_;
      if (A.n_rows != A.n_cols) throw std::logic_error("Matrix isn't quadratic");
      if (A_.n_rows != A_.n_cols) throw std::logic_error("Matrix isn't quadratic");

      arma::cube APow = arma::zeros(A.n_rows, A.n_cols, N);
      arma::mat currPow = arma::eye(A.n_rows, A.n_cols);
      for (unsigned int i = 0; i < N; ++i) {

        APow.slice(i) = currPow;
        currPow *= A;

        e(i) = gmmDistance(B[i], B_[i]);
        d(i) = pmfDistance(A.row(i), A_.row(i)) + e(i);
      }
      arma::vec f = arma::zeros(N);

      for (unsigned int i = 0; i < N-1; ++i) {
        f += APow.slice(i) * d + APow.slice(N-1) * e;  
      }

      return piDistance + arma::as_scalar(hmm1.pi_ * f);
    }

  static double
    symmetric_kld(const HMM & hmm1, const HMM & hmm2) {
      return kld(hmm1, hmm2) + kld(hmm2, hmm1);
    }
};



#endif
