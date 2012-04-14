#include <armadillo>
#include <hmm/hmm.hpp>
#include <cmath>
#include <algorithm>
#ifndef __INCLUDE_HMM_KLD_HPP__
#define __INCLUDE_HMM_KLD_HPP__


struct HMMComp{
  static
  arma::mat
  findTransformationMatrix(const HMM & hmm1, const HMM & hmm2) {
      
      arma::rowvec pi1 = findStationary(hmm1.A_);
      arma::rowvec pi2 = findStationary(hmm2.A_);
      arma::uvec ind1 = arma::sort_index(pi1.t(),1);
      arma::uvec ind2 = arma::sort_index(pi2.t(),1);
      unsigned int i1 = ind1(0);
      unsigned int i2 = ind2(0);
      const GMM & gmm1 = hmm1.BModels_[i1];
      const GMM & gmm2 = hmm2.BModels_[i2];
      arma::vec w1 = gmm1.getWeights();
      arma::vec w2 = gmm2.getWeights();
      arma::uvec wind1 = arma::sort_index(w1,1);
      arma::uvec wind2 = arma::sort_index(w2,1);
      arma::mat sigma1 = (gmm1.getGM(wind1(0))).getSigma();
      arma::mat sigma2 = (gmm2.getGM(wind2(0))).getSigma();
      arma::vec eigval1, eigval2;
      arma::mat eigvec1, eigvec2;
      arma::eig_sym(eigval1, eigvec1, sigma1);
      arma::eig_sym(eigval2, eigvec2, sigma2);


      arma::mat transformation = eigvec1 * arma::sqrt(arma::diagmat(eigval1)) * arma::inv(arma::sqrt(arma::diagmat(eigval2))) * eigvec2.t();
      return transformation;

  }
 static
  arma::cube
  findTransformationMatrix2(const HMM & hmm1, const HMM & hmm2) {
      unsigned int j1 = 0, j2 = 0;
      arma::mat coord1, coord2;

      arma::rowvec pi1 = findStationary(hmm1.A_);
      arma::rowvec pi2 = findStationary(hmm2.A_);
      arma::uvec ind1 = arma::sort_index(pi1.t(),1);
      arma::uvec ind2 = arma::sort_index(pi2.t(),1);
      unsigned int D = hmm1.BModels_[0].getD();
      while(coord1.n_cols < D) {
        unsigned int i1 = -1;
        try{
          i1 = ind1(j1);
        }
        catch(...) {
          std::cout << "Not enough data to find a transformation" << std::endl;
          arma::cube c(D, D, 1);
          c.slice(0) = arma::eye(D, D);
          return c;
        }
        const GMM & gmm1 = hmm1.BModels_[i1];
        arma::vec w1 = gmm1.getWeights();
        arma::vec mu = arma::zeros(D, 1);
        for (unsigned int j = 0; j < w1.n_elem; ++j) {
          mu += w1(j) * gmm1.getGM(j).getMu();
        }
        coord1 = arma::join_rows(coord1, mu);
        ++j1;
      }
      while(coord2.n_cols < D) {
        unsigned int i2 = -1;
        try{
          i2 = ind2(j2);
        }
        catch(...) {
          std::cout << "Not enough data to find a transformation" << std::endl;
          arma::cube c(D, D, 1);
          c.slice(0) = arma::eye(D, D);
          return c;
        }
        const GMM & gmm2 = hmm2.BModels_[i2];
        arma::vec w2 = gmm2.getWeights();
        arma::vec mu = arma::zeros(D, 1);
        for (unsigned int j = 0; j < w2.n_elem; ++j) {
          mu += w2(j) * gmm2.getGM(j).getMu();
        }
        coord2 = arma::join_rows(coord2, mu);
        ++j2;
      }
      arma::uvec indices(D);
      for (unsigned int i = 0; i < D; ++i) {
        indices(i) = i;
      }

      coord1.print("coord1");
      coord2.print("coord2");
      arma::cube transformation;
      do {

        arma::mat B; 
        //if it isn't possible to invert coord2, fiddle around with it.
        arma::mat coordTrans = coord2.cols(indices);
        while (!arma::inv(B, coordTrans)) {
          coordTrans += 1.E-4;
        }
        arma::cube c(D,D,1);
        c.slice(0) = coord1 * B;
        transformation = arma::join_slices(transformation, c);
      } while ( std::next_permutation (indices.begin(),indices.end()) );

      return transformation;
  }




 static
   arma::rowvec findStationary(const arma::mat & A, unsigned int steps = 1000, double eps = 1E-2) {
     arma::rowvec x = arma::ones(1,A.n_rows);
     arma::rowvec oldx = arma::ones(1,A.n_rows);
     for (unsigned int i = 0; i < steps; ++i) {
       x = oldx * A;
       if (arma::norm(x - oldx, "inf") < eps) return x;
       oldx = x;
     }
     arma::rowvec newx = x * A;
     x.print("oldX");
     newx.print("newX");
     return arma::zeros(1, A.n_rows);
   }
 static
   arma::mat
   samplePointOffsets(const arma::mat & M) {
     arma::vec eigval;
     arma::mat eigvec;

     arma::eig_sym(eigval, eigvec, M);

     //eigvec.print("eigvec");
     //eigval.print("eigval");

     arma::mat rootK = eigvec * arma::diagmat(arma::sqrt(eigval));
     //rootK.print("rootK");
     return rootK;
   }


 //src: A Distance measure between GMMs based on the unscented transform//
 //To prevent infinities from happening, we'll already calculate e^-D here
 static 
   double
   expUnscentedTransform(const GMM & gmm1, const GMM & gmm2, double K = 0.5) {
     if (gmm1.getD() != gmm2.getD()) throw std::runtime_error("wrong dimensions in the GMMs");
     unsigned int d = gmm1.getD();
     arma::vec weights = gmm1.getWeights();
     double val = 1.;
     double prefix = 1./(2 * d);
     for (unsigned int i = 0; i < weights.n_elem; ++i) {
       const GM & gm = gmm1.getGM(i);
       arma::mat sigma = gm.getSigma();
       arma::mat mu = gm.getMu();
       arma::mat offsets = samplePointOffsets(d * sigma);
       arma::mat x_1 = mu * arma::ones(1, d) + offsets;
       arma::mat x_2 = mu * arma::ones(1, d) - offsets;
       arma::mat x_i = arma::join_rows(x_1, x_2);
       //x_i.print("x_i");
       for (unsigned int k = 0; k < 2 * d; ++k) {
         //val += weights(i) * std::log(gmm2.getProb(x_i.col(k)));
         val *= std::pow(gmm1.getProb(x_i.col(k)),-K * weights(i) * prefix); 
         val *= std::pow(gmm2.getProb(x_i.col(k)),K * weights(i) * prefix); 
         //std::cout << "prob: " << gmm1.getProb(x_i.col(k)) << "weight: " << weights(i) << std::endl;
         //std::cout << "prob: " << gmm2.getProb(x_i.col(k)) << "weight: " << weights(i) << std::endl;
         //std::cout << "entry " << val << std::endl;
       }
     }
     //std::cout << "FUBLA "  << val << std::endl << std::endl;

     return val;
   }

 static double
   hoyerMeasure(const arma::vec & u) {
     double N = (double)u.n_elem;
     return (std::sqrt(N) - arma::norm(u,1) / arma::norm(u,2)) / (std::sqrt(N) - 1);
   }
 static 
   double
   giniIndex(const arma::vec & u) {
     unsigned int N = u.n_elem;
     arma::uvec indices = arma::sort_index(u);
     double sum = 0.;
     double invNorm = 1./arma::norm(u,1);

     for (unsigned int k = 0; k < N; ++k) {
       sum += u(indices(k)) * invNorm * ((double)N - (double) k - 0.5) / double(N);
       //std::cout << "Index " << k << " " << u(indices(k)) << std::endl;
     }
     //std::cout << "sum: " << sum << std::endl;
     return 1. - 2. * sum; 
   }
 static
   double
   normalizedGiniIndex(const arma::vec & u) {
     double N = (double) u.n_elem;
     if (N == 1. || arma::norm(u,1) == 0) return 0.;
     return N/(N-1.) * giniIndex(u);
   }
 template <typename TransformationFunctor = IdentityFunctor>
   static
   double
   sMrandomWalk(const HMM & hmm1, const HMM & hmm2, TransformationFunctor func = TransformationFunctor(), int distanceSelect = 0){
     //calculate stationary probabilities for the states
     arma::rowvec pi1 = findStationary(hmm1.A_);
     arma::rowvec pi2 = findStationary(hmm2.A_);
     pi1.print("pi1");
     pi2.print("pi2");


     //calculate Qij matrix, which represent the similarity between the states in both matrices
     //using the unscentedTransform here, could also use KLD
     arma::mat Sij = arma::mat(hmm1.N_, hmm2.N_);
     double ES = 0.;
     arma::mat Qij = arma::mat(hmm1.N_, hmm2.N_);
     //pi1.print("pi1");
     //pi2.print("pi2");
     for (unsigned int j = 0; j < hmm2.N_; ++j) {
       GMM gmm2 = func(hmm2.BModels_[j]);

       //gmm2.print("gmm2");
       for (unsigned int i = 0; i < hmm1.N_; ++i) {
         switch(distanceSelect) {

           case 0:
             Sij(i,j) = expUnscentedTransform(hmm1.BModels_[i], gmm2);
             break;
           case 1:
             Sij(i,j) = std::exp(-gmmDistance(hmm1.BModels_[i], gmm2));
             break;
         }
         Qij(i,j) = pi1(i) * pi2(j) * Sij(i,j);
         ES += Qij(i,j);
       }
     }
     if (ES != 0.) {
       Qij /= ES;
     }
     Sij.print("Sij");
     Qij.print("Qij");
     arma::vec Hi = arma::zeros(hmm1.N_, 1);
     arma::vec Hj = arma::zeros(hmm2.N_, 1);

     for (unsigned int i = 0; i < hmm1.N_; ++i) {
       Hi(i) = normalizedGiniIndex(Qij.row(i).t());

       //Hi(i) = hoyerMeasure(Qij.row(i).t());
     }
     for (unsigned int j = 0; j < hmm2.N_; ++j) {
       Hj(j) = normalizedGiniIndex(Qij.col(j));
       //Hj(j) = hoyerMeasure(Qij.col(j));
     }
     Hi.print("Hi");
     Hj.print("Hj");

     double similarity = 0.5 * (1./double(hmm1.N_) * arma::accu(Hi) + 1./double(hmm2.N_) * arma::accu(Hj));;

     return ES / std::max(hmm1.N_, hmm2.N_) * similarity;

   }

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
     double eps = 1E-7;
     //v1_.elem(v1Zero) += eps;
     //v2_.elem(v2Zero) += eps;
     double distance = 0.0;
     for (unsigned int i = 0; i < v1.n_elem; ++i) {
       if (v1(i) <= eps) continue;
       distance += v1(i) * (std::log(v1(i)) - std::log(v2(i)));
     }

     //if (distance != distance) {

     //v1.print("v1");
     //v2.print("v2");
     //std::cout << "distance: " << distance << std::endl;
     //throw std::runtime_error("NaN detected");
     // }


     return distance;

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
     if (distance != distance) throw std::runtime_error("NaN detected");
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
     if (distance != distance) throw std::runtime_error("NaN detected");
     return 0.5 * distance;

   }


 double 
   static kld(const HMM & hmm1, const HMM & hmm2) {

     arma::rowvec pi1 = findStationary(hmm1.A_);
     arma::rowvec pi2 = findStationary(hmm2.A_);

     //double piDistance = pmfDistance(hmm1.pi_, hmm2.pi_);
     double piDistance = pmfDistance(pi1, pi2);
     if (hmm1.N_ != hmm2.N_) throw std::logic_error("different number of states in the hmms");

     unsigned int N = hmm1.N_;
     arma::vec d = arma::zeros(N);
     arma::vec e = arma::zeros(N);


     const arma::mat & A = hmm1.A_;
     const arma::mat & A_ = hmm2.A_;
     const std::vector<GMM> & B = hmm1.BModels_;
     const std::vector<GMM> & B_ = hmm2.BModels_;
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
     /* 
        if (!f.is_finite())  {
        d.print("d");
        e.print("e");
        f.print("f");
        throw std::runtime_error("NaN detected");
        }*/
     return piDistance + arma::as_scalar(pi1 * f);
     //return piDistance + arma::as_scalar(hmm1.pi_ * f);
   }

 static double
   symmetric_kld(const HMM & hmm1, const HMM & hmm2) {
     return kld(hmm1, hmm2) + kld(hmm2, hmm1);
   }
};



#endif
