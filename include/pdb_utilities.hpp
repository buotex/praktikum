#ifndef __INCLUDE_PDB_UTILITIES_HPP__
#define __INCLUDE_PDB_UTILITIES_HPP__
#include <ESBTL/default.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <armadillo>
#include <iterator>
#include <functional>
#include <hmm/kmeans.hpp>
#include <hmm/hmm.hpp>


#include <ESBTL/weighted_atom_iterator.h>

/** 
 *  \brief a Method to extract the first system of a PDB file
 *
 *  \param[in] path path to a pdb file
 *  \return A System, created by ESBTL, used for further parsing.
 *
 * */
ESBTL::Default_system
parsePdb(const boost::filesystem::path & path) {


  typedef ESBTL::Accept_none_occupancy_policy<ESBTL::PDB::Line_format<> > Accept_none_occupancy_policy;

  ESBTL::PDB_line_selector_two_systems sel;

  std::vector<ESBTL::Default_system> systems;
  ESBTL::All_atom_system_builder<ESBTL::Default_system> builder(systems,sel.max_nb_systems());

  if (ESBTL::read_a_pdb_file(path.c_str(),sel,builder,Accept_none_occupancy_policy())){

    if ( systems.empty() || systems[0].has_no_model() ){
      std::cerr << "No atoms found" << std::endl;
    }
    return systems[0];
  }
  std::cerr << "Problems reading the file" << std::endl;
  return systems[0];
}

/** 
 *  \brief a Functor to apply extract data from a given model and label it with the KMeans algorithm.
 *  
 *  
 *
 * */


struct
KMeansFunctor {
  private:
 
  /** ParameterPack of necessary parameters for KMeans, it's enough to pass an integer for the number of clusters
   * though 
   * */
  
  const KMeansParams params_;
  /** An optional parameter, enabling the use of a transformation function used on the data*/
  std::function<arma::mat (const arma::mat& )> transformation_;
  public:

  /** Most basic constructor, sets default parameters for everything other than the number of clusters*/
  KMeansFunctor(size_t numClusters): params_(KMeansParams(numClusters)), transformation_([](const arma::mat & mat) {return mat;}) {}
  /** A Basic constructor, pass a ParameterPack \see KMeansParams */
  KMeansFunctor(const KMeansParams & p): params_(p), transformation_([](const arma::mat & mat) {return mat;}) {}
  /** It is possible to pass a function which will transform the data before doing the labeling, for normalizing or
   * other things.*/
  KMeansFunctor(const KMeansParams & p, const std::function<arma::mat (const arma::mat& )> trans): params_(p), transformation_(trans) {}
  
  /** Spawn another Object which uses the same parameters but with an updated transformation function*/
  KMeansFunctor bind(const std::function<arma::mat (const arma::mat& )> & trans) {
    KMeansFunctor newFunc(params_, trans);
    return newFunc;
  }


  /** \brief Extracts the data and passes it to the kmeans algorithm
   * \param[in] model An ESBTL Protein model
   * \param[in] An armadillo matrix, will get overwritten
   * \return A labeling vector
   * */
  arma::urowvec
    operator() (const ESBTL::Default_system::Model& model, arma::mat & data) {
      data = arma::mat(3, (unsigned int) std::distance(model.atoms_begin(), model.atoms_end()));
      auto mat_it = data.begin();
      for (ESBTL::Default_system::Model::Atoms_const_iterator it_atm=model.atoms_begin();it_atm!=model.atoms_end();++it_atm){
        *(mat_it++) = it_atm->x();
        *(mat_it++) = it_atm->y();
        *(mat_it++) = it_atm->z();
      }
      data = transformation_(data);
      arma::urowvec labels = kmeans(data, params_);
      return labels;
    }
};


/** 
 *  \brief Extracts the data from a model and then labels it according to the protein chains.
 *
 * */
struct
ProteinChainFunctor {

  /** \brief Extracts the data and passes it to the kmeans algorithm
   * \param[in] model An ESBTL Protein model
   * \param[in] An armadillo matrix, will get overwritten
   * \return A labeling vector
   */
  arma::urowvec
    operator() (const ESBTL::Default_system::Model& model, arma::mat & data) {
      data = arma::mat(3, (unsigned int) std::distance(model.atoms_begin(), model.atoms_end()));
      arma::urowvec labels = arma::urowvec(data.n_cols);
      unsigned int * label_it = labels.begin();
      unsigned int label = 0;
      auto mat_it = data.begin();
      for (ESBTL::Default_system::Model::Chains_const_iterator it_cha=model.chains_begin();it_cha!=model.chains_end();++it_cha){
        for (ESBTL::Default_system::Model::Chain::Atoms_const_iterator it_atm = it_cha->atoms_begin(); it_atm != it_cha->atoms_end(); ++it_atm) {
          *(mat_it++) = it_atm->x();
          *(mat_it++) = it_atm->y();
          *(mat_it++) = it_atm->z();
          *(label_it++) = label;
        }
        ++label;
      }
      return labels;
    }
};


/**
 *  \brief A method for parsing the systems file into an armadillo matrix and a label vector
 *  
 *  \param[in] ClusteringFunctor
 *  \param[out] data A Matrix containing an atom in every column.
 *  \param[out] labels Every atom in the protein gets assigned a cluster-id, according to the clustering scheme
 *
 * */
template <typename ClusteringFunctor>
HMM
buildHMM(const ESBTL::Default_system & system, GMMCreator creator, ClusteringFunctor func) {
  arma::mat data;
  arma::urowvec labels;
  HMM hmm;
  unsigned int offset = 0;
  for (ESBTL::Default_system::Models_const_iterator it_model = system.models_begin();
      it_model != system.models_end();
      ++it_model) {
    const ESBTL::Default_system::Model & model = *it_model;
    arma::mat tempdata;
    arma::urowvec templabels = func(model, tempdata); 
    templabels += offset;
    data = arma::join_rows(data, tempdata);
    labels = arma::join_rows(labels, templabels);
    offset = arma::max(labels);
  }
  hmm.baumWelchCached(data, creator(data, labels));
  return hmm;
}


#endif
