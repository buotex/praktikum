#ifndef __INCLUDE_PDB_UTILITIES_HPP__
#define __INCLUDE_PDB_UTILITIES_HPP__
#include <ESBTL/default.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <armadillo>
#include <iterator>


#include <ESBTL/weighted_atom_iterator.h>


arma::mat
createMatrix(const boost::filesystem::path & path) {

  arma::mat mat;

  typedef ESBTL::Accept_none_occupancy_policy<ESBTL::PDB::Line_format<> > Accept_none_occupancy_policy;

  ESBTL::PDB_line_selector_two_systems sel;

  std::vector<ESBTL::Default_system> systems;
  ESBTL::All_atom_system_builder<ESBTL::Default_system> builder(systems,sel.max_nb_systems());

  if (ESBTL::read_a_pdb_file(argv[1],sel,builder,Accept_none_occupancy_policy())){

    if ( systems.empty() || systems[0].has_no_model() ){
      std::cerr << "No atoms found" << std::endl;
      return mat;
    }
  const ESBTL::Default_system::Model& model=* systems[0].models_begin();
    mat = arma::mat(3, std::distance(model.atom_begin(), model.atoms_end()));
    arma::mat::iterator mat_it = mat.begin();
    for (ESBTL::Default_system::Model::Atoms_const_iterator it_atm=model.atoms_begin();it_atm!=model.atoms_end();++it_atm){
      std::cout << it_atm->x() << " " << it_atm->y() << " " << it_atm->z() << " " ;
      *(mat_it++) = it_atm->x();
      *(mat_it++) = it_atm->y();
      *(mat_it++) = it_atm->z();
    }
  }
  else
    return mat;
  }
  mat.print("mat");

  return mat;
}






#endif
