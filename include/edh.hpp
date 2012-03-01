#ifndef __EDH_HPP__
#define __EDH_HPP__

#include <cmath>

#include <tuple>
#include <array>
#include <algorithm>
#include <boost/foreach.hpp>

#include <config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include "boost/graph/graph_utility.hpp"
#include <boost/graph/copy.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>

using std::placeholders::_1;


struct
EuclideanVSpaceFunctor {

  //saving the angles to indices 1 -- N-1
  //0 holds the length
  template <typename Tag>
  typename std::enable_if<(std::tuple_size<Tag>::value > 1), Tag>::type
    Tag operator() (const Tag & x, const Tag & y) const {
      static_assert(x.size() == y.size(), "objects do not have the same dimensionality");
      Tag results;
      typename Tag::value_type dim0 = x[0] - y[0];
      results[0] = pow(dim0, std::tuple_size<Tag>::value);

      for (int i = 1; i < std::tuple_size<Tag>::value; ++i) {
        results[i] = atan2((x[i] - y[i]), dim0) ;
        results[0] += pow(results[i],std::tuple_size<Tag>::value);
      }
      results[0] = pow(results[0], 1./std::tuple_size<Tag>::value);  
      return results;
    }
}
template <typename value_type>
struct BinLimits {
  value_type lowLim_;
  value_type stepSize_;
  value_type highLim_;
};

//TagType consists of length, angle0, angle1, ...angleN-1
template <typename TagType, int M = 0, int N = std::tuple_size<TagType>::value>
class calculateBins {

 private: 
  typedef typename TagType::value_type value_type;

  template <typename I>
    void 
    findMin(I) {
      std::sort(edge_properties_.begin(), edge_properties_.end(),
          [] (const TagType & x, const TagType & y) -> bool { return std::get<I::value>(x) < std::get<I::value>(y); } );
      
      
      value_type eps = 1.0E-6;

      value_type lowLim =   std::numeric_limits<value_type>::max();
      value_type stepSize = std::numeric_limits<value_type>::max();
      value_type highLim = -std::numeric_limits<value_type>::max();

      value_type prev = std::numeric_limits<value_type>::min();
      
      BOOST_FOREACH(const TagType & edge, edge_properties) {
        
        const value_type & temp = std::get<I::value>(edge);

        lowLim = std::min(temp, lowLim);

        if ( (temp - prev) > eps) {
          stepSize = std::min(temp - prev, stepSize);
        }
        highLim = std::max(temp, highLim);
        
        prev = temp;
      }
      binLimits_[I::value].lowLim_ = lowLim;
      binLimits_[I::value].stepSize_ = stepSize;
      binLimits_[I::value].highLim_ = highLim;

    }

  std::vector<TagType> & edge_properties_;
  std::array<BinLimits<value_type>, N - M> binLimits_; 

 public:
  findDiffs( std::vector<TagType> & edge_properties ): edge_properties_(edge_properties) {
    boost::mpl::for_each<boost::mpl::range_c<int,M,N> >(findMin);
  }

  apply() -> decltype(binLimits_) const {
    return binLimits_;
  }


}

template <typename TagType, typename B>
std::vector<int>
createHistogram(std::vector<TagType> & edge_properties, const B & binLimits_ = std::array<BinLimits<typename TagType::value_type>, 0> ()) {
  //different Histogramfunctions should be considered, this version has its bins defined by the smallest difference between 2
  //objects.

  auto binLimits = (!binLimits_.empty()) ? binLimits_ : calculateBins<TagType, 1>(edge_properties).apply();
  
  const static int numAngles = std::tuple_size<decltype(binLimits)>::value;
  std::array<std::size_t, numAngles > numBinsInDim; 
  std::size_t numBins = 1;
  for (int i = 0; i < numAngles; ++i) {
    numBinsInDim[i] = (size_t) ceil((binLimits[i].highLim_ - binLimits[i].lowLim_) / binLimits[i].stepSize_);
    numBins *= numBinsInDim[i];
  }
 
  std::vector<double> bins(numBins);

//Put edges into bins, canonical hash function: equidistant bins.
//
  BOOST_FOREACH(const TagType & edge, edge_properties) {
    std::size_t index = 0;
    std::size_t offset = 1;
    for (int i = 0; i < numAngles; ++i) {  
      index += offset * ((edge[i+1] - binLimits[i].lowLim_) / binLimits[i].stepSize_);
      offset *= numBinsInDim[i];
    }
    bins[index] += edge[0];
  }

  return bins;
}



  template <class Graph, class VSpaceFunctor>
std::vector<int> hist
calcEdhForGraph(const Graph & g, VSpaceFunctor & vsf)
{

  typedef typename boost::graph::graph_traits<Graph>::edge_descriptor edge_descriptor;
  auto num_edges = boost::graph::num_edges(g);

  typedef typename property_traits<property_map<Graph, vertex_location>::const_type>::value_type TagType;
  //edge_properties' value_type is convertible from TagType, it's == TagType for most appliances.
  std::vector<typename std::result_of<VSpaceFunctor(TagType, TagType)>::type> edge_properties(num_edges);
  decltype(num_edges) current_edge = 0;


  BOOST_FOREACH(const edge_descriptor & edge, g.edges()) {

    edge_properties.insert(
        vsf(boost::graph::get(vertex_location, g, boost::graph::source(edge, g)), 
          boost::graph::get(vertex_location, g, boost::graph::target(edge, g))
          ),
        current_edge
        );
    current_edge++;
  }



  std::vector<int> hist = createHistogram(edge_properties);
  return hist;
}

template <class Graph>
void 
normalizeGraph(Graph & g) {
  typedef typename boost::graph::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
  typedef typename property_traits<property_map<Graph, vertex_location>::const_type>::value_type TagType;
  
  //scan all vertices to get the min/max values;
  std::array<typename TagType::value_type, 2> limits;
  limits[0] = std::numeric_limits<typename TagType::value_type>::max();
  limits[1] = -std::numeric_limits<typename TagType::value_type>::max();

  BOOST_FOREACH(vertex_descriptor & vertex, g.vertices()) {
    TagType & v = boost::graph::get(vertex_location, vertex);
    for (int i = 0; i < std::tuple_size<TagType>::value; ++i) {
      limits[0] = std::min(limits[0],v[i] );
      limits[1] = std::max(limits[1],v[i] );
    }
  }
  typename TagType::value_type scalingFactor = 1./(limits[1] - limits[0]);

  BOOST_FOREACH(vertex_descriptor & vertex, g.vertices()) {
    TagType v = boost::graph::get(vertex_location, vertex);
    
    for (int i = 0; i < std::tuple_size<TagType>::value; ++i) {
      v[i] *= scalingFactor;
    }
    boost::graph::put(vertex_location, g, vertex, v);
  }
}


calcEmdForGraphs() {
  


  
  }



#endif
