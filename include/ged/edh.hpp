#ifndef __EDH_HPP__
#define __EDH_HPP__

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES
#endif

#include <tuple>
#include <array>
#include <algorithm>
#include <iterator>
#include <boost/foreach.hpp>

#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/filtered_graph.hpp>
#include "boost/graph/graph_utility.hpp"
//#include <boost/graph/copy.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/accumulators/statistics/weighted_median.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>

#include <utilities.hpp>
#include <datatypes.hpp>


using std::placeholders::_1;



struct
EuclideanVSpaceFunctor {

  //saving the angles to indices 1 -- N-1
  //0 holds the length
  template <typename Tag>
  typename std::enable_if<(std::tuple_size<Tag>::value > 1), Tag>::type
    operator() (Tag &&x, Tag &&y) const {
      Tag results;
      typename Tag::value_type dim0 = x[0] - y[0];
      results[0] = pow(dim0, std::tuple_size<Tag>::value);

      for (size_t i = 1; i < std::tuple_size<Tag>::value; ++i) {
        results[i] = atan2((x[i] - y[i]), dim0) ;
        results[0] += pow(results[i],std::tuple_size<Tag>::value);
      }
      results[0] = pow(results[0], 1./std::tuple_size<Tag>::value);  
      return results;
    }
};



//TagType consists of length, angle0, angle1, ...angleN-1
template <typename TagType, size_t M = 0, size_t N = std::tuple_size<TagType>::value>
class CalculateBins {

 private: 
  typedef typename TagType::value_type value_type;
  typedef std::vector<TagType> EdgePropType;
  typedef std::array<BinLimits<value_type>, N - M> Bins;
  Bins binLimits_; 
   
  struct FindMin {
    EdgePropType edgeProperties_;
    Bins & binLimits_;

    FindMin(const EdgePropType & edgeProperties, Bins & binLimits) : edgeProperties_(edgeProperties), binLimits_(binLimits) {}
    template <typename I>
    void operator() (I i) {
      std::sort(edgeProperties_.begin(), edgeProperties_.end(),
          [] (const TagType & x, const TagType & y) -> bool { return std::get<I::value>(x) < std::get<I::value>(y); } );
      
      
      value_type eps = 1.0E-6;

      value_type lowLim =   std::numeric_limits<value_type>::max();
      value_type binSize = std::numeric_limits<value_type>::max();
      value_type highLim = -std::numeric_limits<value_type>::max();

      value_type prev = std::numeric_limits<value_type>::min();
      
      std::for_each(edgeProperties_.begin(),edgeProperties_.end(), [&](const TagType & edge){

        const value_type & temp = std::get<I::value>(edge);

        lowLim = std::min(temp, lowLim);

        if ( (temp - prev) > eps) {
          binSize = std::min(temp - prev, binSize);
        }
        highLim = std::max(temp, highLim);
        
        prev = temp;
      });
      binLimits_[I::value].lowLim_ = lowLim;
      binLimits_[I::value].binSize_ = binSize;
      binLimits_[I::value].highLim_ = highLim;

    }
};
  friend class FindMin;

 public:
  CalculateBins( const std::vector<TagType> & edgeProperties ) {
    boost::mpl::for_each<boost::mpl::range_c<size_t,M,N> >(FindMin(edgeProperties, binLimits_));
  }

  auto
  apply() -> decltype(binLimits_) const {
    return binLimits_;
  }


};

template <typename TagType, typename B>
auto
createDenseHistogram(const std::vector<TagType> & edgeProperties, const B & binLimits_ = std::array<BinLimits<typename TagType::value_type>, 0> ()) ->
std::vector<typename TagType::value_type>
{
  //different Histogramfunctions should be considered, this version has its bins defined by the smallest difference between 2
  //objects.

  auto binLimits = (!binLimits_.empty()) ? binLimits_ : CalculateBins<TagType, 1>(edgeProperties).apply();
  
  enum { 
    numAngles = std::tuple_size<decltype(binLimits)>::value
  };
  
  std::array<std::size_t, numAngles > numBinsInDim; 
  std::size_t numBins = 1;
  for (size_t i = 0; i < numAngles; ++i) {
    numBinsInDim[i] = (size_t) ceil((binLimits[i].highLim_ - binLimits[i].lowLim_) / binLimits[i].binSize_);
    numBins *= numBinsInDim[i];
  }
 
  std::vector<typename TagType::value_type> bins(numBins);

//Put edges into bins, canonical hash function: equidistant bins.
//
  std::for_each(edgeProperties.cbegin(), edgeProperties.cend(),[&](const TagType& edge) {
    std::size_t index = 0;
    std::size_t offset = 1;
    for (size_t i = 0; i < numAngles; ++i) {  
      index += offset * (size_t) ((edge[i+1] - binLimits[i].lowLim_) / binLimits[i].binSize_);
      offset *= numBinsInDim[i];
    }
    bins[index] += edge[0];
  });
  return bins;
}



  template <class Graph, class VSpaceFunctor>
auto
prepareGraphForHist(const Graph & g, VSpaceFunctor && vsf)
-> std::vector<
typename std::result_of<
  VSpaceFunctor(decltype(boost::vertex_bundle_type<Graph>::type::location_),
                decltype(boost::vertex_bundle_type<Graph>::type::location_))
  >::type
>
{

  typedef typename boost::graph_traits<Graph>::edge_descriptor edgeDescriptor;
  auto numEdges = boost::num_edges(g);

  typedef typename boost::vertex_bundle_type<Graph>::type VertexType;
  typedef decltype(VertexType::location_) TagType;

  //edgeProperties' value_type is convertible from TagType, it's == TagType for most appliances.
  //std::vector<typename std::result_of<VSpaceFunctor(TagType, TagType)>::type> edgeProperties(numEdges);
  //TODO
  std::vector<TagType> edgeProperties(numEdges);


  std::transform(boost::edges(g).first, boost::edges(g).second, edgeProperties.begin() ,[&] (const edgeDescriptor & edge) {
    return vsf(boost::get(&VertexType::location_, g, boost::source(edge, g)), 
          boost::get(&VertexType::location_, g, boost::target(edge, g))
          );
  });


  return edgeProperties;
}



template <typename Hist>
Hist
normalizeHistogram(Hist && hist){
  
  typedef typename Hist::value_type histType;

  histType sum = 0;
  std::for_each(hist.begin(), hist.end(), [&](const histType & t) {
    sum += t;
  });
  histType inverseSum = (sum)?1./ sum : 1.;

  std::for_each(hist.begin(), hist.end(), [&](histType & t) {
      t *= inverseSum ;
      });

  return hist;
}

struct
EmdMod {
  //this only works for normalized cyclic histograms
  template <typename MassType>
    EmdResult
    operator() (const std::vector<MassType> & v1, const std::vector<MassType> & v2, size_t index1, size_t index2) {
      assert (v1.size() == v2.size());
      
      size_t N = v1.size();
      
      std::vector<MassType> q(v1);
      std::vector<MassType> p(v2);


      EmdResult result; 
      result.index1_ = index1;
      result.index2_ = index2;

      std::vector<MassType> f(N);
      MassType cumQ = 0;
      MassType cumP = 0;

      for (size_t i = 0; i < N; ++i) {
        cumQ += q[i];
        cumP += p[i];
        f[i] +=  cumQ - cumP;
      }


      boost::accumulators::accumulator_set<MassType, boost::accumulators::stats<boost::accumulators::tag::weighted_median(boost::accumulators::with_p_square_cumulative_distribution) >, MassType > 
        acc( boost::accumulators::tag::weighted_p_square_cumulative_distribution::num_cells = (size_t) (sqrt(double(N))) );



      for (size_t i = 0; i < N; ++i) {
        acc(i, boost::accumulators::weight = f[i]);
      }

      const size_t medianIndex = (size_t) ceil(boost::accumulators::weighted_median(acc)) % N;

      std::function<size_t (size_t)> indexCalc = [=] (size_t i) {return (i+medianIndex) % N;};

      size_t tQ = 0;
      size_t tP = 0;
      size_t iQ = indexCalc(tQ);
      size_t iP = indexCalc(tP);
      result.emd_ = 0.;
      while (1) {
        while (q[iQ] <= 0) {
          ++tQ;
          if (tQ == N) {
            result.emd_ /= (double) N;
            return result; 
          }
          iQ = indexCalc(tQ);
        }
        while (p[iP] <= 0) {
          ++tP;
          if (tP == N) {
            result.emd_ /= (double) N;
            return result; 
          }
          iP = indexCalc(tP);
        }
        double f = std::min(q[iQ], p[iP]);
        q[iQ] -= f;
        p[iP] -= f;
        size_t dist = size_t(abs(int(iQ) - int(iP)));
        size_t dMod = std::min<size_t>(dist,N - dist);
        result.emd_ += f * (double) dMod;
        result.fij_.push_back(Match(iQ, iP, f));
      }
      return result;
    }
};

//As of now, we will just use the proposed angle-representation, so normalized cyclic histograms are viable
template <typename EmdFunctor, typename Graph, typename ... Graphs >
auto
calcAngledEmdForGraphs(size_t angleGranularity, EmdFunctor && emdFunctor, Graph & g, Graphs & ... gs) ->
std::array<EmdResult, (sizeof...(Graphs) * (sizeof...(Graphs)+1)) / 2 >
{
  enum {
    NumGraphs = sizeof...(Graphs) + 1,
    NumResults = (sizeof...(Graphs) * (sizeof...(Graphs)+1)) / 2
  };

  //make an array, containing length/angle versions of the edges for all graphs.
  //auto test = std::make_tuple(prepareGraphForHist(g, EuclideanVSpaceFunctor())...);

  std::array<decltype(prepareGraphForHist(g, EuclideanVSpaceFunctor())), NumGraphs> edgeVectorArray = {{prepareGraphForHist(g, EuclideanVSpaceFunctor()), prepareGraphForHist(gs, EuclideanVSpaceFunctor())...}};

  //auto edgeVectorArray = prepareGraphForHist(g, EuclideanVSpaceFunctor())...);

  typedef decltype(edgeVectorArray) EdgeVectorArrayType;
  typedef typename EdgeVectorArrayType::value_type EdgeVectorType;

  //array
  typedef typename EdgeVectorType::value_type SingleEdgeType;

  // define bins, we will use equidistant bins, spanning all values that an angle can have (-Pi to +Pi), with a given
  // granularity which decides the binSize;
  //-1 because the length isn't needed for the histogram dimension
  std::array<BinLimits<typename SingleEdgeType::value_type>, std::tuple_size<SingleEdgeType>::value - 1> binLimits;
  std::for_each(binLimits.begin(), binLimits.end(), [=](BinLimits<typename SingleEdgeType::value_type> & singleAngleLimits) {
      singleAngleLimits.lowLim_ = - M_PI; 
      singleAngleLimits.binSize_ = 2. * M_PI / (double)angleGranularity;
      singleAngleLimits.highLim_ = M_PI;
      });

  typedef decltype(createDenseHistogram(*edgeVectorArray.begin(), binLimits)) HistogramType;
  std::array<HistogramType, NumGraphs> histogramArray;

  //create a histogram for every edgeVector
  
  for (size_t i = 0; i < NumGraphs; ++i) {
    histogramArray[i] = normalizeHistogram(createDenseHistogram(edgeVectorArray[i], binLimits));
  }

  MapHistogramsToGraphs<0>::map(histogramArray, g, gs...);

  std::array<EmdResult, NumResults> results;
  for (size_t i = 0, counter = 0; i < NumGraphs; ++i) {
    for (size_t j = i + 1; j < NumGraphs; ++j) {
      results[counter] = emdFunctor(histogramArray[i], histogramArray[j], i, j);
      ++counter;
    }
  }
  
  return results;

}



#endif
