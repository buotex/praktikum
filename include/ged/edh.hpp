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

  template<typename Array>
  struct ArrayTypeHelper {
    typedef typename Array::value_type value_type;
    static constexpr size_t N = std::tuple_size<Array>::value;
    typedef value_type WeightType;
    typedef std::array<value_type, N-1> MassType;
    typedef std::pair<WeightType, MassType> ResultType;
  };
  //saving the angles to indices 1 -- N-1
  //0 holds the length
  template <typename EdgeDescriptor, typename Graph>
    auto
    operator() (EdgeDescriptor && e, const Graph & g) const ->
    typename ArrayTypeHelper<decltype(boost::vertex_bundle_type<Graph>::type::location_)>::ResultType
    {

      typedef typename boost::vertex_bundle_type<Graph>::type VertexType;
      typedef decltype(VertexType::location_) Tag;
      typedef typename ArrayTypeHelper<decltype(VertexType::location_)>::ResultType ResultType;
      static constexpr size_t N = ArrayTypeHelper<decltype(VertexType::location_)>::N ;

      ResultType result;
      const Tag & x = boost::get(&VertexType::location_, g, boost::source(e, g));
      const Tag & y = boost::get(&VertexType::location_, g, boost::target(e, g));

      typename Tag::value_type dim0 = x[0] - y[0];
      result.first = pow(dim0, std::tuple_size<Tag>::value);

      for (size_t i = 0; i < N-1; ++i) {
        result.second[i] = atan2((x[i+1] - y[i+1]), dim0) ;
        result.first += pow(result.second[i],N);
      }
      result.first = pow(result.first, 1./N);  
      return result;
    }
};



//TagType consists of length, angle0, angle1, ...angleN-1
template <typename TagType, size_t M = 0, size_t N = std::tuple_size<TagType>::value>
class CalculateBins {

  private: 
    typedef typename TagType::value_type value_type;
    typedef std::array<BinLimits<value_type>, N - M> Bins;
    Bins binLimits_; 

    template <typename ForwardIterator>
    struct FindMin {
      ForwardIterator first_;
      ForwardIterator last_;
      Bins & binLimits_;

      FindMin(ForwardIterator first, ForwardIterator last, Bins & binLimits) : first_(first),last_(last), binLimits_(binLimits) {}
      template <typename I>
        void operator() (I i) {
          std::vector<typename std::iterator_traits<ForwardIterator>::value_type> edgeProperties(first_, last_);

          std::sort(edgeProperties.begin(), edgeProperties.end(),
              [] (const TagType & x, const TagType & y) -> bool { return std::get<I::value>(x) < std::get<I::value>(y); } );


          value_type eps = 1.0E-6;

          value_type lowLim =   std::numeric_limits<value_type>::max();
          value_type binSize = std::numeric_limits<value_type>::max();
          value_type highLim = -std::numeric_limits<value_type>::max();

          value_type prev = std::numeric_limits<value_type>::min();

          std::for_each(edgeProperties.cbegin(),edgeProperties.cend(), [&](const TagType & edge){

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

  public:
    template <typename ForwardIterator>
    CalculateBins(ForwardIterator first, ForwardIterator last) {
      boost::mpl::for_each<boost::mpl::range_c<size_t,M,N> >(FindMin<ForwardIterator>(first, last, binLimits_));
    }

    auto
      apply() -> decltype(binLimits_) const {
        return binLimits_;
      }


};
struct
CreateDenseHistogram {
  //TODO: crashes in higher dimensions!?
  template <typename ForwardIterator, typename B>
    auto
    operator()(ForwardIterator first, ForwardIterator last, const B & binLimits, bool normalize = true) -> 
 std::vector<std::pair<typename std::iterator_traits<ForwardIterator>::value_type::first_type, std::list<size_t> > >
    {
      //different Histogramfunctions should be considered, this version has its bins defined by the smallest difference between 2
      //objects.

      typedef typename std::iterator_traits<ForwardIterator>::value_type TagType;

      typedef typename TagType::first_type WeightType;
      typedef typename TagType::second_type MassType;
      typedef std::vector<std::pair<WeightType, std::list<size_t> > > HistogramType;

      enum { 
        numDims = std::tuple_size<MassType>::value
      };

      std::array<std::size_t, numDims > numBinsInDim; 
      std::size_t numBins = 1;
      for (size_t i = 0; i < numDims; ++i) {
        numBinsInDim[i] = (size_t) ceil((binLimits.highLim_[i] - binLimits.lowLim_[i]) / binLimits.binSize_[i]);
        numBins *= numBinsInDim[i];
      }

      HistogramType bins(numBins);

      WeightType invertSum = 0.;
      if (normalize) {
        WeightType sum = 0.;
        std::for_each(first, last, [&sum] (const TagType & tag)->WeightType{return sum += tag.first; });
        invertSum = (sum)?1./ sum:1.;
      } else {
        invertSum = 1.;
      }

      //Put edges into bins, canonical hash function: equidistant bins.
      //

      size_t counter = 0;
      for(; first != last; ++first) {
        const TagType & edge = *first;

        std::size_t index = 0;
        std::size_t offset = 1;

        for (size_t i = 0; i < numDims; ++i) {  
          index += offset * ((size_t) ((edge.second[i] - binLimits.lowLim_[i]) / binLimits.binSize_[i]) % numBinsInDim[i]);
          offset *= numBinsInDim[i];
        }

        //std::cout << index << std::endl;
        bins[index].first += edge.first * invertSum;
        bins[index].second.push_back(counter);
        ++counter;
      }
      return bins;
    }
};


template <typename EdgeExtractor, typename Graph>
struct
EmdMod {

  const size_t N_;
  EmdMod(size_t granularity):N_(granularity) {}

  typedef typename boost::graph_traits<Graph>::edge_descriptor edge_descriptor;
  typedef typename std::result_of<EdgeExtractor(edge_descriptor &&, const Graph &)>::type ExtractedEdgeType;
  typedef typename ExtractedEdgeType::second_type MassType;
  typedef typename std::vector<ExtractedEdgeType>::const_iterator ForwardIterator;

  typedef typename std::result_of<CreateDenseHistogram(ForwardIterator, ForwardIterator, const BinLimits<MassType> &, bool)>::type HistogramType;
  typedef typename HistogramType::value_type::first_type WeightType;
  typedef EmdResult<Graph,size_t> ResultType;

  template <typename ForwardIterator>
    HistogramType
    createHistogram(ForwardIterator first, ForwardIterator last) {

      BinLimits<MassType> binLimits({{-M_PI}}, {{2. * M_PI/(double) N_}}, {{M_PI}});
      HistogramType hist = CreateDenseHistogram()(first, last, binLimits, true);
      return hist;
    }


  ResultType
    calcEmd (const HistogramType & v1, const HistogramType & v2) {

      std::vector<WeightType> q(N_);
      std::vector<WeightType> p(N_);
      std::transform(v1.cbegin(), v1.cend(), q.begin(), [](const typename HistogramType::value_type& val){return val.first;} );
      std::transform(v2.cbegin(), v2.cend(), p.begin(), [](const typename HistogramType::value_type& val){return val.first;} );

      ResultType result; 

      std::vector<WeightType> f(N_);
      WeightType cumQ = 0;
      WeightType cumP = 0;

      for (size_t i = 0; i < N_; ++i) {
        cumQ += q[i];
        cumP += p[i];
        f[i] +=  cumQ - cumP;
      }


      boost::accumulators::accumulator_set<WeightType, boost::accumulators::stats<boost::accumulators::tag::weighted_median(boost::accumulators::with_p_square_cumulative_distribution) >, WeightType> 
        acc( boost::accumulators::tag::weighted_p_square_cumulative_distribution::num_cells = (size_t) (sqrt(double(N_))) );

      for (size_t i = 0; i < N_; ++i) {
        acc(i, boost::accumulators::weight = f[i]);
      }

      const size_t medianIndex = (size_t) ceil(boost::accumulators::weighted_median(acc)) % N_;

      std::function<size_t (size_t)> indexCalc = [=] (size_t i) {return (i+medianIndex) % N_;};

      size_t tQ = 0;
      size_t tP = 0;
      size_t iQ = indexCalc(tQ);
      size_t iP = indexCalc(tP);
      result.emd_ = 0.;
      while (1) {
        while (q[iQ] <= 0) {
          ++tQ;
          if (tQ == N_) {
            result.emd_ /= (double) N_;
            return result; 
          }
          iQ = indexCalc(tQ);
        }
        while (p[iP] <= 0) {
          ++tP;
          if (tP == N_) {
            result.emd_ /= (double) N_;
            return result; 
          }
          iP = indexCalc(tP);
        }
        double f = std::min(q[iQ], p[iP]);
        q[iQ] -= f;
        p[iP] -= f;
        size_t dist = size_t(abs(int(iQ) - int(iP)));
        size_t dMod = std::min<size_t>(dist,N_ - dist);
        result.emd_ += f * (double) dMod;
        result.fijVector_.push_back(Match<size_t>(iQ, iP, f));
      }
      return result;
    }
};


struct EMD {

  template <typename EdgeExtractor, class EmdVariantFunctor, typename Graph>
    struct Traits { 
      typedef typename std::remove_reference<EdgeExtractor>::type EE;
      typedef typename std::remove_reference<EmdVariantFunctor>::type EVF;
      typedef typename std::remove_reference<Graph>::type G;
      typedef typename std::result_of<EE(typename boost::graph_traits<G>::edge_descriptor &&, const G &)>::type ExtractedEdgeType;
      typedef std::vector<ExtractedEdgeType> HistogramInputType;
      typedef typename EVF::HistogramType HistogramType;
      typedef typename EVF::ResultType ResultType;
    };
  template <typename EdgeExtractor, template <class, class> class EmdVariantFunctor, typename Graph> 
    struct TraitsHelper:Traits<EdgeExtractor, EmdVariantFunctor<EdgeExtractor, Graph>, Graph> {};

  template <class EdgeExtractor, typename EmdVariantFunctor, typename Graph>
    auto
    static prepareGraphForHist(EdgeExtractor && vsf, EmdVariantFunctor &&, const Graph & g)
    -> typename Traits<EdgeExtractor, EmdVariantFunctor, Graph>::HistogramInputType
    {

      typedef typename boost::graph_traits<Graph>::edge_descriptor edgeDescriptor;

      auto numEdges = boost::num_edges(g);

      typename Traits<EdgeExtractor, EmdVariantFunctor, Graph>::HistogramInputType edgeProperties(numEdges);

      std::transform(boost::edges(g).first, boost::edges(g).second, edgeProperties.begin() ,[&] (const edgeDescriptor & edge) {
          return vsf(edge, g);
          });
      return edgeProperties;
    }





  template <typename EdgeExtractor, typename EmdVariantFunctor, typename Graph, typename ... Graphs>
    static
    auto 
    calcEmdForGraphs(EdgeExtractor && ee, EmdVariantFunctor && emdFunctor, Graph & g, Graphs & ... gs) ->
    typename std::enable_if<SameTypes<Graph, Graphs...>::value, 
             std::array<typename Traits<EdgeExtractor, EmdVariantFunctor, Graph>::ResultType, (sizeof...(Graphs) * (sizeof...(Graphs)+1)) / 2 >
               >::type {
                 enum {
                   NumGraphs = sizeof...(Graphs) + 1,
                   NumResults = (sizeof...(Graphs) * (sizeof...(Graphs)+1)) / 2
                 };
                 typedef typename Traits<EdgeExtractor, EmdVariantFunctor, Graph>::HistogramInputType HistogramInputType;
                 typedef typename Traits<EdgeExtractor, EmdVariantFunctor, Graph>::HistogramType HistogramType;
                 typedef typename Traits<EdgeExtractor, EmdVariantFunctor, Graph>::ResultType ResultType;

                 std::array<Graph*, NumGraphs> graphPointers = {{&g, &gs...}};

                 std::array<HistogramInputType, NumGraphs> edgeVectorArray = {{prepareGraphForHist(ee,  emdFunctor, g), prepareGraphForHist(ee, emdFunctor, gs)...}};

                 std::array<HistogramType, NumGraphs> histogramArray;
                 std::transform(edgeVectorArray.cbegin(), edgeVectorArray.cend(), histogramArray.begin(), [&emdFunctor](const HistogramInputType & hist) {return emdFunctor.createHistogram(hist.cbegin(), hist.cend());});

                 MapHistogramsToGraphs<0>::map(histogramArray, g, gs...);

                 std::array<ResultType, NumResults> results;
                 for (size_t i = 0, counter = 0; i < NumGraphs; ++i) {
                   for (size_t j = i + 1; j < NumGraphs; ++j) {
                     results[counter] = emdFunctor.calcEmd(histogramArray[i], histogramArray[j]);
                     results[counter].g1_ = graphPointers[i];
                     results[counter].g2_ = graphPointers[j];

                     ++counter;
                   }
                 }
                 return results;
               }
};








#endif
