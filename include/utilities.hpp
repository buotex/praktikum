#include <datatypes.hpp>
#ifndef _UTILITIES_HPP__
#define _UTILITIES_HPP__



template<class T, class... Tail>
std::array<T,1+sizeof...(Tail)> make_array(T&& head, Tail&&... values)
{
  return { std::forward<T>(head), std::forward<Tail>(values)... };
}

template <size_t Index>
struct MapHistogramsToGraphs{

  template <typename HistogramArray, typename Graph, typename ... Graphs>
    void
    static map(const HistogramArray & histArray, Graph & graph, Graphs &  ... graphs) {
       assign(graph[boost::graph_bundle], histArray[Index]);
       MapHistogramsToGraphs<Index + 1>::map(histArray, graphs...);
    }
  template <typename Bundle, typename Histogram>
  static 
  void
  assign(Bundle &, const Histogram & hist){}

  template <typename Histogram>
  static 
  void
  assign(EmdGraphPropertyBundle<Histogram> & bundle, const Histogram & hist) {
    bundle.histogram_ = hist;
}

  template <typename HistogramArray>
static void map(const HistogramArray & histArray){}

};
 
 template<class, class ...> // not defined
 struct SameTypes;

 template<class ValueT>
 struct SameTypes<ValueT> : std::true_type
 { };

 template<class FirstT, class SecondT, class ... ArgTypes>
 struct SameTypes<FirstT, SecondT, ArgTypes...>
 : public std::integral_constant<bool,
 std::is_same<FirstT, SecondT>::value &&
 SameTypes<FirstT, ArgTypes...>::value >
{ };


#endif
