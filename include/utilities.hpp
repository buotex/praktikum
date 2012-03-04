template<class T, class... Tail>
std::array<T,1+sizeof...(Tail)> make_array(T&& head, Tail&&... values)
{
  return { std::forward<T>(head), std::forward<Tail>(values)... };
}

template <size_t Index>
struct MapHistogramsToGraphs{

  template <typename HistogramArray, typename Graph, typename ... Graphs>
    typename std::enable_if<!std::is_same<typename boost::graph_bundle_type<Graph>::type, boost::no_property>::value
    ,void>::type
    static map(const HistogramArray & histArray, Graph & graph, Graphs &  ... graphs) {
       graph[boost::graph_bundle].histogram_ = histArray[Index];
       MapHistogramsToGraphs<Index + 1>::map(histArray, graphs...);
    }

  template <typename HistogramArray, typename Graph, typename ... Graphs>
    typename std::enable_if<std::is_same<typename boost::graph_bundle_type<Graph>::type, boost::no_property>::value
    ,void>::type
    static map(const HistogramArray & histArray, Graph & graph, Graphs &  ... graphs) {
       MapHistogramsToGraphs<Index + 1>::map(histArray, graphs...);
    }
  template <typename HistogramArray>
static void map(const HistogramArray & histArray){}

};





