#include <boost/graph/graph_utility.hpp>

#include <vector>
#include <map>

template <typename Graph, typename std::enable_if<std::is_same<typename Graph::vertex_list_selector, boost::vecS>::value,void >::type>
std::vector<std::size_t> chineseWhisper(const Graph & g, size_t numIterations) {
  size_t N = boost::num_vertices(g);
  auto vertIterPair = boost::vertices(g);
  typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
  std::vector<vertex_descriptor> descriptors(N);
  std::copy(vertIterPair.first, vertIterPair.second, descriptors.begin());

  std::vector<std::size_t> labels(N);

  for (size_t i = 0; i < N; ++i) labels[i] = i;

  for (size_t i = 0; i < numIterations; ++i) { 
    std::random_shuffle(descriptors.begin(), descriptors.end());
    std::for_each(descriptors.begin(), descriptors.end(), [&g, &labels](vertex_descriptor desc) {
     std::map<size_t, size_t> labelcounts;
     auto invIterPair = boost::inv_adjacent_vertices(desc, g);
     std::for_each(invIterPair.first, invIterPair.second, [&labelcounts](vertex_descriptor invDesc){
       ++labelcounts[invDesc];
       });
     auto labelIter = std::max_element(labelcounts.begin(), labelcounts.end());
     labels[desc] = labelIter->first;
    });
  }
  return labels;

}
