#include "unittest.hxx"

#include <random>
#include "ged/edh.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <ctime>

using namespace boost::graph;

using namespace vigra;

struct EmdByEdhTestSuite : vigra::test_suite {
  EmdByEdhTestSuite() : vigra::test_suite("EmdByEdh") {
    add(testCase(&EmdByEdhTestSuite::test));
  }
  void test() {
    //undirected graph, using NodeWithLocation as a property for the nodes
    typedef NodeWithLocation<double, 2> NodeProperty;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,  
            NodeProperty, boost::no_property, EmdGraphPropertyBundle > Graph;
    srand((int)time(NULL));

    Graph g, h, i, j;
    for (int iter = 0 ; iter < 1000; ++iter) { 
      boost::add_vertex(NodeProperty({{double(rand() % 50), double(rand() % 50)}}), g);

    }
    size_t num_v = boost::num_vertices(g);
    for (int iter = 0; iter < 1000; ++iter) {
      boost::add_edge(rand() %num_v ,rand() % num_v,g);
    }
    copy_graph(g,h);
    copy_graph(g,i);
    copy_graph(g,j);
    if (num_vertices(h) != num_vertices(g)) {
      failTest("The vertices weren't copied correctly"); 
    }
    //std::array<double, 2> testArray = {{5.0,2.0}};
    //put(&NodeProperty::location_, g,0, testArray);
    for (size_t iter = 0; iter < num_vertices(h); ++iter) {
      auto x = get(&NodeProperty::location_, g, iter);
      auto y = get(&NodeProperty::location_, h, iter);
      if (x[0] != y[0] || x[1] != y[1]) {
        std::cout << iter << "\n" << "dim0 " << x[0] << " " <<y[0] << "\n" << "dim1 "
          << x[1] << " " << y[1] << "\n";
        failTest("The vertex-locations weren't copied correctly");
      }
    }

    for (int iter = 0 ; iter < 100; ++iter) { 
      boost::add_vertex(NodeProperty({{double(rand() % 50), double(rand() % 50)}}), g);

    }
    for (int iter = 0; iter < 100; ++iter) {
      boost::add_edge(rand() %boost::num_vertices(g) ,rand() % boost::num_vertices(g),g);
    }
    
    
    for (size_t iter = 0; iter < num_vertices(h); ++iter) {
      std::array<double, 2> testArray = get(&NodeProperty::location_, h, iter);
      testArray[0] += 100.0;
      put(&NodeProperty::location_, h,iter, testArray);

    }

    for (size_t iter = 0; iter < num_vertices(i); ++iter) {
      std::array<double, 2> testArray = get(&NodeProperty::location_, i, iter);
      testArray[0] *= 100.0;
      testArray[1] *= 100.0;
      put(&NodeProperty::location_, i,iter, testArray);

    }


    auto results = calcAngledEmdForGraphs(10, EmdMod(), g, h, i, j);
    for (size_t iter = 0; iter < std::tuple_size<decltype(results)>::value; ++iter) {
      std::cout << results[iter];
    }



  }

};

int main() {
  EmdByEdhTestSuite test;
  int success = test.run();
  std::cout << test.report() << std::endl;
  return success;


}

