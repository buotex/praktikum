
template <typename ValType>
struct ArrayFunctor {
  template <size_t N>
  ValType & addTo(std::array<ValType, N> & arrX, const std::array<ValType, N> & arrY){
    std::for_each(arrX.begin(), arrX.end(), arrY.cbegin(), [ValType & varX, const ValType & varY] {varX += varY;});
    return arrX;
  }
  template <size_t N, typename numericType>
  ValType & divideBy(std::array<ValType, N> & arrX, numericType divisor) {
    std::for_each(arrX.begin(), arrX.end(), [ValType & varX] {varX /= divisor});
    return arrX;
  }
  template <size_t N>
  ValType distance(const std::array<ValType, N> & arrX, const std::array<ValType, N> & arrY) {
    
    ValType result = 0;
    std::for_each(arrX.cbegin(), arrX.cend(), arrY.cbegin(), 
                  [&result](const ValType & varX, const ValType & varY){result += pow((varX - varY), N);});

    
    return pow(result, 1./N);
  }

  template <size_t N, typename std::enable_if<std::is_same<ForwardIterator::value_type, ValType> >::type >
  size_t findClosest(const std::array<ValType, N> & location, ForwardIterator first, ForwardIterator last) {
    
    size_t label = 0;
    ValType minVal = distance(*first, location);
    ++first;
    counter = 1;
    for (; first != last; ++first, ++counter) {
      if (distance(*first, location) < minVal) {
        minVal = distance(*first, location);
        label = counter;
      }
    }
    return label;
  }
};

template <typename ForwardIterater, typename LocationFunctor>
std::vector<size_t>
kmeans(ForwardIterator first, ForwardIterator last, size_t numClusters, size_t maxIterations, LocationFunctor && func = ArrayFunctor<typename ForwardIterator::value_type>())

  typedef typename ForwardIterator::value_type LocationType;
  std::vector<size_t> labels(std::distance(last,first));


  //fill with random clusternumbers

  std::mt19937 rSeedEngine;
  typedef std::uniform_int_distribution<size_t> Distribution;
  std::for_each(labels.begin(), labels.end(), [&] (size_t & label) {label = Distribution(0, numClusters)(rSeedEngine);});

  std::vector<LocationType> means(numClusters);
  std::vector<size_t> meanCounter(numClusters, 0);

  //TODO clear means;


  size_t checksum = 0;
  size_t oldChecksum = 0;
  size_t counter = 0;

  while(1) {
    checksum = 0;

    std::for_each(means.begin(), means.end(), [](LocationType & loc) {loc.fill(0);});
    std::for_each(meanCounter.begin(), meanCounter.end(), [](size_t & counter) {counter = 0;});

    //add to mean
    std::for_each(first, last, labels.cbegin(), [](const LocationType & loc, size_t label) {
                  func.addTo(means[label],loc);
                  ++meanCounter[label];
                  });
    //get mean
    std::for_each(means.begin(), means.end(),meanCounter.cbegin(), [](LocationType & loc, size_t counter) {func.divideBy(loc, counter);});

    //adjust to new clusters
    std::for_each(first, last, labels.begin(), [](const LocationType & loc, size_t & label) {
                  label = func.findClosest(loc, means.cbegin(), means.cend());
                  });
    std::accumulate(labels.begin(), labels.end(), checksum);
    if (checksum == oldChecksum || counter == maxIterations) break;
    ++counter;
    oldChecksum = checksum;

  }
return labels;

}
template <typename RandomAccessIterator, typename RandomAccessIterator>
std::vector<size_t>
kmeansWithSubset(ForwardIterator first, ForwardIterator last, size_t numClusters, size_t maxIterations, LocationFunctor && func = ArrayFunctor<typename ForwardIterator::value_type>()) {
  size_t vecSize = std::distance(last, first);
  typedef typename ForwardIterator::value_type LocationType;
  std::vector<typename ForwardIterator::value_type> subset(sqrt(vecSize));
  std::generate(subset.begin(), subset.end(), [&]() {return *(first + Distribution(0,vecSize)(rSeedEngine));});
}



}
