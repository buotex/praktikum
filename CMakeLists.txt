PROJECT(GEDLIB)

# require at least cmake 2.6
cmake_minimum_required(VERSION 2.6)
# include macro definitions
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros/require_out_of_src_build.cmake)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

#############################################################################
# versioning
#############################################################################
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
SET(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}")

#############################################################################
# build libraries
#############################################################################

# require out-of-source build
MACRO_REQUIRE_OUT_OF_SRC_BUILD("GEDLIB requires out-of-source builds.")

SET(CMAKE_CXX0X_FLAGS "-std=c++0x" )
# more warnings against everything
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -Wall -Wno-sign-conversion" )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX0X_FLAGS}" )

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
    ENDIF(NOT CMAKE_BUILD_TYPE)


# add subdirs
OPTION(ENABLE_TESTING "Compile tests" ON)
OPTION(ENABLE_COVERAGE "Enable GCov coverage analysis (defines a 'coverage' target and enforces static build of ms++)" OFF)
OPTION(ENABLE_BENCHMARK "Do benchmarking and plotting" ON)
OPTION(ENABLE_EXAMPLES "Compile examples" ON)

IF (ENABLE_BENCHMARK OR ENABLE_EXAMPLES)
    # require boost
    SET(Boost_USE_STATIC_LIBS OFF)
    SET(Boost_USE_MULTITHREAD OFF)
    FIND_PACKAGE(Boost COMPONENTS program_options filesystem system REQUIRED)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
ENDIF (ENABLE_BENCHMARK OR ENABLE_EXAMPLES)

#############################################################################
# syncing docs
#############################################################################
ADD_CUSTOM_TARGET(syncdocs COMMAND ${GEDLIB_SOURCE_DIR}/scripts/sync-docs-gh-pages.sh ${GEDLIB_SOURCE_DIR} ${GEDLIB_BINARY_DIR})
ADD_DEPENDENCIES(syncdocs doc)

#############################################################################
# regression tests
#############################################################################
SET(UPDATE_TYPE git)
IF (ENABLE_TESTING)
  INCLUDE(CTest)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(test)
ENDIF (ENABLE_TESTING)

#############################################################################
# code coverage analysis
#############################################################################
IF (ENABLE_COVERAGE)
    SET (ENABLE_EXAMPLES ON)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    CONFIGURE_FILE(${GEDLIB_SOURCE_DIR}/cmake/scripts/coverage.sh.in
        ${GEDLIB_BINARY_DIR}/cmake/scripts/coverage.sh
        @ONLY IMMEDIATE
    )
    CONFIGURE_FILE(${GEDLIB_SOURCE_DIR}/cmake/scripts/unit_coverage.sh.in
        ${GEDLIB_BINARY_DIR}/cmake/scripts/unit_coverage.sh
        @ONLY IMMEDIATE
    )
    ADD_CUSTOM_TARGET(
        coverage COMMAND /bin/bash ${GEDLIB_BINARY_DIR}/cmake/scripts/coverage.sh
    )
ENDIF (ENABLE_COVERAGE)


############################################################################
# examples
############################################################################
IF (ENABLE_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
    FILE(COPY examples/data DESTINATION ${GEDLIB_BINARY_DIR})
ENDIF (ENABLE_EXAMPLES)



#############################################################################
# documentation
#############################################################################
  find_package(Doxygen)
if(DOXYGEN_FOUND)
# doxygen support

  CONFIGURE_FILE(${GEDLIB_SOURCE_DIR}/doc/Doxyfile.cmake
      ${GEDLIB_BINARY_DIR}/doc/Doxyfile
      @ONLY IMMEDIATE
      )

# target for doxygen
ADD_CUSTOM_TARGET(doc ${DOXYGEN_EXECUTABLE} ${GEDLIB_BINARY_DIR}/doc/Doxyfile WORKING_DIRECTORY ${GEDLIB_BINARY_DIR} 
COMMENT "Generating API documentation with Doxygen" VERBATIM
)

endif(DOXYGEN_FOUND)

INCLUDE(CPack)

