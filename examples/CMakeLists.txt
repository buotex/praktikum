INCLUDE_DIRECTORIES(
${GEDLIB_SOURCE_DIR}/include
)



add_executable(comp comparison.cpp)

target_link_libraries(comp ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} armadillo )
