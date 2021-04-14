cmake_minimum_required (VERSION 3.5.1)
project (ABTLite)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo) 
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

include(GNUInstallDirs)
if(DEFINED ENV{OPPT_SEARCH_PATH})
   message("-- searching for oppt in $ENV{OPPT_SEARCH_PATH}")
   find_package(oppt REQUIRED PATHS $ENV{OPPT_SEARCH_PATH} NO_DEFAULT_PATH)
   message("${oppt_LIBRARY_DIRS}")
else()
   message("-- OPPT_SEARCH_PATH not set. Searching in default locations")
   find_package(oppt REQUIRED)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/thirdparty/OPPTTree/CMakeLists.txt)
add_executable(abtLite
               ${CMAKE_CURRENT_LIST_DIR}/main.cpp
               ${CMAKE_CURRENT_LIST_DIR}/src/Solver.cpp          
               ${CMAKE_CURRENT_LIST_DIR}/src/ActionEdgeData.cpp
               ${CMAKE_CURRENT_LIST_DIR}/src/BeliefNodeData.cpp
               ${CMAKE_CURRENT_LIST_DIR}/src/ABTBeliefNode.cpp
               ${CMAKE_CURRENT_LIST_DIR}/src/ABTActionEdge.cpp
               ${CMAKE_CURRENT_LIST_DIR}/src/ABTObservationEdge.cpp
               ${CMAKE_CURRENT_LIST_DIR}/src/Episode.cpp
               ${OPPT_TREE_SRC})
target_include_directories(abtLite 
	PUBLIC 
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/OPPTTree/include>
	PRIVATE
		oppt)
target_link_libraries(abtLite PUBLIC oppt)
target_compile_features(abtLite PUBLIC cxx_std_11)
set_target_properties(abtLite
    PROPERTIES    
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin"
)
