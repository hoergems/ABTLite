#include <oppt/problemEnvironment/ProblemEnvironment.hpp>
#include "Solver.hpp"
#include "ABTOptions.hpp"

int main(int argc, char const* argv[]) {
	oppt::ProblemEnvironment p;
	p.setup<solvers::ABTLite, oppt::ABTLiteOptions>(argc, argv);
	p.runEnvironment(argc, argv);
	return 0;
}