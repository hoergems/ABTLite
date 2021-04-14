#ifndef _ABT_LITE_SOLVER_HPP_
#define _ABT_LITE_SOLVER_HPP_
#include <oppt/solver/solver.hpp>
#include "Tree.hpp"
#include "TreeElement.hpp"
#include "Episode.hpp"
#include <oppt/filter/particleFilter/ParticleFilter.hpp>
#include "ObservationComparator.hpp"


using namespace oppt;

namespace solvers {
class ABTLite: public Solver {
public:
	ABTLite();

	~ABTLite() = default;

	void setup() override;

	bool reset() override;

	bool improvePolicy(const FloatType &timeout) override;

	ActionSharedPtr getNextAction() override;

	bool updateBelief(const ActionSharedPtr& action,
	                  const ObservationSharedPtr& observation,
	                  const bool &allowTerminalStates = false) override;

	VectorRobotStatePtr getBeliefParticles() override;

	virtual void stepFinished(const size_t &step) override;

private:
	EpisodePtr sampleEpisode_();

	void backupEpisode_(Episode *historySequence);

	void initBeliefNode_(TreeElement *const belief);	

private:
	std::unique_ptr<Tree> beliefTree_;

	std::vector<ActionSharedPtr> allActions_;

	std::unique_ptr<ParticleFilter> particleFilter_ = nullptr;

	ObservationComparator observationComparator_;

};

}

#endif