#ifndef _BELIEF_NODE_DATA_HPP_
#define _BELIEF_NODE_DATA_HPP_
#include "TreeElementData.hpp"
#include <oppt/opptCore/core.hpp>
#include "Episode.hpp"

namespace oppt {
class BeliefNodeData: public TreeElementData {
public:
	BeliefNodeData(TreeElement *const parentElement, RobotEnvironment *robotEnvironment);

	virtual ~BeliefNodeData() = default;

	void setParticles(VectorRobotStatePtr &particles);

	VectorRobotStatePtr getParticles() const;

	size_t getNumParticles() const;

	RobotStateSharedPtr sampleParticle() const;

	void addEpisode(EpisodePtr episode);	

protected:
	VectorRobotStatePtr particles_;

	RobotEnvironment *robotEnvironment_;

	std::unique_ptr<std::uniform_int_distribution<unsigned int>> unformParticleDistribution_;

	std::vector<EpisodePtr> episodes_;

protected:
	void initBeliefNode_(TreeElement *const belief) const;
};
}

#endif