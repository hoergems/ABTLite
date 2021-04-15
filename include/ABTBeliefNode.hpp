#ifndef _ABT_BELIEF_NODE_HPP_
#define _ABT_BELIEF_NODE_HPP_
#include "TreeElement.hpp"
#include <oppt/opptCore/core.hpp>
#include "ABTActionEdge.hpp"
#include "ABTObservationEdge.hpp"

namespace oppt {

class ABTBeliefNode: public TreeElement {
public:
	ABTBeliefNode(TreeElement *const parentElement);

	virtual ~ABTBeliefNode() = default;

	virtual void print() const override;

	/**
	 * Sample a particle from this belief
	 */
	RobotStateSharedPtr sampleParticle() const;	

	/**
	 * Get the action with the largest Q-value from this belief
	 */
	const Action* getBestAction() const;

	/**
	 * Recalculate the value of this belief according to V(b) = max_a Q(b, a)
	 */
	FloatType recalculateValue();

	/**
	 * Get the estimated value V(b) of this belief
	 */
	FloatType getCachedValue();

	/**
	 * Get the total number of times we have visited this belief during the episode sampling process
	 */
	long getTotalVisitCount() const;

	/**
	 * Update the visitation count
	 */
	void updateVisitCount(const long &visitCount);

	/**
	 * Initialize a random sequence of actions to be selected from this belief (via getUCBAction)
	 * during the episode sampling process. Once all actions have been selected at least once,
	 * the actions are selected according to UCB1.
	 */
	void initActionSequence(RandomEngine *randomEngine);

	/**
	 * @brief Get or create the child node given an action and observation. If no such child exists, create one.
	 */
	template<typename NodeType>
	TreeElement *const getOrCreateChild(const Action *action, const ObservationSharedPtr &observation, RandomEngine *const randomEngine) {
		TreeElement *childActionEdge = nullptr;
		for (auto it = getChildren(); it != children_.end(); it++) {
			if ((*it)->as<ABTActionEdge>()->getAction()->equals(*action))
				childActionEdge = (*it).get();
		}

		if (!childActionEdge) {
			std::unique_ptr<TreeElement> actionEdge(new ABTActionEdge(this, action, randomEngine));
			childActionEdge = addChild(std::move(actionEdge));
		}		

		TreeElement *const observationEdge = childActionEdge->as<ABTActionEdge>()->getOrCreateObservationEdge(observation);
		return observationEdge->as<ABTObservationEdge>()->createOrGetChild<NodeType>();
	}

	/**
	 * Select an action from this belief according to UCB1
	 */
	const Action *getUCBAction(const FloatType &explorationFactor);

protected:
	FloatType cachedValue_ = 0.0;

	long totalVisitCount_ = 0;

	std::vector<unsigned int> actionSequence_;
};
}

#endif