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

	RobotStateSharedPtr sampleParticle() const;

	const Action* getNextAction(const FloatType &explorationFactor) const;

	const Action* getBestAction() const;

	FloatType recalculateValue();

	FloatType getCachedValue();

	long getTotalVisitCount() const;

	void updateVisitCount(const long &visitCount);

	void initActionSequence(RandomEngine *randomEngine);

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

		//for (auto &arg: {args...}) {
		//	static_cast<const BeliefNode *>(arg);
		//}

		//for(auto &&arg: { args... }) {
		//}

		TreeElement *const observationEdge = childActionEdge->as<ABTActionEdge>()->getOrCreateObservationEdge(observation);
		return observationEdge->as<ABTObservationEdge>()->createOrGetChild<NodeType>();
	}

	const Action *getUCBAction(const FloatType &explorationFactor);

protected:
	FloatType cachedValue_ = 0.0;

	long totalVisitCount_ = 0;

	std::vector<unsigned int> actionSequence_;
};
}

#endif