#include "ABTBeliefNode.hpp"
#include "BeliefNodeData.hpp"
#include "ActionEdgeData.hpp"
#include "ABTActionEdge.hpp"

namespace oppt {
ABTBeliefNode::ABTBeliefNode(TreeElement *const parentElement):
	TreeElement(parentElement) {

}

void ABTBeliefNode::print() const {
	std::vector<std::pair<FloatType, std::string>> actionStats;
	for (size_t i = 0; i != getNumChildren(); ++i) {
		auto actionEdge = children_[i]->as<ABTActionEdge>();
		std::stringstream s;
		s << *(actionEdge->getAction());
		std::string actionString = "action " + s.str();
		actionString += ", numVisits=" + std::to_string(actionEdge->getData()->as<ActionEdgeData>()->getNumVisits());
		actionString += ", meanQ=" + std::to_string(actionEdge->getData()->as<ActionEdgeData>()->getMeanQ());
		actionString += ", numObservations: " + std::to_string(actionEdge->getNumChildren());

		std::pair<FloatType, std::string> stat = {actionEdge->getData()->as<ActionEdgeData>()->getMeanQ(), actionString}; 
		actionStats.push_back(stat);		
	}

	std::sort(actionStats.begin(), actionStats.end(), [](const std::pair<FloatType, std::string> &p1, const std::pair<FloatType, std::string> &p2) {
		return p1.first > p2.first;
	});

	for (auto &stat : actionStats) {
		cout << stat.second << endl;
	}
}

RobotStateSharedPtr ABTBeliefNode::sampleParticle() const {
	return static_cast<BeliefNodeData *>(treeElementData_.get())->sampleParticle();
}

const Action * ABTBeliefNode::getBestAction() const {
	FloatType bestMeanQ = -std::numeric_limits<FloatType>::infinity();
	const Action *bestAction = nullptr;
	for (auto &child : children_) {
		FloatType meanQ = child->getData()->as<ActionEdgeData>()->getMeanQ();
		if (meanQ > bestMeanQ) {
			bestMeanQ = meanQ;
			bestAction = child->as<ABTActionEdge>()->getAction();
		}
	}

	return bestAction;
}

void ABTBeliefNode::initActionSequence(RandomEngine *randomEngine) {
	actionSequence_ = std::vector<unsigned int>(getNumChildren(), 0);
	for (size_t i = 0; i != actionSequence_.size(); ++i) {
		actionSequence_[i] = i;
	}

	std::shuffle(actionSequence_.begin(), actionSequence_.end(), *randomEngine);
}

const Action *ABTBeliefNode::getUCBAction(const FloatType &explorationFactor) {
	if (actionSequence_.size() != 0) {
		const Action *untriedAction = children_[actionSequence_.back()]->as<ABTActionEdge>()->getAction();
		actionSequence_.pop_back();
		return untriedAction;
	}
	FloatType bestUCBScore = -std::numeric_limits<FloatType>::infinity();
	const Action *bestAction = nullptr;
	std::vector<const Action *> untriedActions;
	for (auto &child : children_) {
		FloatType numVisits = child->getData()->as<ActionEdgeData>()->getNumVisits();
		FloatType meanQValue = child->getData()->as<ActionEdgeData>()->getMeanQ();
		FloatType tmpValue = meanQValue + explorationFactor * std::sqrt(std::log(totalVisitCount_) / numVisits);
        if (tmpValue > bestUCBScore) {
        	bestUCBScore = tmpValue;
        	bestAction = child->as<ABTActionEdge>()->getAction();
        }

	}

	if (untriedActions.size() > 0) {
		//action = 
	}

	if (!bestAction)
		WARNING("Retuning null action");

	return bestAction;
}

FloatType ABTBeliefNode::recalculateValue() {
	FloatType maxQValue = -std::numeric_limits<FloatType>::infinity();
	for (auto &child : children_) {
		FloatType meanQChild = child->getData()->as<ActionEdgeData>()->getMeanQ();
		if (meanQChild > maxQValue)
			maxQValue = meanQChild;
	}

	cachedValue_ = maxQValue;
	return cachedValue_;
}

FloatType ABTBeliefNode::getCachedValue() {
	if (totalVisitCount_ <= 0)
		return 0.0;
	return cachedValue_;
}

long ABTBeliefNode::getTotalVisitCount() const {
	return totalVisitCount_;
}

void ABTBeliefNode::updateVisitCount(const long &visitCount) {
	totalVisitCount_ += visitCount;
	if (totalVisitCount_ < 0)
		totalVisitCount_ = 0;
}



}