#include "ActionEdgeData.hpp"
#include "ABTBeliefNode.hpp"

namespace oppt {
ActionEdgeData::ActionEdgeData(TreeElement *const parentElement):
	TreeElementData(parentElement),
	meanQ_(-std::numeric_limits<FloatType>::infinity()) {

}

void ActionEdgeData::update(const long &visitCount, const FloatType &discountedReward, const unsigned int &sgn) {
	if (numVisits_ + visitCount <= 0) {
		parentElement_->getParent()->as<ABTBeliefNode>()->updateVisitCount(-numVisits_);
		numVisits_ = 0;
		meanQ_ = -std::numeric_limits<FloatType>::infinity();
		totalQ_ = 0.0;
		return;
	}

	totalQ_ += ((FloatType)sgn) * discountedReward;
	numVisits_ += visitCount;
	parentElement_->getParent()->as<ABTBeliefNode>()->updateVisitCount(visitCount);
	meanQ_ = ((FloatType)(totalQ_)) / ((FloatType)(numVisits_));
}

void ActionEdgeData::reset() {
	parentElement_->getParent()->as<ABTBeliefNode>()->updateVisitCount(-numVisits_);
	numVisits_ = 0.0;
	meanQ_ = -std::numeric_limits<FloatType>::infinity();
	totalQ_ = 0.0;
}


FloatType ActionEdgeData::getTotalQ() const {
	return totalQ_;
}

FloatType ActionEdgeData::getMeanQ() const {
	return meanQ_;
}

long ActionEdgeData::getNumVisits() const {
	return numVisits_;
}


}