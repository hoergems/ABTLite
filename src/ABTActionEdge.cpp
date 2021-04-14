#include "ABTActionEdge.hpp"
#include "ABTObservationEdge.hpp"
#include <oppt/robotHeaders/Observation.hpp>

namespace oppt {
ABTActionEdge::ABTActionEdge(TreeElement *const parentNode, const Action *action, RandomEngine *const randomEngine):
	TreeElement(parentNode),
	randomEngine_(randomEngine),
	action_(action) {

}

void ABTActionEdge::print() const {

}

const Action * ABTActionEdge::getAction() const {
	return action_;
}

TreeElement *const ABTActionEdge::getOrCreateObservationEdge(const ObservationSharedPtr &observation) {
	TreeElement *closestObservationEdge = observationComparator_(observation.get(), this);
	if (closestObservationEdge)
		return closestObservationEdge;

	std::unique_ptr<TreeElement> observationEdge(new ABTObservationEdge(this, observation));
	auto observationEdgeAdd = addChild(std::move(observationEdge));
	return observationEdgeAdd;
}

void ABTActionEdge::setObservationComparator(ObservationComparator observationComparator) {
	observationComparator_ = observationComparator;
}

}