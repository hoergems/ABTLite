#include "ABTObservationEdge.hpp"

namespace oppt {
ABTObservationEdge::ABTObservationEdge(TreeElement *const parentElement, const ObservationSharedPtr &observation):
	TreeElement(parentElement),
	observation_(observation) {

}

void ABTObservationEdge::print() const {

}

const Observation *ABTObservationEdge::getObservation() const {
	return observation_.get();
}
}