#ifndef _ABT_OBSERVATION_EDGE_HPP_
#define _ABT_OBSERVATION_EDGE_HPP_
#include "TreeElement.hpp"
#include <oppt/opptCore/core.hpp>

namespace oppt {
class ABTObservationEdge: public TreeElement {
public:
	ABTObservationEdge(TreeElement *const parentElement, const ObservationSharedPtr &observation);

	virtual ~ABTObservationEdge() = default;

	virtual void print() const override;

	const Observation *getObservation() const;

	template <typename NodeType>
	TreeElement *const createOrGetChild()
	{
		if (children_.size() == 1)
			return children_[0].get();

		std::unique_ptr<TreeElement> childBelief(new NodeType(this));
		return addChild(std::move(childBelief));
	}

protected:
	const ObservationSharedPtr observation_;

};
}

#endif