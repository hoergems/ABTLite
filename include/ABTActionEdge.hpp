#ifndef _ABT_ACTION_EDGE_HPP_
#define _ABT_ACTION_EDGE_HPP_
#include "TreeElement.hpp"
#include "ObservationComparator.hpp"

namespace oppt {
class ABTActionEdge: public TreeElement {
public:
	ABTActionEdge(TreeElement *const parentNode, const Action *action, RandomEngine *const randomEngine);

	virtual ~ABTActionEdge() = default;

	virtual void print() const override;	

	const Action * getAction() const;

	virtual TreeElement *const getOrCreateObservationEdge(const ObservationSharedPtr &observation);

	void setObservationComparator(ObservationComparator observationComparator);

protected:
	ObservationComparator observationComparator_;	

	const Action *action_;

private:
	RandomEngine *randomEngine_ = nullptr;

};

}

#endif