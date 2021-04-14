#include "TreeElementData.hpp"
#include <oppt/opptCore/typedefs.hpp>

namespace solvers {
class ABTLite;
}

namespace oppt {

class ActionEdgeData: public TreeElementData {
public:
	friend class solvers::ABTLite;
	ActionEdgeData(TreeElement *const parentElement);

	virtual ~ActionEdgeData() = default;

	virtual void reset() override;

	void update(const long &visitCount, const FloatType &discountedReward, const unsigned int &sgn);

	FloatType getTotalQ() const;

	FloatType getMeanQ() const;

	long getNumVisits() const;

private:
	long numVisits_ = 0;

	FloatType meanQ_;

	FloatType totalQ_ = 0.0;	
};
}