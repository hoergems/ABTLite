#include "Solver.hpp"
#include "ABTActionEdge.hpp"
#include "ActionEdgeData.hpp"
#include "BeliefNodeData.hpp"
#include "ABTBeliefNode.hpp"
#include "ABTOptions.hpp"
#include <oppt/robotHeaders/ActionSpaceDiscretizer.hpp>

namespace oppt {
extern ObservationComparator observationComparator;
}

namespace solvers {
ABTLite::ABTLite():
	Solver(),
	beliefTree_(nullptr) {
	solverName_ = "ABTLite";
}

void ABTLite::setup() {
	reset();

	auto options = static_cast<const ABTLiteOptions *>(problemEnvironmentOptions_);
	FloatType maxObservationDistance = options->maxObservationDistance;

	// A function to determine whether two observations are equal.
	observationComparator_ = [maxObservationDistance](const Observation * observation, TreeElement * treeElement) {
		auto it = treeElement->getChildren();
		unsigned int numChildren = treeElement->getNumChildren();
		unsigned int idx = 0;
		FloatType smallestDistance = maxObservationDistance;
		TreeElement *closestObservationEdge = nullptr;
		while (true) {
			idx++;
			if (idx > numChildren)
				break;
			FloatType dist = (*it)->as<ABTObservationEdge>()->getObservation()->distanceTo(*observation);
			if (dist < smallestDistance) {
				smallestDistance = dist;
				closestObservationEdge = (*it).get();
			}

			it++;
		}

		return closestObservationEdge;
	};

	particleFilter_ = std::make_unique<ParticleFilter>();
}

bool ABTLite::reset() {
	beliefTree_ = std::make_unique<Tree>();
	beliefTree_->initRoot<ABTBeliefNode>();

	TreeElement *const root = beliefTree_->getRoot();

	TreeElementDataPtr rootData(new BeliefNodeData(root, robotPlanningEnvironment_));
	root->setData(std::move(rootData));

	oppt::ActionSpaceSharedPtr actionSpace = robotPlanningEnvironment_->getRobot()->getActionSpace();
	auto options_ = static_cast<const ABTLiteOptions *>(problemEnvironmentOptions_);

	auto actionSpaceDiscretizer = actionSpace->getActionSpaceDiscretizer();
	if (!actionSpaceDiscretizer) {
		actionSpaceDiscretizer = std::make_shared<oppt::ActionSpaceDiscretizer>(actionSpace);
		if (options_->actionDiscretization.size() > 0) {
			actionSpaceDiscretizer = std::make_unique<CustomActionSpaceDiscretizer>(actionSpace, options_->actionDiscretization);
		}
	}

	allActions_ = actionSpaceDiscretizer->getAllActionsInOrder(options_->numInputStepsActions);
	initBeliefNode_(root);
	return true;
}

bool ABTLite::improvePolicy(const FloatType &timeout) {
	cout << "PLANNING FROM:" << endl;
	beliefTree_->getRoot()->print();

	size_t numSampledEpisodes = 0;
	unsigned long maxNumEpisodes = static_cast<const ABTLiteOptions *>(problemEnvironmentOptions_)->historiesPerStep;
	FloatType endTime = oppt::clock_ms() + timeout;
	while (true) {
		//sampleHistory_();
		EpisodePtr episode = std::move(sampleEpisode_());
		backupEpisode_(episode.get());
		beliefTree_->getRoot()->getData()->as<BeliefNodeData>()->addEpisode(std::move(episode));
		numSampledEpisodes++;
		if (maxNumEpisodes > 0) {
			if (numSampledEpisodes == maxNumEpisodes)
				break;
		} else {
			if (oppt::clock_ms() >= endTime)
				break;
		}
	}

	cout << "Sampled " << numSampledEpisodes << " episodes" << endl;

	return true;
}

EpisodePtr ABTLite::sampleEpisode_() {
	auto options = static_cast<const ABTLiteOptions *>(problemEnvironmentOptions_);
	TreeElement *currentBelief = beliefTree_->getRoot();
	PropagationRequestSharedPtr propagationRequest(new PropagationRequest);
	ObservationRequestSharedPtr observationRequest(new ObservationRequest);
	PropagationResultSharedPtr propRes = nullptr;
	ObservationResultSharedPtr obsRes = nullptr;
	FloatType reward = 0;
	bool terminal = false;
	std::shared_ptr<HeuristicInfo> heuristicInfo(new HeuristicInfo);
	EpisodePtr episode(new Episode);

	// Sample a state from the current belief
	RobotStateSharedPtr state = currentBelief->as<ABTBeliefNode>()->sampleParticle();
	
	auto randomEngine = robotPlanningEnvironment_->getRobot()->getRandomEngine().get();

	while (true) {
		if (!currentBelief)
			ERROR("No belief");
		auto action = currentBelief->as<ABTBeliefNode>()->getUCBAction(options->ucbExplorationFactor);
		if (!action) {
			ERROR("No action")
		}

		// Sample next state
		propagationRequest->currentState = state;
		propagationRequest->action = action;
		propRes = robotPlanningEnvironment_->getRobot()->propagateState(propagationRequest);

		// Sample observation
		observationRequest->currentState = propRes->nextState;
		observationRequest->action = action;
		obsRes = robotPlanningEnvironment_->getRobot()->makeObservationReport(observationRequest);

		// Get reward
		reward = robotPlanningEnvironment_->getReward(propRes);

		// Check if we're terminal
		terminal = robotPlanningEnvironment_->isTerminal(propRes);

		episode->addEpisodeEntry(currentBelief, state, action, obsRes->observation, reward, terminal);
		state = propRes->nextState;

		// Go to the next belief		
		currentBelief = currentBelief->as<ABTBeliefNode>()->getOrCreateChild<ABTBeliefNode>(action, obsRes->observation, randomEngine);
		if (terminal) {
			episode->addEpisodeEntry(currentBelief, nullptr, nullptr, nullptr, 0.0, terminal);
			break;
		}

		// Check if the belief is new
		if (currentBelief->getData() == nullptr) {
			TreeElementDataPtr beliefNodeData(new BeliefNodeData(currentBelief, robotPlanningEnvironment_));
			currentBelief->setData(std::move(beliefNodeData));

			// Make the outgoing edges for the new belief
			initBeliefNode_(currentBelief);

			// If the belief is new, get a heuristic estimate
			heuristicInfo->currentState = state;
			heuristicInfo->action = action;
			heuristicInfo->discountFactor = problemEnvironmentOptions_->discountFactor;
			heuristicInfo->timeout = static_cast<const ABTLiteOptions *>(problemEnvironmentOptions_)->heuristicTimeout;
			FloatType heuristic = heuristicPlugin_->getHeuristicValue(heuristicInfo.get());
			episode->addEpisodeEntry(currentBelief, nullptr, nullptr, nullptr, heuristic, false);
			break;
		}
	}

	return std::move(episode);
}

void ABTLite::backupEpisode_(Episode *episode) {
	auto it = episode->getReverseIterator();
	FloatType qValue = (*it)->reward_;
	auto beliefNode = (*it)->beliefNode_;
	it++;
	while (true) {
		qValue = (*it)->reward_ + problemEnvironmentOptions_->discountFactor * qValue;
		beliefNode->getParent()->getParent()->getData()->as<ActionEdgeData>()->update(1, qValue, +1);
		beliefNode = beliefNode->getParent()->getParent()->getParent();
		beliefNode->as<ABTBeliefNode>()->recalculateValue();
		qValue = beliefNode->as<ABTBeliefNode>()->getCachedValue();
		if (beliefNode->getParent() == nullptr)
			break;
		it++;
	}
}

ActionSharedPtr ABTLite::getNextAction() {
	cout << "GETTING ACTION FROM:" << endl;
	beliefTree_->getRoot()->print();
	const oppt::Action *bestAction =
	    beliefTree_->getRoot()->as<ABTBeliefNode>()->getBestAction();
	for (auto &action : allActions_) {
		if (action.get() == bestAction)
			return action;
	}

	WARNING("NO BEST ACTION FOUND");
	return nullptr;
}

bool ABTLite::updateBelief(const ActionSharedPtr& action,
                           const ObservationSharedPtr& observation,
                           const bool &allowTerminalStates) {
	auto options = static_cast<const ABTLiteOptions *>(problemEnvironmentOptions_);
	auto randomEngine = robotPlanningEnvironment_->getRobot()->getRandomEngine();
	FilterRequestPtr filterRequest = std::make_unique<FilterRequest>();
	filterRequest->robotEnvironment = robotPlanningEnvironment_;
	filterRequest->allowZeroWeightParticles = false;
	filterRequest->numParticles = options->minParticleCount;
	filterRequest->action = action.get();
	filterRequest->observation = observation.get();
	filterRequest->allowCollisions = robotPlanningEnvironment_->getRobot()->getCollisionsAllowed();
	filterRequest->randomEngine = randomEngine;

	for (size_t i = 0; i != options->minParticleCount; ++i) {
		auto state = beliefTree_->getRoot()->as<ABTBeliefNode>()->sampleParticle();
		filterRequest->previousParticles.push_back(std::make_shared<Particle>(state, 1.0));
	}

	FilterResultPtr filterResult = particleFilter_->filter(filterRequest);
	if (filterResult->particles.empty())
		return false;

	VectorRobotStatePtr nextParticles(filterResult->particles.size(), nullptr);
	for (size_t i = 0; i != filterResult->particles.size(); ++i) {
		nextParticles[i] = filterResult->particles[i]->getState();
	}

	TreeElement *const childBelief =
	    beliefTree_->getRoot()->as<ABTBeliefNode>()->getOrCreateChild<ABTBeliefNode>(action.get(),
	            observation,
	            randomEngine.get());
	TreeElementPtr newBelief = std::move(childBelief->getParent()->releaseChild(childBelief));
	if (newBelief->getData() == nullptr or options->resetTree) {
		if (options->resetTree)
			LOGGING("RESETTING TREE");
		TreeElementDataPtr beliefNodeData(new BeliefNodeData(newBelief.get(), robotPlanningEnvironment_));
		newBelief->setData(std::move(beliefNodeData));
		initBeliefNode_(newBelief.get());
	}

	newBelief->getData()->as<BeliefNodeData>()->setParticles(nextParticles);
	beliefTree_->updateRoot(std::move(newBelief));
	return true;
}

void ABTLite::initBeliefNode_(TreeElement *const belief) {
	if (belief->getNumChildren() != 0)
		belief->removeSubtrees();
	for (size_t i = 0; i != allActions_.size(); ++i) {
		TreeElementPtr actionEdge(new ABTActionEdge(belief, allActions_[i].get(), robotPlanningEnvironment_->getRobot()->getRandomEngine().get()));
		TreeElement *const actionEdgePtr = belief->addChild(std::move(actionEdge));
		TreeElementDataPtr actionData(new ActionEdgeData(actionEdgePtr));
		actionEdgePtr->setData(std::move(actionData));
		actionEdgePtr->as<ABTActionEdge>()->setObservationComparator(observationComparator_);
	}

	belief->as<ABTBeliefNode>()->initActionSequence(robotPlanningEnvironment_->getRobot()->getRandomEngine().get());
	belief->as<ABTBeliefNode>()->updateVisitCount(-(belief->as<ABTBeliefNode>()->getTotalVisitCount()));
	belief->as<ABTBeliefNode>()->recalculateValue();
}

VectorRobotStatePtr ABTLite::getBeliefParticles() {
	return beliefTree_->getRoot()->getData()->as<BeliefNodeData>()->getParticles();
}

void ABTLite::stepFinished(const size_t &step) {

}


}