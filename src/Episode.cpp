#include "Episode.hpp"
#include "TreeElement.hpp"

namespace oppt {
EpisodeEntry::EpisodeEntry(TreeElement *const beliefNode,
                           const RobotStateSharedPtr &state,
                           const Action *action,
                           const ObservationSharedPtr &observation,
                           const FloatType &reward, 
                           const bool &terminal):
	beliefNode_(beliefNode),
	state_(state),
	action_(action),
	observation_(observation),
	reward_(reward),
	terminal_(terminal) {

}

EpisodeEntry *const Episode::addEpisodeEntry(TreeElement *const beliefNode,
        const RobotStateSharedPtr &state,
        const Action *action,
        const ObservationSharedPtr &observation,
        const FloatType &reward, 
        const bool &terminal) {
	EpisodeEntryPtr episodeEntry = std::make_unique<EpisodeEntry>(beliefNode, state, action, observation, reward, terminal);
	EpisodeEntry *const episodeEntr = episodeEntry.get();
	sequence_.push_back(std::move(episodeEntry));
	return episodeEntr;
}

std::unique_ptr<Episode> Episode::copy() const {
	std::unique_ptr<Episode> copied(new Episode);
	for (auto &entry: sequence_) {
		copied->addEpisodeEntry(entry->beliefNode_, entry->state_, entry->action_, entry->observation_, entry->reward_, entry->terminal_);
	}

	return copied;
}

EpisodeEntry *Episode::getFirstEntry() const {
	if (sequence_.empty())
		return nullptr;
	return sequence_[0].get();
}


EpisodeEntry *Episode::getLastEntry() const {
	if (sequence_.empty())
		return nullptr;
	return sequence_[sequence_.size() - 1].get();
}

EpisodeIterator Episode::begin() const {
	return sequence_.begin();
}

EpisodeIterator Episode::end() const {
	return sequence_.end();
}

EpisodeReverseIterator Episode::getReverseIterator() const {
	return sequence_.crbegin();
}

EpisodeReverseIterator Episode::crend() const {
	return sequence_.crend();
}

size_t Episode::getLength() const {
	return sequence_.size();
}



}