#ifndef _ABT_LITE_HISTORY_HPP_
#define _ABT_LITE_HISTORY_HPP_
#include <oppt/opptCore/core.hpp>

namespace oppt {
class TreeElement;

struct EpisodeEntry {
	EpisodeEntry(TreeElement *const beliefNode,
	             const RobotStateSharedPtr &state,
	             const Action *action,
	             const ObservationSharedPtr &observation,
	             const FloatType &reward,
	             const bool &terminal);

	virtual ~EpisodeEntry() = default;

	TreeElement *beliefNode_;

	const RobotStateSharedPtr state_;

	const Action *action_;

	const ObservationSharedPtr observation_;

	const FloatType reward_;

	const bool terminal_;
};

typedef std::unique_ptr<EpisodeEntry> EpisodeEntryPtr;
typedef std::vector<EpisodeEntryPtr> EpisodeVec;
typedef EpisodeVec::const_iterator EpisodeIterator;
typedef EpisodeVec::const_reverse_iterator EpisodeReverseIterator;

class Episode {
public:
	Episode() = default;

	virtual ~Episode() = default;

	EpisodeEntry *const addEpisodeEntry(TreeElement *const beliefNode,
	                                    const RobotStateSharedPtr &state,
	                                    const Action *action,
	                                    const ObservationSharedPtr &observation,
	                                    const FloatType &reward,
	                                    const bool &terminal);

	EpisodeEntry *getFirstEntry() const;

	EpisodeEntry *getLastEntry() const;

	EpisodeIterator begin() const;

	EpisodeIterator end() const;

	EpisodeReverseIterator getReverseIterator() const;

	EpisodeReverseIterator crend() const;

	size_t getLength() const;

	std::unique_ptr<Episode> copy() const;

private:
	EpisodeVec sequence_;
};

typedef std::unique_ptr<Episode> EpisodePtr;
}

#endif