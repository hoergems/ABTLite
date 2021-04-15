#ifndef _ABT_OPTIONS_HPP_
#define _ABT_OPTIONS_HPP_
#include <oppt/problemEnvironment/ProblemEnvironmentOptions.hpp>

namespace oppt
{
struct ABTLiteOptions: public oppt::ProblemEnvironmentOptions {
public:
    ABTLiteOptions() = default;
    virtual ~ABTLiteOptions() = default;

    /* Maximum amount of time in seconds to compute the rollout heuristic. */
    FloatType heuristicTimeout = 0.1;    

    /** The minimum number of particles to maintain in the active belief node. */
    unsigned long minParticleCount = 1000;

    /** Allow zero weight particles to be part of the next belief */
    bool allowZeroWeightParticles = false;

    /** The maximum number of new episodes to sample on each search step (0 => wait for timeout). */
    unsigned long maxNumEpisodes = 1000;

    /** The maximum depth to search, relative to the current belief node. */
    long maximumDepth = 100;

    /** Reset the policy after every step */
    bool resetPolicy = false;

    /** The UCB exploration factor */
    FloatType ucbExplorationFactor = 2.0;

    /* ---------- ABT settings: advanced customization  ---------- */
    /** The maximum distance between observations to group together; only applicable if
     * approximate observations are in use. */
    FloatType maxObservationDistance = 0.0;    

    unsigned int numInputStepsActions = 3;

    std::vector<unsigned int> actionDiscretization = std::vector<unsigned int>();

    static std::unique_ptr<options::OptionParser> makeParser(bool simulating) {
        std::unique_ptr<options::OptionParser> parser =
            ProblemEnvironmentOptions::makeParser(simulating);
        addABTOptions(parser.get());
        return std::move(parser);
    }

    static void addABTOptions(options::OptionParser* parser) {
        parser->addOption<FloatType>("ABT", "heuristicTimeout", &ABTLiteOptions::heuristicTimeout);        
        parser->addOptionWithDefault<unsigned long>("ABT", "minParticleCount",
                &ABTLiteOptions::minParticleCount, 1000);
        parser->addOptionWithDefault<bool>("ABT", "allowZeroWeightParticles",
                                           &ABTLiteOptions::allowZeroWeightParticles, false);
        parser->addOptionWithDefault<unsigned long>("ABT",
                "maxNumEpisodes",
                &ABTLiteOptions::maxNumEpisodes,
                0);
        parser->addOptionWithDefault<long>("ABT", "maximumDepth", &ABTLiteOptions::maximumDepth, 1000);        
        parser->addOptionWithDefault<FloatType>("ABT", "maxObservationDistance",
                                                &ABTLiteOptions::maxObservationDistance, 0.0);        
        parser->addOptionWithDefault<unsigned int>("ABT", "numInputStepsActions",
                &ABTLiteOptions::numInputStepsActions, 3);
        std::vector<unsigned int> defaultUIntVec;
        parser->addOptionWithDefault<std::vector<unsigned int>>("ABT",
                "actionDiscretization",
                &ABTLiteOptions::actionDiscretization, defaultUIntVec);
        parser->addOption<FloatType>("ABT", "ucbExplorationFactor", &ABTLiteOptions::ucbExplorationFactor);
        parser->addOptionWithDefault<bool>("ABT", "resetPolicy", &ABTLiteOptions::resetPolicy, false);
    }    
};
}

#endif
