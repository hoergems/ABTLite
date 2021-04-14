#ifndef _ABT_OPTIONS_HPP_
#define _ABT_OPTIONS_HPP_
#include <oppt/problemEnvironment/ProblemEnvironmentOptions.hpp>

namespace oppt
{
struct ABTLiteOptions: public oppt::ProblemEnvironmentOptions {
public:
    ABTLiteOptions() = default;
    virtual ~ABTLiteOptions() = default;

    /* Maximum amount of time ins seconds to compute the rollout heuristic */
    FloatType heuristicTimeout = 0.1;

    /* The particle filter to use */
    std::string particleFilter = "default";

    /** The minimum number of particles to maintain in the active belief node. */
    unsigned long minParticleCount = 1000;

    /** Allow zero weight particles to be part of the next belief */
    bool allowZeroWeightParticles = false;

    /** The number of new histories to generate on each search step. */
    unsigned long historiesPerStep = 1000;

    /** The maximum depth to search, relative to the current belief node. */
    long maximumDepth = 100;    

    /* ------------------------- ABT settings --------------------- */
    /** Save the policy when done **/
    bool savePolicy = false;

    /** True iff we should load an initial policy before running the simulation. */
    bool loadInitialPolicy = false;

    /** Path to the policy to load */
    std::string policyPath = "";

    bool resetTree = false;

    /* ---------- ABT settings: advanced customization  ---------- */
    /** The maximum distance between observations to group together; only applicable if
     * approximate observations are in use. */
    FloatType maxObservationDistance = 0.0;    

    /** 'discrete' or 'continuous' */
    std::string observationType = "continuous";    

    unsigned int numInputStepsActions = 3;

    std::vector<unsigned int> actionDiscretization = std::vector<unsigned int>();

    FloatType ucbExplorationFactor = 2.0;

    static std::unique_ptr<options::OptionParser> makeParser(bool simulating) {
        std::unique_ptr<options::OptionParser> parser =
            ProblemEnvironmentOptions::makeParser(simulating);
        addABTOptions(parser.get());
        return std::move(parser);
    }

    static void addABTOptions(options::OptionParser* parser) {
        parser->addOption<FloatType>("ABT", "heuristicTimeout", &ABTLiteOptions::heuristicTimeout);
        parser->addOption<std::string>("ABT", "particleFilter", &ABTLiteOptions::particleFilter);
        parser->addOptionWithDefault<unsigned long>("ABT", "minParticleCount",
                &ABTLiteOptions::minParticleCount, 1000);
        parser->addOptionWithDefault<bool>("ABT", "allowZeroWeightParticles",
                                           &ABTLiteOptions::allowZeroWeightParticles, false);
        parser->addOptionWithDefault<unsigned long>("ABT",
                "historiesPerStep",
                &ABTLiteOptions::historiesPerStep,
                0);
        parser->addOptionWithDefault<long>("ABT", "maximumDepth", &ABTLiteOptions::maximumDepth, 1000);
        parser->addOption<std::string>("ABT", "observationType", &ABTLiteOptions::observationType);
        parser->addOptionWithDefault<FloatType>("ABT", "maxObservationDistance",
                                                &ABTLiteOptions::maxObservationDistance, 0.0);        
        parser->addOptionWithDefault<unsigned int>("ABT", "numInputStepsActions",
                &ABTLiteOptions::numInputStepsActions, 3);
        std::vector<unsigned int> defaultUIntVec;
        parser->addOptionWithDefault<std::vector<unsigned int>>("ABT",
                "actionDiscretization",
                &ABTLiteOptions::actionDiscretization, defaultUIntVec);
        parser->addOptionWithDefault<bool>("ABT",
                                           "savePolicy",
                                           &ABTLiteOptions::savePolicy,
                                           false);
        parser->addOptionWithDefault<bool>("ABT",
                                           "loadInitialPolicy",
                                           &ABTLiteOptions::loadInitialPolicy,
                                           false);
        parser->addOptionWithDefault<std::string>("ABT",
                "policyPath",
                &ABTLiteOptions::policyPath,
                "pol.pol");
        parser->addOption<FloatType>("ABT", "ucbExplorationFactor", &ABTLiteOptions::ucbExplorationFactor);
        parser->addOptionWithDefault<bool>("ABT", "resetTree", &ABTLiteOptions::resetTree, false);
    }    
};
}

#endif
